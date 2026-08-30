from collections import Counter
import hashlib
import os
from typing import Optional
from fastapi import Depends, FastAPI, Header, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, joinedload
from .database import Base, engine, get_db
from .models import Application, Candidate, Job
from .schemas import ApplyIn, ApplicationOut, CandidateIn, CandidateOut, ChatIn, JobIn, JobOut, LoginIn, MatchIn, RegisterIn, StatusIn
from .rag_service import answer_job_question, generate_match_assistant_reply, index_job_embedding, remove_job_embedding, semantic_job_matches
from .services import extract_preferences, match_jobs, norm

app = FastAPI(title="AI Job Board API")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173"], allow_methods=["*"], allow_headers=["*"])

def candidate_or_404(db, id):
    obj = db.get(Candidate, id)
    if not obj: raise HTTPException(404, "Candidate not found")
    return obj
def job_or_404(db, id):
    obj = db.get(Job, id)
    if not obj: raise HTTPException(404, "Job not found")
    return obj

def hash_password(password: str) -> str:
    salt = os.urandom(16).hex()
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 200_000).hex()
    return f"{salt}${digest}"

def password_matches(password: str, stored: Optional[str]) -> bool:
    if not stored or "$" not in stored: return False
    salt, digest = stored.split("$", 1)
    test = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 200_000).hex()
    return digest == test

def current_user(x_user_id: Optional[int] = Header(default=None), db: Session = Depends(get_db)):
    if not x_user_id: raise HTTPException(401, "Please log in")
    return candidate_or_404(db, x_user_id)

def admin_user(user: Candidate = Depends(current_user)):
    if user.role != "admin": raise HTTPException(403, "Admin access required")
    return user

@app.on_event("startup")
def startup():
    Base.metadata.create_all(engine)
    # Lightweight migration so existing SQLite databases gain account fields.
    with engine.begin() as connection:
        columns = {row[1] for row in connection.exec_driver_sql("PRAGMA table_info(candidates)")}
        if "role" not in columns: connection.exec_driver_sql("ALTER TABLE candidates ADD COLUMN role VARCHAR(16) DEFAULT 'candidate'")
        if "password_hash" not in columns: connection.exec_driver_sql("ALTER TABLE candidates ADD COLUMN password_hash VARCHAR(256)")
    with Session(engine) as db:
        if not db.scalar(select(Job.id).limit(1)):
            db.add_all([Job(title="Backend Engineer", description="Build healthcare platform APIs with a fast-moving startup team.", required_skills=["Python", "FastAPI", "PostgreSQL", "Docker"], experience_level="0-2 years", location="Bangalore", status="open"), Job(title="Frontend Engineer", description="Create delightful financial dashboards using React.", required_skills=["React", "TypeScript", "CSS"], experience_level="2-4 years", location="Remote", status="open"), Job(title="Data Engineer", description="Design data pipelines for AI products.", required_skills=["Python", "SQL", "AWS"], experience_level="3-5 years", location="Hyderabad", status="open")]); db.commit()

@app.post("/api/auth/register", response_model=CandidateOut, status_code=201)
def register(data: RegisterIn, db: Session = Depends(get_db)):
    values = data.model_dump(exclude={"password"})
    values["name"] = data.name or data.email.split("@", 1)[0]
    obj = Candidate(**values, password_hash=hash_password(data.password))
    db.add(obj)
    try: db.commit()
    except IntegrityError: db.rollback(); raise HTTPException(409, "An account with this email already exists")
    db.refresh(obj); return obj

@app.post("/api/auth/login", response_model=CandidateOut)
def login(data: LoginIn, db: Session = Depends(get_db)):
    user = db.scalar(select(Candidate).where(Candidate.email == data.email))
    if not user or not password_matches(data.password, user.password_hash):
        raise HTTPException(401, "Invalid email or password")
    return user

@app.post("/api/profiles", response_model=CandidateOut, status_code=201)
def create_profile(data: CandidateIn, db: Session = Depends(get_db)):
    obj = Candidate(**data.model_dump()); db.add(obj)
    try: db.commit()
    except IntegrityError: db.rollback(); raise HTTPException(409, "Email already exists")
    db.refresh(obj); return obj
@app.get("/api/profiles/{candidate_id}", response_model=CandidateOut)
def get_profile(candidate_id: int, db: Session = Depends(get_db)): return candidate_or_404(db, candidate_id)
@app.put("/api/profiles/{candidate_id}", response_model=CandidateOut)
def update_profile(candidate_id: int, data: CandidateIn, db: Session = Depends(get_db)):
    obj = candidate_or_404(db, candidate_id)
    for k,v in data.model_dump().items(): setattr(obj,k,v)
    db.commit(); db.refresh(obj); return obj

@app.get("/api/jobs", response_model=list[JobOut])
def list_jobs(skill: Optional[str] = None, location: Optional[str] = None, experience_level: Optional[str] = None, db: Session = Depends(get_db)):
    jobs = db.scalars(select(Job).order_by(Job.created_at.desc())).all()
    return [j for j in jobs if (not skill or norm(skill) in {norm(x) for x in j.required_skills}) and (not location or norm(location) == norm(j.location)) and (not experience_level or experience_level.lower() in j.experience_level.lower())]
@app.get("/api/jobs/{job_id}", response_model=JobOut)
def get_job(job_id: int, db: Session = Depends(get_db)): return job_or_404(db, job_id)
@app.post("/api/jobs", response_model=JobOut, status_code=201)
def create_job(data: JobIn, db: Session = Depends(get_db), _: Candidate = Depends(admin_user)):
    obj = Job(**data.model_dump()); db.add(obj); db.commit(); db.refresh(obj)
    # Admin job publishing immediately creates the persistent Chroma embedding.
    index_job_embedding(obj); return obj
@app.put("/api/jobs/{job_id}", response_model=JobOut)
def update_job(job_id: int, data: JobIn, db: Session = Depends(get_db), _: Candidate = Depends(admin_user)):
    obj = job_or_404(db, job_id)
    for k,v in data.model_dump().items(): setattr(obj,k,v)
    db.commit(); db.refresh(obj); index_job_embedding(obj); return obj
@app.patch("/api/jobs/{job_id}/status", response_model=JobOut)
def set_job_status(job_id: int, data: StatusIn, db: Session = Depends(get_db), _: Candidate = Depends(admin_user)):
    if data.status not in ["open", "closed"]: raise HTTPException(422, "Status must be open or closed")
    obj=job_or_404(db, job_id); obj.status=data.status; db.commit(); db.refresh(obj); index_job_embedding(obj); return obj
@app.delete("/api/jobs/{job_id}", status_code=204)
def delete_job(job_id: int, db: Session = Depends(get_db), _: Candidate = Depends(admin_user)):
    obj = job_or_404(db,job_id); remove_job_embedding(obj.id); db.delete(obj); db.commit()

def application_out(a): return ApplicationOut(id=a.id,candidate_id=a.candidate_id,job_id=a.job_id,status=a.status,applied_at=a.applied_at,candidate_name=a.candidate.name,job_title=a.job.title,candidate_skills=a.candidate.skills or [],candidate_education=a.candidate.education,candidate_projects=a.candidate.project_summaries or [],candidate_preferred_role=a.candidate.preferred_role_type,candidate_preferred_location=a.candidate.preferred_location)
@app.post("/api/applications", response_model=ApplicationOut, status_code=201)
def apply(data: ApplyIn, db: Session = Depends(get_db), user: Candidate = Depends(current_user)):
    if user.role != "candidate" or user.id != data.candidate_id: raise HTTPException(403, "Candidates can only apply for themselves")
    candidate_or_404(db,data.candidate_id); job=job_or_404(db,data.job_id)
    if job.status != "open": raise HTTPException(400, "This job is closed")
    obj=Application(**data.model_dump()); db.add(obj)
    try: db.commit()
    except IntegrityError: db.rollback(); raise HTTPException(409, "Already applied to this job")
    return application_out(db.scalar(select(Application).options(joinedload(Application.candidate),joinedload(Application.job)).where(Application.id==obj.id)))
@app.get("/api/applications/candidate/{candidate_id}", response_model=list[ApplicationOut])
def candidate_apps(candidate_id:int, db:Session=Depends(get_db)):
    candidate_or_404(db,candidate_id); return [application_out(a) for a in db.scalars(select(Application).options(joinedload(Application.candidate),joinedload(Application.job)).where(Application.candidate_id==candidate_id)).all()]
@app.get("/api/jobs/{job_id}/applications", response_model=list[ApplicationOut])
def job_apps(job_id:int, db:Session=Depends(get_db)):
    job_or_404(db,job_id); return [application_out(a) for a in db.scalars(select(Application).options(joinedload(Application.candidate),joinedload(Application.job)).where(Application.job_id==job_id)).all()]
@app.patch("/api/applications/{application_id}/status", response_model=ApplicationOut)
def application_status(application_id:int,data:StatusIn,db:Session=Depends(get_db), _: Candidate = Depends(admin_user)):
    if data.status not in ["Applied","Shortlisted","Rejected"]: raise HTTPException(422,"Invalid pipeline status")
    a=db.get(Application,application_id)
    if not a: raise HTTPException(404,"Application not found")
    a.status=data.status; db.commit(); return application_out(db.scalar(select(Application).options(joinedload(Application.candidate),joinedload(Application.job)).where(Application.id==a.id)))
@app.get("/api/admin/dashboard")
def dashboard(db:Session=Depends(get_db), _: Candidate = Depends(admin_user)):
    jobs=db.scalars(select(Job)).all(); apps=db.scalars(select(Application).options(joinedload(Application.candidate))).all()
    skills=Counter(s for a in apps for s in a.candidate.skills)
    pipe=Counter(a.status for a in apps)
    return {"applications_per_job":[{"job_id":j.id,"job_title":j.title,"application_count":len(j.applications)} for j in jobs],"skill_distribution":[{"skill":s,"count":c} for s,c in skills.most_common()],"pipeline_counts":{s:pipe.get(s,0) for s in ["Applied","Shortlisted","Rejected"]}}
@app.post("/api/ai/match-jobs")
def ai_match(data:MatchIn,db:Session=Depends(get_db), user: Candidate = Depends(current_user)):
    if user.role != "candidate" or user.id != data.candidate_id: raise HTTPException(403, "Candidates can only match jobs for themselves")
    candidate=candidate_or_404(db,data.candidate_id); prefs=extract_preferences(data.query,candidate)
    jobs=db.scalars(select(Job).where(Job.status=="open")).all()
    # Retrieve jobs by vector similarity first; Gemini sees only those retrieved listings.
    matches = semantic_job_matches(data.query, candidate, jobs)
    return {"parsed_preferences":prefs,"matches":matches,"assistant_reply":generate_match_assistant_reply(data.query, candidate, matches)}

@app.post("/api/chat")
def chat(data: ChatIn, db: Session = Depends(get_db), user: Candidate = Depends(current_user)):
    if user.role != "candidate": raise HTTPException(403, "The job assistant is available to candidates")
    jobs = db.scalars(select(Job).where(Job.status == "open")).all()
    try:
        return answer_job_question(data.message, user, jobs)
    except ValueError as error:
        raise HTTPException(503, str(error))
