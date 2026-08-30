"""Transparent, deterministic preference extraction and job-match scoring."""
import re
from .models import Candidate, Job

# Small controlled vocabulary: keeps matching explainable without an external LLM.
ALIASES = {"bengaluru": "bangalore", "new delhi": "delhi", "backend developer": "backend", "backend engineer": "backend", "full stack": "fullstack"}
KNOWN_SKILLS = ["python", "fastapi", "django", "flask", "postgresql", "sql", "react", "javascript", "typescript", "node", "aws", "docker", "kubernetes", "java", "machine learning", "ai", "figma", "css"]
LOCATIONS = ["bangalore", "bengaluru", "mumbai", "delhi", "new delhi", "hyderabad", "pune", "chennai", "remote"]
ROLES = ["backend", "frontend", "fullstack", "data", "devops", "designer", "product"]

def norm(value):
    """Normalize user/job values before comparing them."""
    value = (value or "").strip().lower()
    return ALIASES.get(value, value)

def extract_preferences(query: str, candidate: Candidate):
    """Extract structured preferences; profile skills are a safe fallback."""
    text = query.lower()
    skills = [s for s in KNOWN_SKILLS if re.search(r"\b" + re.escape(s) + r"\b", text)]
    if not skills: skills = candidate.skills or []
    location = next((norm(x) for x in LOCATIONS if x in text), None)
    role = next((x for x in ROLES if x in text), None)
    experience = next((x for x in ["0-2", "2-4", "3-5", "5+", "junior", "senior", "entry"] if x in text), None)
    domain = next((x for x in ["healthcare", "fintech", "ai", "education", "ecommerce", "startup"] if x in text), None)
    return {"skills": skills, "location": location, "role_type": role, "domain_interest": domain, "experience_level": experience}

def match_jobs(preferences, jobs):
    """Score each supplied open job with transparent weights and rank descending."""
    matches = []
    for job in jobs:
        score, reasons = 0.0, []
        query_skills = {norm(x) for x in preferences["skills"]}
        job_skills = {norm(x) for x in job.required_skills}
        matched = sorted(query_skills & job_skills)
        if query_skills:
            # Skill points are proportional, so partial matches receive partial credit.
            points = 50 * len(matched) / len(query_skills); score += points
            if matched: reasons.append("skills: " + ", ".join(matched))
        if preferences["location"] and norm(job.location) == preferences["location"]:
            score += 20; reasons.append("location matches")
        if preferences["role_type"] and preferences["role_type"] in job.title.lower():
            score += 15; reasons.append("role aligns")
        if preferences["experience_level"] and preferences["experience_level"] in job.experience_level.lower():
            score += 10; reasons.append("experience aligns")
        if preferences["domain_interest"] and preferences["domain_interest"] in (job.title + " " + job.description).lower():
            score += 5; reasons.append("domain aligns")
        if score:
            explanation = "Good match because " + ", ".join(reasons) + "."
            matches.append({"job_id": job.id, "title": job.title, "location": job.location, "experience_level": job.experience_level, "match_score": round(score), "matched_skills": matched, "explanation": explanation})
    return sorted(matches, key=lambda m: m["match_score"], reverse=True)
