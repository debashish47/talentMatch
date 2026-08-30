"""Job-board RAG: retrieve open job context, rerank it, generate a grounded answer."""
import hashlib
import os
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
from dotenv import load_dotenv

from .models import Candidate, Job

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

_embedder = None
_reranker = None
_answer_cache: list[dict[str, Any]] = []
_cache_fingerprint = ""

def _job_store():
    """Open the persistent Chroma collection used by both matching and chat."""
    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings
    persist_dir = str(Path(__file__).resolve().parents[1] / "chroma_db" / "jobs")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    # Cosine distance makes a returned distance convertible to 0–100 similarity.
    return Chroma(collection_name="open_jobs", persist_directory=persist_dir, embedding_function=embeddings, collection_metadata={"hnsw:space": "cosine"})

def _models():
    """Load ML models only when chat is first used (startup stays fast)."""
    global _embedder, _reranker
    if _embedder is None:
        from sentence_transformers import CrossEncoder, SentenceTransformer
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
        _reranker = CrossEncoder("BAAI/bge-reranker-base")
    return _embedder, _reranker

def _job_text(job: Job) -> str:
    """Build the complete, sourceable context document for one open job."""
    return "\n".join([
        f"Job title: {job.title}", f"Location: {job.location}",
        f"Experience: {job.experience_level}", f"Required skills: {', '.join(job.required_skills)}",
        f"Description: {job.description}",
    ])

def _fingerprint(jobs: list[Job]) -> str:
    """Any job status/update change invalidates cached answers."""
    text = "|".join(f"{j.id}:{j.updated_at}:{j.status}" for j in jobs)
    return hashlib.sha256(text.encode()).hexdigest()

def index_job_embedding(job: Job) -> None:
    """Upsert one admin-created/edited open job into the persistent index."""
    from langchain_core.documents import Document
    store = _job_store()
    job_id = str(job.id)
    # Chroma's add does not overwrite an existing id, so replace the old vector first.
    if job_id in store.get().get("ids", []): store.delete(ids=[job_id])
    if job.status == "open":
        document = Document(page_content=_job_text(job), metadata={"job_id": job.id, "title": job.title, "location": job.location})
        store.add_documents([document], ids=[job_id])

def remove_job_embedding(job_id: int) -> None:
    """Remove a closed or deleted job so it can never appear in semantic results."""
    store = _job_store()
    key = str(job_id)
    if key in store.get().get("ids", []): store.delete(ids=[key])

def ensure_job_embeddings(jobs: list[Job]) -> None:
    """One-time backfill for seeded/existing open jobs created before RAG was enabled."""
    store = _job_store()
    indexed = set(store.get().get("ids", []))
    for job in jobs:
        if str(job.id) not in indexed: index_job_embedding(job)

def semantic_job_matches(query: str, candidate: Candidate, jobs: list[Job]) -> list[dict]:
    """Embed a candidate request, retrieve nearest open-job vectors, and explain overlap."""
    from .services import extract_preferences, norm
    if not jobs: return []
    ensure_job_embeddings(jobs)
    store = _job_store()
    # Chroma performs the vector similarity search using the query embedding.
    results = store.similarity_search_with_score(query, k=min(10, len(jobs)))
    jobs_by_id = {job.id: job for job in jobs}
    preferences = extract_preferences(query, candidate)
    requested_skills = {norm(skill) for skill in preferences["skills"]}
    matches = []
    for document, distance in results:
        job = jobs_by_id.get(document.metadata["job_id"])
        if not job: continue
        required = {norm(skill) for skill in job.required_skills}
        matched = sorted(requested_skills & required)
        # Chroma cosine distance: 0 means identical; transform it to an intuitive percent.
        similarity = round(max(0.0, 1.0 - float(distance)) * 100)
        explanation = f"Semantic match from your request. Matched requirements: {', '.join(matched) if matched else 'related role and description'}."
        matches.append({"job_id": job.id, "title": job.title, "description": job.description, "location": job.location, "experience_level": job.experience_level, "required_skills": job.required_skills, "match_score": similarity, "matched_skills": matched, "explanation": explanation})
    return matches

def generate_match_assistant_reply(question: str, candidate: Candidate, matches: list[dict]) -> Optional[str]:
    """Turn already-retrieved semantic matches into a grounded one-turn job assistant reply."""
    if not matches or not os.getenv("GEMINI_API_KEY"):
        return None
    context = "\n\n---\n\n".join(
        f"Job: {match['title']}\nLocation: {match['location']}\nExperience: {match['experience_level']}\nRequired skills: {', '.join(match['required_skills'])}\nDescription: {match['description']}\nMatched requirements: {', '.join(match['matched_skills']) or 'semantic match'}"
        for match in matches[:5]
    )
    profile = f"Candidate skills: {', '.join(candidate.skills or []) or 'not provided'}; preferred role: {candidate.preferred_role_type or 'not provided'}; preferred location: {candidate.preferred_location or 'not provided'}."
    prompt = f"""You are TalentMatch's job assistant. Recommend and explain the most suitable jobs using ONLY the retrieved job listings below. Mention specific job titles and why they fit the candidate. If none is a strong fit, say that clearly. Do not invent information. Keep the answer under 180 words.

{profile}

RETRIEVED JOB LISTINGS:
{context}

CANDIDATE REQUEST: {question}
ANSWER:"""
    from langchain_google_genai import ChatGoogleGenerativeAI
    llm = ChatGoogleGenerativeAI(model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"), google_api_key=os.getenv("GEMINI_API_KEY"), temperature=0.2)
    return str(llm.invoke(prompt).content)

def _cache_lookup(question: str, fingerprint: str):
    """Return a cached answer only when cosine similarity is at least 0.80."""
    global _answer_cache, _cache_fingerprint
    if fingerprint != _cache_fingerprint:
        _answer_cache, _cache_fingerprint = [], fingerprint
        return None, 0.0
    if not _answer_cache: return None, 0.0
    embedder, _ = _models()
    vector = embedder.encode(question, normalize_embeddings=True)
    scores = [float(np.dot(vector, item["vector"])) for item in _answer_cache]
    index, score = int(np.argmax(scores)), max(scores)
    return (_answer_cache[index] if score >= 0.80 else None), score

def answer_job_question(question: str, candidate: Candidate, jobs: list[Job]) -> dict:
    if not os.getenv("GEMINI_API_KEY"):
        raise ValueError("Chatbot is not configured. Add GEMINI_API_KEY to backend/.env, then restart the backend.")
    if not jobs:
        return {"answer": "There are no open jobs to search right now.", "sources": [], "cache_hit": False, "latency_ms": 0}
    started = time.perf_counter()
    fingerprint = _fingerprint(jobs)
    cached, similarity = _cache_lookup(question, fingerprint)
    if cached:
        return {"answer": cached["answer"], "sources": cached["sources"], "cache_hit": True, "similarity": round(similarity, 2), "latency_ms": round((time.perf_counter()-started)*1000, 2)}

    from langchain_core.documents import Document
    from langchain_chroma import Chroma
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_huggingface import HuggingFaceEmbeddings

    # A persistent Chroma collection is refreshed from the live open-job data,
    # so closed/deleted roles can never be cited by the assistant.
    persist_dir = str(Path(__file__).resolve().parents[1] / "chroma_db" / "jobs")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    documents = [Document(page_content=_job_text(job), metadata={"job_id": job.id, "title": job.title, "location": job.location}) for job in jobs]
    store = Chroma(collection_name="open_jobs", persist_directory=persist_dir, embedding_function=embeddings)
    existing = store.get().get("ids", [])
    if existing: store.delete(ids=existing)
    store.add_documents(documents, ids=[str(job.id) for job in jobs])
    # Embedding search is a broad first pass: return up to ten candidate jobs.
    retrieved = store.similarity_search(question, k=min(10, len(jobs)))

    # CrossEncoder reads question + job together, then narrows context to the top five.
    _, reranker = _models()
    ranked = sorted(zip(reranker.predict([(question, doc.page_content) for doc in retrieved]), retrieved), key=lambda item: item[0], reverse=True)[:5]
    context = "\n\n---\n\n".join(doc.page_content for _, doc in ranked)
    profile = f"Candidate skills: {', '.join(candidate.skills or []) or 'not provided'}; preferred role: {candidate.preferred_role_type or 'not provided'}; preferred location: {candidate.preferred_location or 'not provided'}; domain interest: {candidate.domain_interest or 'not provided'}."
    prompt = f"""You are TalentMatch's job-board assistant. Answer only using the OPEN JOB CONTEXT below. Be concise, helpful, and honest. If the context does not support the answer, say so. Do not invent a job, salary, or company detail. When recommending a role, mention its exact title and explain the fit using available skills/location.

{profile}

OPEN JOB CONTEXT:
{context}

QUESTION: {question}
ANSWER:"""
    # Gemini receives only reranked source context, not the full database.
    llm = ChatGoogleGenerativeAI(model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"), google_api_key=os.getenv("GEMINI_API_KEY"), temperature=0.2)
    answer = llm.invoke(prompt).content
    sources = [{"job_id": doc.metadata["job_id"], "title": doc.metadata["title"], "location": doc.metadata["location"]} for _, doc in ranked]
    embedder, _ = _models()
    _answer_cache.append({"vector": embedder.encode(question, normalize_embeddings=True), "answer": str(answer), "sources": sources})
    return {"answer": str(answer), "sources": sources, "cache_hit": False, "similarity": round(similarity, 2), "latency_ms": round((time.perf_counter()-started)*1000, 2)}
