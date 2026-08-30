"""Embedding-based job matching backed by ChromaDB.

This module has no LLM, chatbot, RAG generation, reranking, or semantic cache.
It stores vectors for open jobs and retrieves the closest vectors for a candidate query.
"""
from pathlib import Path

from .models import Candidate, Job


def _job_text(job: Job) -> str:
    """Combine the searchable fields of one job into the text that gets embedded."""
    return "\n".join([
        f"Job title: {job.title}",
        f"Location: {job.location}",
        f"Experience: {job.experience_level}",
        f"Required skills: {', '.join(job.required_skills)}",
        f"Description: {job.description}",
    ])


def _job_store():
    """Open the persistent Chroma collection containing open-job embeddings."""
    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings

    # This local model turns text into a fixed-length semantic vector.
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    persist_dir = str(Path(__file__).resolve().parents[1] / "chroma_db" / "jobs")
    # Cosine distance makes it easy to present similarity as a percentage.
    return Chroma(collection_name="open_jobs", persist_directory=persist_dir, embedding_function=embeddings, collection_metadata={"hnsw:space": "cosine"})


def index_job_embedding(job: Job) -> None:
    """Create or replace the vector for an admin-created/edited open job."""
    from langchain_core.documents import Document

    store = _job_store()
    job_id = str(job.id)
    # Chroma does not overwrite an existing id, so remove its old representation first.
    if job_id in store.get().get("ids", []):
        store.delete(ids=[job_id])
    if job.status == "open":
        document = Document(page_content=_job_text(job), metadata={"job_id": job.id})
        store.add_documents([document], ids=[job_id])


def remove_job_embedding(job_id: int) -> None:
    """Remove closed/deleted jobs from vector search results."""
    store = _job_store()
    key = str(job_id)
    if key in store.get().get("ids", []):
        store.delete(ids=[key])


def _backfill_existing_jobs(jobs: list[Job]) -> None:
    """Index pre-existing seeded jobs once, after semantic matching is enabled."""
    store = _job_store()
    indexed = set(store.get().get("ids", []))
    for job in jobs:
        if str(job.id) not in indexed:
            index_job_embedding(job)


def semantic_job_matches(query: str, candidate: Candidate, jobs: list[Job]) -> list[dict]:
    """Embed a candidate request and return the closest open job listings.

    Chroma handles semantic retrieval. The keyword extractor is only used to
    display which explicit skill requirements overlap with the candidate request.
    """
    from .services import extract_preferences, norm

    if not jobs:
        return []
    _backfill_existing_jobs(jobs)
    store = _job_store()
    # The query string is embedded automatically by Chroma's embedding function.
    results = store.similarity_search_with_score(query, k=min(10, len(jobs)))
    jobs_by_id = {job.id: job for job in jobs}
    preferences = extract_preferences(query, candidate)
    requested_skills = {norm(skill) for skill in preferences["skills"]}
    matches = []
    for document, distance in results:
        job = jobs_by_id.get(document.metadata["job_id"])
        if not job:
            continue
        required_skills = {norm(skill) for skill in job.required_skills}
        matched_skills = sorted(requested_skills & required_skills)
        # With cosine distance, 0 means identical; 1 - distance is similarity.
        similarity = round(max(0.0, 1.0 - float(distance)) * 100)
        matches.append({
            "job_id": job.id,
            "title": job.title,
            "description": job.description,
            "location": job.location,
            "experience_level": job.experience_level,
            "required_skills": job.required_skills,
            "match_score": similarity,
            "matched_skills": matched_skills,
            "explanation": "Semantic match from your request. Matched requirements: " + (", ".join(matched_skills) or "related role and description") + ".",
        })
    return matches
