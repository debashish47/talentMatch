from datetime import datetime
from pydantic import BaseModel, Field
from typing import Literal, Optional

class CandidateIn(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    # Keep email optional and dependency-free for an MVP. Browser validation covers
    # the candidate form; an API client may provide any valid-looking string.
    email: Optional[str] = Field(default=None, max_length=255)
    skills: list[str] = []
    education: Optional[str] = None
    project_summaries: list[str] = []
    preferred_location: Optional[str] = None
    preferred_role_type: Optional[str] = None
    domain_interest: Optional[str] = None
class CandidateOut(CandidateIn):
    id: int
    role: Literal["candidate", "admin"] = "candidate"
    model_config = {"from_attributes": True}

class RegisterIn(CandidateIn):
    name: Optional[str] = None
    email: str = Field(min_length=3, max_length=255)
    password: str = Field(min_length=6, max_length=128)
    role: Literal["candidate", "admin"]

class LoginIn(BaseModel):
    email: str = Field(min_length=3, max_length=255)
    password: str = Field(min_length=1, max_length=128)

class JobIn(BaseModel):
    title: str = Field(min_length=1, max_length=180)
    description: str = Field(min_length=1)
    required_skills: list[str] = []
    experience_level: str
    location: str
    status: Literal["open", "closed"] = "open"
class JobOut(JobIn):
    id: int
    model_config = {"from_attributes": True}
class StatusIn(BaseModel):
    status: str
class ApplyIn(BaseModel):
    candidate_id: int
    job_id: int
class MatchIn(BaseModel):
    candidate_id: int
    query: str = Field(min_length=2)
class ChatIn(BaseModel):
    message: str = Field(min_length=2, max_length=2000)
class ApplicationOut(BaseModel):
    id: int
    candidate_id: int
    job_id: int
    status: str
    applied_at: datetime
    candidate_name: Optional[str] = None
    job_title: Optional[str] = None
    candidate_skills: list[str] = []
    candidate_education: Optional[str] = None
    candidate_projects: list[str] = []
    candidate_preferred_role: Optional[str] = None
    candidate_preferred_location: Optional[str] = None
