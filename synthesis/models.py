from typing import Optional, List
from pydantic import BaseModel, model_validator

class WorkHistoryItem(BaseModel):
    title: str
    company: str
    duration_years: float
    domain: str
    achievements: List[str]

class CandidateProfile(BaseModel):
    source: str
    full_name: str
    current_title: str
    years_experience: int
    skills: List[str]
    work_history: List[WorkHistoryItem]
    project_highlights: List[str]
    summary: str
    # Internal specific
    department: Optional[str] = None
    tenure_years: Optional[float] = None
    performance_rating: Optional[str] = None
    promotion_readiness: Optional[str] = None
    mobility_interest: Optional[str] = None
    # External specific
    current_company: Optional[str] = None
    location: Optional[str] = None
    notice_period_days: Optional[int] = None
    salary_expectation: Optional[int] = None

    @model_validator(mode='after')
    def validate_tenure(self) -> 'CandidateProfile':
        if self.source == "internal" and self.tenure_years is not None:
            if self.tenure_years > self.years_experience:
                raise ValueError(f"Tenure ({self.tenure_years}) cannot exceed total experience ({self.years_experience})")
        return self

class CandidateList(BaseModel):
    candidates: List[CandidateProfile]
