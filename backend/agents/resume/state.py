# backend/agents/resume/state.py

from typing import Annotated, Optional

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages      # Reducer that appends to the message list
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


# ──────────────────────────────────────────────────────────────
# 1. Structured extraction (target of the LLM function call)
#    Field descriptions are sent to the LLM, so they double as extraction instructions.
# ──────────────────────────────────────────────────────────────

class EducationItem(BaseModel):
    school:   str = Field(description="School name")
    major:    str = Field(description="Major or field of study")
    degree:   str = Field(description="Degree, e.g. Associate, Bachelor's, Master's")
    duration: str = Field(description="Dates attended, e.g. Sep 2020 - Jun 2024")
    gpa:      str = Field(default="", description="GPA or grades (optional)")


class ProjectItem(BaseModel):
    name:        str       = Field(description="Project name")
    role:        str       = Field(description="Role, e.g. backend developer, full-stack, lead")
    duration:    str       = Field(description="Project dates, e.g. Jun 2023 - Dec 2023")
    tech_stack:  list[str] = Field(description="Technologies used, e.g. [FastAPI, PostgreSQL, Redis]")
    description: str       = Field(description="Project description, verbatim")
    # Mutable defaults need default_factory; default=[] would be shared by every instance
    highlights:  list[str] = Field(default_factory=list, description="Sentences that contain numbers (quantified results)")


class WorkItem(BaseModel):
    company:     str       = Field(description="Company name")
    position:    str       = Field(description="Job title")
    duration:    str       = Field(description="Employment dates")
    tech_stack:  list[str] = Field(default_factory=list, description="Technologies used")
    description: str       = Field(description="Job description, verbatim")


class ResumeStructured(BaseModel):
    """Full extraction schema; output of the extract_structured node."""
    # Basic info
    name:            str  = Field(description="Candidate name")
    phone:           str  = Field(default="", description="Phone number")
    email:           str  = Field(default="", description="Email address")
    target_position: str  = Field(default="", description="Target job title")
    # Education, most recent first
    education:       list[EducationItem] = Field(default_factory=list)
    # Skills
    skills_raw:      str       = Field(default="", description="Skills section, verbatim")
    skills_list:     list[str] = Field(default_factory=list, description="Parsed list of technology tags")
    # Projects
    projects:        list[ProjectItem] = Field(default_factory=list)
    # Work and internships
    work_experience: list[WorkItem] = Field(default_factory=list)
    # Other
    certificates:    list[str] = Field(default_factory=list, description="Certifications")
    self_intro:      str       = Field(default="", description="Summary or objective section, verbatim")


# ──────────────────────────────────────────────────────────────
# 2. Six-dimension review
# ──────────────────────────────────────────────────────────────

class DimensionScore(BaseModel):
    # dimension and weight are filled in by code from SIX_DIMENSIONS; the LLM only scores
    dimension:   str       = Field(default="", description="Dimension name (set by code; may be empty)")
    score:       int       = Field(description="Score from 0 to 100")
    weight:      float     = Field(default=0.0, description="Weight (set by code; may be 0)")
    issues:      list[str] = Field(default_factory=list, description="Problems found in this dimension")
    suggestions: list[str] = Field(default_factory=list, description="Improvement suggestions")


class IssueItem(BaseModel):
    priority:    str = Field(description="Priority: high / medium / low")
    dimension:   str = Field(description="Dimension the issue belongs to")
    description: str = Field(description="One-sentence description of the problem")
    location:    str = Field(description="Where it is in the resume, e.g. Projects - Order Service - bullet 2")
    suggestion:  str = Field(description="Concrete, actionable fix")


class IssueList(BaseModel):
    """Wrapper for a list of IssueItem.

    with_structured_output needs an object at the top level, not a bare list,
    so the LLM returns {"items": [...]}.
    """
    items: list[IssueItem]


class ResumeSummary(BaseModel):
    """Overall assessment; output of the generate_summary node."""
    highlights:        list[str] = Field(description="2-3 key strengths")
    core_improvements: list[str] = Field(description="2-3 most important improvements")
    overall_comment:   str       = Field(description="1-2 sentence overall assessment")
    fit_assessment:    str       = Field(description="One sentence on fit for the target role")


# ──────────────────────────────────────────────────────────────
# 3. Graph state, shared by all 8 nodes. Fields are grouped by the node that produces them.
# ──────────────────────────────────────────────────────────────

class ResumeState(TypedDict):
    # ── Request context ──
    messages:       Annotated[list[BaseMessage], add_messages]
    student_id:     str
    tenant_id:      str
    review_id:      str        # resume_reviews.id
    pdf_minio_path: str        # Object storage path; unused in local mode (legacy)
    pdf_local_path: str        # Local temp file actually read

    # ── extract_text ──
    raw_text:       str
    page_count:     int

    # ── extract_structured ──
    structured:     Optional[dict]   # ResumeStructured.model_dump()

    # ── run_six_dimensions ──
    dimension_scores: list[dict]     # One entry per dimension
    weighted_score:   float          # Weighted total, 0-100

    # ── diagnose_issues ──
    issues:          list[dict]      # IssueItem dicts, sorted by priority

    # ── generate_summary ──
    summary:         Optional[dict]  # ResumeSummary.model_dump()

    # ── Fallback flags ──
    fallback_used:     bool
    structured_output: Optional[dict]
