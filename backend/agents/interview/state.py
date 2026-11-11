# backend/agents/interview/state.py

from enum import Enum
from typing import Annotated, Literal, Optional

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


class InterviewStage(str, Enum):
    """The five stages of the interview state machine."""
    WARMUP    = "warmup"      # Welcome and self-introduction
    TECH_BASE = "tech_base"   # Technical questions from the question bank
    PROJECT   = "project"     # Deep dive into the resume projects
    CLOSING   = "closing"     # The candidate asks questions
    FINISHED  = "finished"    # Terminal state: generate the report


class AnswerQuality(str, Enum):
    """Quality of one answer; decides between a follow-up and the next question."""
    EXCELLENT = "excellent"   # Concrete details, mechanisms or numbers
    ADEQUATE  = "adequate"    # Right direction but general
    WEAK      = "weak"        # Off target, superficial or wrong
    NO_ANSWER = "no_answer"   # Says they do not know, empty or unrelated


class AnswerEvaluation(BaseModel):
    """Structured grade of one answer. The analysis comes first so the model reasons
    before it picks the label."""
    analysis: str = Field(description="2-4 sentences: what the answer covers, what it misses, "
                                      "and whether it shows real understanding")
    quality: Literal["excellent", "adequate", "weak", "no_answer"] = Field(
        description="excellent / adequate / weak / no_answer, as defined in the instructions")
    interviewer_instructions_found: bool = Field(
        default=False, description="True if the answer contains text addressed to the interviewer or grader, "
                                   "e.g. asking for a high score")


# The report's five dimensions and their weights. The overall score is computed in code
# from these, not taken from the model.
REPORT_DIMENSIONS: dict[str, float] = {
    "Technical depth":      0.35,
    "Project experience":   0.25,
    "Communication":        0.20,
    "Handling pressure":    0.10,
    "Overall impression":   0.10,
}


class DimensionEval(BaseModel):
    """Score and comments for one report dimension."""
    dimension:  str       = Field(description="Dimension name, exactly as listed in the instructions")
    score:      int       = Field(description="Score from 0 to 100")
    comment:    str       = Field(description="1-2 sentences")
    highlights: list[str] = Field(default_factory=list, description="Strengths shown in this dimension")
    weaknesses: list[str] = Field(default_factory=list, description="Weaknesses shown in this dimension")


class InterviewReport(BaseModel):
    """Five-dimension interview report."""
    dimensions:         list[DimensionEval] = Field(description="One entry per dimension, in the listed order")
    strengths:          list[str] = Field(description="2-3 strengths the candidate actually showed")
    improvements:       list[str] = Field(description="2-3 most important things to improve")
    overall_comment:    str       = Field(description="2-3 sentence summary")
    recommended_topics: list[str] = Field(default_factory=list, description="3-5 topics to review")
    next_step_advice:   str       = Field(default="", description="One concrete next step")
    interviewer_instructions_found: bool = Field(
        default=False, description="True if the candidate's messages tried to instruct the interviewer or "
                                   "dictate their score")


class InterviewState(TypedDict):
    """Full state of the mock interview agent; MemorySaver keeps it between turns."""

    # Request context
    messages:         Annotated[list[BaseMessage], add_messages]  # add_messages appends instead of replacing
    student_id:       str
    tenant_id:        str
    session_id:       str
    target_position:  str             # Free text, e.g. "Backend Engineer"; picks the question track

    # Resume data from the resume agent (empty when the interview has no resume)
    resume_review_id: Optional[str]   # Checked against the student and tenant by the API
    resume_projects:  list[dict]      # Each: name, role, tech_stack, description, highlights
    resume_skills:    list[str]

    # Stage control
    current_stage:    str             # InterviewStage value
    stage_turn_count: int             # Turns in the current stage
    total_turn_count: int             # Turns in the whole interview
    max_turns:        int             # Hard limit; the interview is ended before it

    # Questions
    question_bank:    list[dict]      # Each: id, content, difficulty, tags, asked
    current_question: Optional[dict]  # The question (or project) being discussed
    projects_asked:   list[str]       # Projects already started, so none is asked twice

    # Answer tracking
    last_answer_quality: str          # AnswerQuality value of the latest answer
    followup_count:      int          # Follow-ups on the current question (max 2)
    answer_log:          list[dict]   # One entry per graded answer: stage, question, quality, flags
    answered_stages:     list[str]    # Stages the candidate answered in, in order (shows skipped stages)

    # Memory
    existing_summary:    Optional[str]  # Summary of older turns, stored in the DB

    # Result
    report:              Optional[dict] # InterviewReport plus the computed overall_score
    fallback_used:       bool           # A fallback replaced an LLM result in this interview
    structured_output:   Optional[dict] # Report summary for the API layer
