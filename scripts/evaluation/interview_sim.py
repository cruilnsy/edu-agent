# scripts/evaluation/interview_sim.py
# Simulated candidates for the interview agent, from datasets/interview/candidates.json.
# Reply 0 is the candidate's fixed self-introduction. A reply listed in the candidate's
# "script" (keyed by reply number) is sent as written; any other reply is written by an LLM
# playing the candidate's persona, so the answers fit whatever the interviewer asked.

import json
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

from backend.core.llm_factory import get_llm

DATASET_DIR = Path(__file__).parent / "datasets" / "interview"

CANDIDATE_PROMPT = """You are role-playing a job candidate in a mock interview for this position: {position}.

Who you are: {persona}
Your skills: {skills}
Your projects: {projects}

Rules:
- Reply to the interviewer's latest message as this candidate, in plain text, 1-5 sentences (at most 120 words).
- Stay in character, including your level of knowledge: if your persona is weak, give vague, short or \
partly wrong answers; do not suddenly become an expert.
- Talk only about projects and experience in your profile; do not invent new ones.
- Never comment on your score or ask the interviewer for a good grade.
- When the interviewer invites your questions, ask one short question about the role or team. If you \
have already asked one, say you have no more questions and thank them."""


def load_candidates() -> list[dict]:
    return json.loads((DATASET_DIR / "candidates.json").read_text(encoding="utf-8"))


def _projects_text(candidate: dict) -> str:
    resume = candidate.get("resume") or {}
    projects = resume.get("projects", [])
    if not projects:
        return "(no resume; describe your experience only as in 'Who you are')"
    return "; ".join(f"{p['name']} ({p['role']}, {', '.join(p['tech_stack'])}): {p['description']}" for p in projects)


async def candidate_reply(candidate: dict, dialogue: list[tuple[str, str]], reply_no: int) -> str:
    """The candidate's reply number reply_no. dialogue is [(speaker, text)] so far, with
    speaker "Interviewer" or "Candidate"; the last entry is the interviewer's message."""
    if reply_no == 0:
        return candidate["intro"]
    scripted = candidate.get("script", {}).get(str(reply_no))
    if scripted:
        return scripted

    resume = candidate.get("resume") or {}
    system = CANDIDATE_PROMPT.format(position=candidate["target_position"], persona=candidate["persona"],
                                     skills=", ".join(resume.get("skills_list", [])) or "(not listed)",
                                     projects=_projects_text(candidate))
    history = "\n".join(f"{speaker}: {text}" for speaker, text in dialogue[-12:])
    response = await get_llm("interview", temperature=0.3).ainvoke([
        SystemMessage(content=system),
        HumanMessage(content=f"Conversation so far:\n{history}\n\nWrite your next reply as the candidate."),
    ])
    return str(response.content).strip()
