# backend/agents/resume/nodes.py

import asyncio
import json
import os
import re
from datetime import date

from langchain_core.messages import HumanMessage, SystemMessage
from sqlalchemy import text

from backend.agents.resume.prompts import (
    DIAGNOSE_ISSUES_PROMPT, DIAGNOSE_THINK_PROMPT, DIMENSION_REVIEW_PROMPTS,
    EXTRACT_STRUCTURED_PROMPT, GENERATE_SUMMARY_PROMPT, SYSTEM_PROMPT,
)
from backend.agents.resume.state import (
    DimensionScore, IssueList, ResumeState, ResumeStructured, ResumeSummary,
)
from backend.core.llm_factory import get_llm, get_structured_llm
from backend.core.logger import get_logger
from backend.dependencies import AsyncSessionLocal

logger = get_logger(__name__)

DEFAULT_TARGET_POSITION = "Backend Engineer"   # Used when the resume does not state a target role

# Resumes run 1-2 pages (about 3,000 characters each). The caps leave room for a dense 3-page
# resume and stop a huge PDF from multiplying token cost across the 8 LLM calls that read the text.
MAX_PAGES = 5
MAX_RESUME_CHARS = 12_000


def _today() -> str:
    """Today's date for the timeline prompts. The model cannot know it, and without it
    flags past dates after its training cutoff as "in the future"."""
    return date.today().isoformat()

# ── The six review dimensions: name, weight, focus. Weights add up to 1.0. ──
SIX_DIMENSIONS = [
    {"key": "project_depth",  "name": "Project depth",  "weight": 0.30,
     "focus": "Do project descriptions include metrics, reasons for technical choices, "
              "personal contribution, and problems solved?"},
    {"key": "tech_match",     "name": "Tech match",     "weight": 0.25,
     "focus": "Do the skills match the target role's keywords, and is each listed skill "
              "backed by real use in the experience section?"},
    {"key": "expression",     "name": "Writing",        "weight": 0.15,
     "focus": "Bullets start with strong action verbs, use consistent tense, no first-person "
              "pronouns, no typos or filler buzzwords"},
    {"key": "structure",      "name": "Structure",      "weight": 0.15,
     "focus": "Appropriate length, reverse-chronological order, ATS-friendly layout, "
              "no photo or protected personal details (a work-authorization line is fine)"},
    {"key": "quantification", "name": "Quantification", "weight": 0.10,
     "focus": "Use of numbers such as performance, user counts, and improvement percentages"},
    {"key": "authenticity",   "name": "Credibility",    "weight": 0.05,
     "focus": "No exaggeration; technical depth matches years of experience; timeline is plausible"},
]


# ── Nodes 1-2: no-ops in local mode (kept as extension points for object storage) ──
async def upload_to_minio_node(state: ResumeState) -> dict:
    """The file is already local, so there is nothing to upload."""
    logger.info("upload_to_minio.skip", review_id=state["review_id"], reason="local_mode")
    return {}


async def download_pdf_node(state: ResumeState) -> dict:
    """The file is already at pdf_local_path, so there is nothing to download."""
    logger.info("download_pdf.skip", review_id=state["review_id"], reason="local_file_exists")
    return {}


# ── Node 3: extract_text - PDF text extraction (rebuilds visual rows, handles two columns) ──
def _page_lines(page) -> list[tuple[tuple[float, float, float, float], str]]:
    """Every text line on the page as (bbox, text). Ligatures are expanded ("ﬁ" -> "fi"):
    LaTeX and other typesetters emit them and the LLM would report them as corrupt text."""
    import fitz
    flags = fitz.TEXTFLAGS_DICT & ~fitz.TEXT_PRESERVE_LIGATURES & ~fitz.TEXT_PRESERVE_IMAGES
    lines = []
    for block in page.get_text("dict", flags=flags)["blocks"]:
        for line in block.get("lines", []):
            text = "".join(span["text"] for span in line["spans"]).strip()
            if text:
                lines.append((line["bbox"], text))
    return lines


def _join_rows(lines: list) -> str:
    """Merge lines that sit on the same row, top to bottom and left to right.

    Word and PDF exporters often split one visual row into separate text lines: a header like
    "email | phone | LinkedIn", or a job title with its dates right-aligned. Read line by line,
    those come out as "email\n|\nphone" and dates detached from their role.
    """
    rows: list[list] = []
    for bbox, text in sorted(lines, key=lambda l: (l[0][1], l[0][0])):
        x0, y0, x1, y1 = bbox
        center, half_height = (y0 + y1) / 2, max(y1 - y0, 1) / 2
        if rows and abs(center - rows[-1][0]) <= half_height:
            rows[-1][1].append((x0, x1, text))
        else:
            rows.append([center, [(x0, x1, text)]])

    out = []
    for _, items in rows:
        items.sort()
        row = items[0][2]
        for (_, prev_x1, _), (x0, _, text) in zip(items, items[1:]):
            row += ("   " if x0 - prev_x1 > 20 else " ") + text   # Wide gap: e.g. title ... dates
        out.append(row)
    return "\n".join(out)


def _page_text(page) -> str:
    lines = _page_lines(page)
    if not lines:
        return ""
    # Two columns leave a gutter that almost no line crosses. In a single column most bullets
    # cross the middle, so right-aligned dates alone do not make a page look two-column.
    midpoint = page.rect.width / 2
    spanning = [l for l in lines if l[0][0] < midpoint < l[0][2]]
    left     = [l for l in lines if l[0][2] <= midpoint]
    right    = [l for l in lines if l[0][0] >= midpoint]
    if len(left) >= 5 and len(right) >= 5 and len(spanning) <= 0.1 * len(lines):
        # Full-width lines (usually the name header) first, then the left column, then the right
        return "\n".join(part for part in (_join_rows(spanning), _join_rows(left), _join_rows(right)) if part)
    return _join_rows(lines)


def _sync_extract_text(pdf_path: str) -> dict:
    """Blocking PDF parsing, run in a thread pool. Pages are joined with a blank line and no
    marker; the real page count is returned separately for the length check."""
    import fitz                                       # PyMuPDF
    doc = fitz.open(pdf_path)
    page_count = len(doc)
    pages = [_page_text(page) for page in doc.pages(0, min(page_count, MAX_PAGES))]
    doc.close()
    raw_text = "\n\n".join(text for text in pages if text)
    return {"raw_text": raw_text, "page_count": page_count}


async def extract_text_node(state: ResumeState) -> dict:
    """Run the blocking parser in a thread so the event loop stays free."""
    pdf_path = state["pdf_local_path"]
    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, _sync_extract_text, pdf_path)
        raw_text, page_count = result["raw_text"], result["page_count"]
        if page_count > MAX_PAGES or len(raw_text) > MAX_RESUME_CHARS:
            logger.warning("extract_text.truncated", page_count=page_count, text_length=len(raw_text),
                           max_pages=MAX_PAGES, max_chars=MAX_RESUME_CHARS)
            raw_text = raw_text[:MAX_RESUME_CHARS]   # Every later node reads this capped text
        if len(raw_text.strip()) < 200:               # Probably a scanned/image PDF; warn only
            logger.warning("extract_text.text_too_short", text_length=len(raw_text.strip()))
        logger.info("extract_text.done", page_count=page_count, text_length=len(raw_text))
        return {"raw_text": raw_text, "page_count": page_count}
    except Exception as e:
        logger.error("extract_text.failed", error=str(e))
        raise


# ── Node 4: extract_structured - LLM structured extraction ──
async def extract_structured_node(state: ResumeState) -> dict:
    """Turn the raw text into ResumeStructured through function calling."""
    raw_text = state["raw_text"]
    fallback_used = state.get("fallback_used", False)
    prompt = EXTRACT_STRUCTURED_PROMPT.format(resume_text=raw_text)   # Already capped in extract_text
    structured_llm = get_structured_llm("resume", ResumeStructured)
    structured_dict = None
    for attempt in range(2):                           # Structured output occasionally returns None
        try:
            result = await structured_llm.ainvoke([
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ])
            if result is None:
                raise ValueError("structured output returned None")
            structured_dict = result.model_dump()
            break
        except Exception as e:
            if attempt == 0:
                logger.warning("extract_structured.retry", error=str(e))
                await asyncio.sleep(1)
            else:
                logger.warning("extract_structured.failed", error=str(e))
    if structured_dict is None:
        fallback_used = True
        structured_dict = ResumeStructured(name="(extraction failed)").model_dump()   # Empty fallback
    logger.info("extract_structured.done", projects_count=len(structured_dict.get("projects", [])))
    return {"structured": structured_dict, "fallback_used": fallback_used}


# ── Node 5: run_six_dimensions - review all six dimensions in parallel ──
async def run_six_dimensions_node(state: ResumeState) -> dict:
    """Review the six dimensions concurrently with asyncio.gather and compute the weighted score."""
    raw_text   = state["raw_text"]
    structured = state.get("structured") or {}
    structured_summary = _build_structured_summary(structured)
    fallback_used    = state.get("fallback_used", False)

    async def review_one_dimension(dim: dict) -> dict:
        prompt_template = DIMENSION_REVIEW_PROMPTS.get(dim["key"], "")
        if not prompt_template:
            return _empty_dimension_score(dim)
        prompt = prompt_template.format(
            resume_text=raw_text, structured_summary=structured_summary, focus=dim["focus"],
            today=_today(), page_count=state.get("page_count", 0),   # Unused keys are ignored
        )
        for attempt in range(2):
            try:
                structured_llm = get_structured_llm("resume", DimensionScore)
                result: DimensionScore = await structured_llm.ainvoke([
                    SystemMessage(content=SYSTEM_PROMPT),
                    HumanMessage(content=prompt),
                ])
                d = result.model_dump()
                d["dimension"], d["weight"], d["key"] = dim["name"], dim["weight"], dim["key"]
                return d
            except Exception as e:
                if attempt == 0:
                    logger.warning("six_dimensions.dimension_retry", dimension=dim["key"], error=str(e))
                    await asyncio.sleep(1)
                else:
                    logger.warning("six_dimensions.dimension_failed", dimension=dim["key"], error=str(e))
                    return _empty_dimension_score(dim)

    dimension_scores = await asyncio.gather(*(review_one_dimension(dim) for dim in SIX_DIMENSIONS))
    weighted_score = sum(d["score"] * d["weight"] for d in dimension_scores)
    fallback_used = fallback_used or any(d.get("fallback_used") for d in dimension_scores)
    logger.info("six_dimensions.done", weighted_score=round(weighted_score, 2),
                scores={d["key"]: d["score"] for d in dimension_scores})
    return {"dimension_scores": list(dimension_scores), "weighted_score": round(weighted_score, 2), "fallback_used": fallback_used}


def _build_structured_summary(structured: dict) -> str:
    """Condense the extraction into a few lines for the reviewers (saves tokens)."""
    lines = []
    if structured.get("name"):
        lines.append(f"Name: {structured['name']}")
    if structured.get("target_position"):
        lines.append(f"Target role: {structured['target_position']}")
    if structured.get("education"):
        edu = structured["education"][0]
        lines.append(f"Highest education: {edu.get('school', '')} {edu.get('major', '')} {edu.get('degree', '')}")
    if structured.get("skills_list"):
        lines.append(f"Tech stack: {', '.join(structured['skills_list'][:10])}")
    if structured.get("projects"):
        proj_names = [p.get("name", "") for p in structured["projects"]]
        lines.append(f"Projects: {len(structured['projects'])} ({', '.join(proj_names[:3])})")
    if structured.get("work_experience"):
        roles, today = structured["work_experience"], date.fromisoformat(_today())
        years, dated = _total_experience_years(roles, today)
        lines.append(f"Work experience: {len(roles)} roles; {years} years in total, computed from "
                     f"the dates of {dated} of them with overlaps merged")
        lines.append(_role_order_note(roles, today))
    return "\n".join(lines) if lines else "(Structured extraction failed; review the raw text instead.)"


_MONTHS = {m: i for i, m in enumerate(
    ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"), start=1)}
_PRESENT = ("present", "current", "now", "today")


def _parse_month(text: str, today: date, is_end: bool) -> int | None:
    """Turn "Sep 2022", "09/2022", "2022.09", "2022" or "Present" into a month index."""
    t = text.strip().lower()
    if any(word in t for word in _PRESENT):
        return today.year * 12 + today.month - 1
    if m := re.search(r"([a-z]{3})[a-z]*\.?\s+(\d{4})", t):
        if m.group(1) in _MONTHS:
            return int(m.group(2)) * 12 + _MONTHS[m.group(1)] - 1
    if m := re.search(r"(\d{1,2})\s*/\s*(\d{4})", t):
        return int(m.group(2)) * 12 + int(m.group(1)) - 1
    if m := re.search(r"(\d{4})\s*[./-]\s*(\d{1,2})\b", t):
        return int(m.group(1)) * 12 + int(m.group(2)) - 1
    if m := re.search(r"\b(\d{4})\b", t):                  # Year only: count the whole year
        return int(m.group(1)) * 12 + (11 if is_end else 0)
    return None


def _role_months(role: dict, today: date) -> tuple[int, int] | None:
    """(start, end) month indexes of a role, or None if its dates cannot be read."""
    parts = re.split(r"\s+(?:-|to)\s+|\s*[–—]\s*", role.get("duration", ""), maxsplit=1)
    if len(parts) != 2:
        return None
    start, end = _parse_month(parts[0], today, False), _parse_month(parts[1], today, True)
    return (start, end) if start is not None and end is not None and end >= start else None


def _role_order_note(work_experience: list[dict], today: date) -> str:
    """Whether roles, in resume order, run newest first by end date. LLMs misjudge this."""
    dated = [(role, months) for role in work_experience if (months := _role_months(role, today))]
    for (above, (_, above_end)), (below, (_, below_end)) in zip(dated, dated[1:]):
        if below_end > above_end:
            return (f"Role order: NOT reverse-chronological by end date: \"{above.get('company', '')}\" "
                    f"({above.get('duration', '')}) is listed above \"{below.get('company', '')}\" "
                    f"({below.get('duration', '')})")
    return "Role order: correct (reverse-chronological by end date)"


def _total_experience_years(work_experience: list[dict], today: date) -> tuple[float, int]:
    """Total years across all roles, with overlapping roles merged, and how many roles had
    dates we could read. Done in code because LLMs get date arithmetic wrong."""
    intervals = [(start, end + 1)                             # End month is inclusive
                 for role in work_experience if (months := _role_months(role, today))
                 for start, end in [months]]
    months, current_end = 0, None
    for start, end in sorted(intervals):
        if current_end is not None and start < current_end:   # Overlaps the previous role
            start = current_end
        if end > start:
            months += end - start
            current_end = max(end, current_end or end)
    return round(months / 12, 1), len(intervals)


def _empty_dimension_score(dim: dict) -> dict:
    """Fallback when a dimension review fails."""
    return {"key": dim["key"], "dimension": dim["name"], "score": 50, "weight": dim["weight"],
            "issues": ["This dimension could not be reviewed; manual review recommended."], "suggestions": [], "fallback_used": True}


# ── Node 6: diagnose_issues - itemized issue list ──
async def diagnose_issues_node(state: ResumeState) -> dict:
    """Collect dimension issues -> think step -> structured issue list -> sort by priority."""
    dimension_scores = state.get("dimension_scores", [])
    raw_text         = state["raw_text"]
    structured       = state.get("structured") or {}
    fallback_used    = state.get("fallback_used", False)

    all_raw_issues = [
        f"[{dim['dimension']}] {issue_text}"
        for dim in dimension_scores for issue_text in dim.get("issues", [])
    ]
    raw_issues_text = "\n".join(f"- {i}" for i in all_raw_issues) or "(none)"

    reasoning_trace = ""                              # Think step; allowed to fail
    try:
        dimension_scores_summary = "\n".join(
            f"- {d['dimension']}: {d['score']} - issues: {', '.join(d.get('issues', [])[:2])}"
            for d in dimension_scores
        )
        think_prompt = DIAGNOSE_THINK_PROMPT.format(
            dimension_scores_summary=dimension_scores_summary, raw_issues=raw_issues_text, today=_today())
        think_llm = get_llm("resume", temperature=0)
        think_resp = await think_llm.ainvoke([HumanMessage(content=think_prompt)])
        reasoning_trace = (
            think_resp.text if hasattr(think_resp, "text") and not callable(think_resp.text)
            else str(think_resp.content)
        ).strip()
    except Exception as e:
        logger.warning("diagnose_think.failed", error=str(e))

    think_context = f"\n\n[Analysis before diagnosis]\n{reasoning_trace}" if reasoning_trace else ""

    prompt = DIAGNOSE_ISSUES_PROMPT.format(
        resume_text=raw_text,
        structured_summary=_build_structured_summary(structured),
        raw_issues=raw_issues_text,
        today=_today(),
        page_count=state.get("page_count", 0),
    ) + think_context

    try:
        structured_llm = get_structured_llm("resume", IssueList)
        result: IssueList = await structured_llm.ainvoke([
            SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)])
        issues = [item.model_dump() for item in result.items]
    except Exception as e:
        logger.warning("diagnose_issues.failed", error=str(e))
        issues = [                                    # Fallback: reuse dimension issues as medium priority
            {"priority": "medium", "dimension": dim["dimension"], "description": issue,
             "location": "Whole resume", "suggestion": "See the dimension review for suggestions."}
            for dim in dimension_scores for issue in dim.get("issues", [])
        ]
        fallback_used = True

    priority_order = {"high": 0, "medium": 1, "low": 2}
    issues.sort(key=lambda x: priority_order.get(x.get("priority", "low"), 2))
    logger.info("diagnose_issues.done", total=len(issues),
                high=sum(1 for i in issues if i.get("priority") == "high"))
    return {"issues": issues, "fallback_used": fallback_used}


# ── Node 7: generate_summary - overall assessment ──
async def generate_summary_node(state: ResumeState) -> dict:
    """Combine extraction, scores and issues into a student-facing assessment."""
    structured       = state.get("structured") or {}
    dimension_scores = state.get("dimension_scores", [])
    issues           = state.get("issues", [])
    weighted_score   = state.get("weighted_score", 0.0)
    fallback_used    = state.get("fallback_used", False)

    high_issues = [i["description"] for i in issues if i.get("priority") == "high"][:5]
    high_issues_text = "\n".join(f"- {i}" for i in high_issues) or "(no high-priority issues)"
    scores_text = "\n".join(
        f"- {d['dimension']}: {d['score']} (weight {int(d['weight'] * 100)}%)" for d in dimension_scores)

    prompt = GENERATE_SUMMARY_PROMPT.format(
        structured_summary=_build_structured_summary(structured),
        scores_summary=scores_text, weighted_score=round(weighted_score, 1),
        high_issues=high_issues_text,
        target_position=structured.get("target_position") or DEFAULT_TARGET_POSITION,
    )
    structured_llm = get_structured_llm("resume", ResumeSummary)
    summary_dict = None
    for attempt in range(2):                           # Structured output occasionally returns None
        try:
            result = await structured_llm.ainvoke([
                SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)])
            if result is None:
                raise ValueError("structured output returned None")
            summary_dict = result.model_dump()
            break
        except Exception as e:
            if attempt == 0:
                logger.warning("generate_summary.retry", error=str(e))
                await asyncio.sleep(1)
            else:
                logger.warning("generate_summary.failed", error=str(e))
    if summary_dict is None:
        fallback_used = True
        summary_dict = {                              # Fallback assessment
            "highlights": ["Resume submitted in full."],
            "core_improvements": high_issues[:2] or ["See the dimension feedback for suggestions."],
            "overall_comment": f"Overall score {round(weighted_score, 1)}. See the detailed feedback per dimension.",
            "fit_assessment": "Role fit assessment is not available.",
        }
    logger.info("generate_summary.done", highlights_count=len(summary_dict.get("highlights", [])))
    return {"summary": summary_dict, "fallback_used": fallback_used}


# ── Node 8: save_results - persist the review ──
async def save_results_node(state: ResumeState) -> dict:
    """Write the result to resume_reviews (JSONB columns) and delete the temp file."""
    review_id = state["review_id"]
    fallback_used = state["fallback_used"]

    structured_output = {                              # Full result for callers
        "review_id": review_id, "student_id": state["student_id"],
        "structured": state.get("structured"),
        "weighted_score": state.get("weighted_score", 0),
        "dimension_scores": state.get("dimension_scores", []),
        "issues": state.get("issues", []),
        "summary": state.get("summary"),
    }

    async with AsyncSessionLocal() as session:
        try:
            await session.execute(
                text("""
                    UPDATE resume_reviews
                    SET structured_data = :structured_data,
                        scores          = :scores,
                        issues          = :issues,
                        summary         = :summary,
                        status          = 'done',
                        updated_at      = NOW()
                    WHERE id = :review_id
                """),
                {
                    # JSONB columns take JSON strings; ensure_ascii=False keeps non-ASCII text readable
                    "structured_data": json.dumps(state.get("structured"), ensure_ascii=False),
                    "scores": json.dumps(
                        {"dimension_scores": state.get("dimension_scores", []),
                         "weighted_score": state.get("weighted_score", 0)},
                        ensure_ascii=False),
                    "issues":  json.dumps(state.get("issues", []), ensure_ascii=False),
                    "summary": json.dumps(state.get("summary"), ensure_ascii=False),
                    "review_id": review_id,
                },
            )
            await session.commit()
            logger.info("save_results.db_written", review_id=review_id)
        except Exception as e:
            await session.rollback()
            logger.error("save_results.db_failed", error=str(e))
            raise

    local_path = state.get("pdf_local_path", "")
    if local_path and os.path.exists(local_path):
        os.remove(local_path)
        logger.info("save_results.tmp_cleaned", review_id=review_id)

    return {"fallback_used": fallback_used, "structured_output": structured_output}
