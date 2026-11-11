# backend/agents/exam/nodes.py
# Exam grading nodes:
#   parse_word -> load_questions_meta -> run_three_tracks -> aggregate_results
#   -> analyze_weak_points -> notify_teacher -> teacher_review [interrupt]
#   -> apply_teacher_decision -> publish_results

import asyncio
import json
import re
import uuid

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import interrupt
from sqlalchemy import text

from backend.agents.exam.prompts import (
    CODE_REVIEW_PROMPT,
    SUBJECTIVE_REVIEW_PROMPT,
    SUBJECTIVE_THINK_PROMPT,
    SYSTEM_PROMPT,
    WEAK_POINTS_ANALYSIS_PROMPT,
)
from backend.agents.exam.state import (
    CodeReviewResult,
    ExamState,
    SubjectiveReviewResult,
    WeakPointsReport,
)
from backend.core.llm_factory import get_llm, get_structured_llm
from backend.core.logger import get_logger
from backend.dependencies import AsyncSessionLocal

logger = get_logger(__name__)

OBJECTIVE_TYPES = ("single_choice", "multi_choice", "judge")
REVIEW_CONFIDENCE = 0.7        # Below this, the teacher must check the grade
SUBJECTIVE_GROUP_SIZE = 3      # Short answers graded in parallel per group, to stay under rate limits


def _get_message_content(msg) -> str:
    """Text of an LLM message (content may be a string or a list of parts)."""
    return msg.content if isinstance(msg.content, str) else str(msg.content)


def _wrap_answer(answer: str, empty_text: str) -> str:
    """Put the student's answer in tags so the grader treats it as data.

    A closing tag inside the answer is removed so the answer cannot end the block early.
    """
    body = (answer or "").replace("</student_answer>", "").strip() or empty_text
    return f"<student_answer>\n{body}\n</student_answer>"


def _fallback_result(q: dict, question_type: str, reason: str) -> dict:
    """Placeholder when AI grading fails: 0 points, sent to the teacher for manual grading."""
    return {
        "question_id":    q["question_id"],
        "question_no":    q["question_no"],
        "question_type":  question_type,
        "knowledge_tag":  q.get("knowledge_tag", ""),
        "content":        q.get("content", ""),
        "student_answer": q.get("student_answer", ""),
        "score":          0,
        "full_score":     q["full_score"],
        "needs_review":   True,
        "confidence":     0.0,
        "fallback":       True,
        "ai_feedback":    f"AI grading failed ({reason}); please grade this question manually.",
        "point_results":  [],
    }


# ──────────────────────────────────────────────────────────────
# Node 1: parse_word: read the student's answers from the .docx
# ──────────────────────────────────────────────────────────────

# A question starts at a line like "Question 3", "Question 3.", "Q3:" or "Q 3)".
# Numbered lists inside answers ("1. ...") are not headers, so they are kept as answer text.
QUESTION_HEADER = re.compile(r"^(?:question|q)\s*\.?\s*(\d+)(?=$|[\s.:)\]-])", re.IGNORECASE)
ANSWER_MARKER = re.compile(r"^answer\s*:", re.IGNORECASE)
FENCE = "```"


def parse_answer_lines(lines: list[str]) -> list[dict]:
    """Split answer-sheet lines into questions.

    Returns [{question_no, header_text, student_answer, is_code}] in sheet order.
    Within a question, the answer is everything after the "Answer:" marker; without a marker,
    every line after the header counts. Code inside ``` fences keeps its indentation.
    Lines before the first header (title, name) are ignored.
    """
    questions: list[dict] = []
    current: dict | None = None
    in_code = False

    def finish():
        if current is None:
            return
        body = current["answer_lines"] if current["has_marker"] else current["all_lines"]
        current["student_answer"] = "\n".join(body).strip("\n")
        if not current["is_code"]:
            current["student_answer"] = current["student_answer"].strip()
        for key in ("answer_lines", "all_lines", "has_marker", "in_answer"):
            del current[key]
        questions.append(current)

    for raw in lines:
        stripped = raw.strip()

        if stripped.startswith(FENCE):
            in_code = not in_code
            if current is not None:
                current["is_code"] = True
            continue

        if in_code:
            if current is not None:
                current["all_lines"].append(raw.rstrip())
                if current["in_answer"]:
                    current["answer_lines"].append(raw.rstrip())
            continue

        header = QUESTION_HEADER.match(stripped)
        if header:
            finish()
            current = {"question_no": int(header.group(1)), "header_text": stripped,
                       "is_code": False, "answer_lines": [], "all_lines": [],
                       "has_marker": False, "in_answer": False}
            continue

        if current is None or not stripped:
            continue

        marker = ANSWER_MARKER.match(stripped)
        if marker:
            current["has_marker"] = current["in_answer"] = True
            rest = stripped[marker.end():].strip()
            if rest:
                current["answer_lines"].append(rest)
            continue

        current["all_lines"].append(stripped)
        if current["in_answer"]:
            current["answer_lines"].append(stripped)

    finish()
    return questions


def _sync_parse_word(word_path: str) -> list[dict]:
    """Read the .docx paragraphs (blocking I/O and XML parsing) and parse them."""
    from docx import Document
    return parse_answer_lines([p.text for p in Document(word_path).paragraphs])


async def parse_word_node(state: ExamState) -> dict:
    """Parse the uploaded answer sheet.

    python-docx does blocking file I/O and XML parsing, so it runs in the default thread pool
    (run_in_executor) and the event loop keeps serving other requests meanwhile.
    """
    word_path = state["word_file_path"]
    try:
        loop = asyncio.get_running_loop()
        parsed = await loop.run_in_executor(None, _sync_parse_word, word_path)
        logger.info("parse_word.done", questions_found=len(parsed))
        return {"parsed_questions": parsed}
    except Exception as e:
        # Fallback: a broken file gives no answers. load_questions_meta still loads every
        # question, so each one gets an empty answer and the teacher sees the whole exam.
        logger.error("parse_word.failed", error=str(e))
        return {"parsed_questions": []}


# ──────────────────────────────────────────────────────────────
# Node 2: load_questions_meta: merge answers with the questions in the DB
# ──────────────────────────────────────────────────────────────

def merge_questions(questions: list[dict], scoring_points: dict[str, list[dict]],
                    parsed: list[dict]) -> list[dict]:
    """DB questions are the source of truth; parsed answers are attached by question number."""
    answers_by_no = {p["question_no"]: p.get("student_answer", "") for p in parsed}
    return [{
        "question_id":    str(q["id"]),
        "question_no":    q["question_no"],
        "question_type":  q["question_type"],
        "content":        q["content"],
        "student_answer": answers_by_no.get(q["question_no"], ""),
        "correct_answer": q["correct_answer"] or "",
        "scoring_points": scoring_points.get(str(q["id"]), []),
        "full_score":     q["score"],
        "knowledge_tag":  q["knowledge_tag"] or "",
    } for q in questions]


async def load_questions_meta_node(state: ExamState) -> dict:
    """Load the exam's questions, answer keys, scoring points and tags, and merge the answers in."""
    exam_id = state["exam_id"]

    async with AsyncSessionLocal() as session:
        questions = (await session.execute(
            text("""
                SELECT id, question_no, question_type, content, correct_answer, score, knowledge_tag
                FROM questions
                WHERE exam_id = :exam_id AND tenant_id = :tenant_id
                ORDER BY question_no
            """),
            {"exam_id": exam_id, "tenant_id": state["tenant_id"]},
        )).mappings().all()

        points_rows = []
        question_ids = [str(q["id"]) for q in questions]
        if question_ids:
            # One named parameter per id (asyncpg rejects ANY(:ids::uuid[]) through text())
            names = [f":qid_{i}" for i in range(len(question_ids))]
            points_rows = (await session.execute(
                text(f"""
                    SELECT id, question_id, point_desc, point_score
                    FROM scoring_points
                    WHERE question_id IN ({", ".join(names)}) AND is_active = TRUE
                    ORDER BY question_id, created_at, id
                """),
                {f"qid_{i}": qid for i, qid in enumerate(question_ids)},
            )).mappings().all()

    scoring_points: dict[str, list[dict]] = {}
    for sp in points_rows:
        scoring_points.setdefault(str(sp["question_id"]), []).append(
            {"id": str(sp["id"]), "desc": sp["point_desc"], "score": sp["point_score"]})

    merged = merge_questions([dict(q) for q in questions], scoring_points, state["parsed_questions"])
    logger.info("load_questions_meta.done", exam_id=exam_id, total_questions=len(merged))
    return {"parsed_questions": merged}


# ──────────────────────────────────────────────────────────────
# Track 1: rules (single choice, multiple choice, true/false)
# ──────────────────────────────────────────────────────────────

TRUE_WORDS = {"TRUE", "T", "YES", "Y", "CORRECT", "RIGHT"}
FALSE_WORDS = {"FALSE", "F", "NO", "N", "INCORRECT", "WRONG"}


def normalize_answer(answer: str, question_type: str) -> str:
    """Normalize an objective answer so format differences do not matter.

    true/false: "True", "t", "Yes" -> "TRUE"; "False", "f", "No" -> "FALSE".
    choices: "a, b, d", "D B A", "A and B", "abd", "Option B" -> sorted letters ("ABD", "AB", "B").
    """
    upper = (answer or "").strip().upper()
    if question_type == "judge":
        word = re.sub(r"[^A-Z]", "", upper)
        if word in TRUE_WORDS:
            return "TRUE"
        if word in FALSE_WORDS:
            return "FALSE"
        return word
    letters = re.findall(r"\b[A-H]\b", upper)       # Separated letters: "A, B and D"
    if not letters:
        compact = re.sub(r"[^A-Z]", "", upper)
        if re.fullmatch(r"[A-H]+", compact):        # Run-together letters: "ABD"
            letters = list(compact)
    return "".join(sorted(set(letters)))


async def _run_objective_track(questions: list[dict]) -> list[dict]:
    """Rule-based grading: all or nothing. Async only so it fits asyncio.gather with the other tracks."""
    results = []
    for q in questions:
        student = normalize_answer(q["student_answer"], q["question_type"])
        correct = normalize_answer(q["correct_answer"], q["question_type"])
        is_correct = bool(student) and student == correct
        results.append({
            "question_id":    q["question_id"],
            "question_no":    q["question_no"],
            "question_type":  q["question_type"],
            "knowledge_tag":  q.get("knowledge_tag", ""),
            "content":        q.get("content", ""),
            "student_answer": q["student_answer"],
            "correct_answer": q["correct_answer"],
            "is_correct":     is_correct,
            "score":          q["full_score"] if is_correct else 0,
            "full_score":     q["full_score"],
            "needs_review":   False,          # Rules are exact; no teacher check needed
            "ai_feedback":    "Correct" if is_correct else f"Correct answer: {q['correct_answer']}",
        })
    return results


# ──────────────────────────────────────────────────────────────
# Track 2: LLM grading of short answers against the scoring points
# ──────────────────────────────────────────────────────────────

def _rubric_text(scoring_points: list[dict]) -> str:
    return "\n".join(f"{i + 1}. [{sp['score']} points] {sp['desc']}"
                     for i, sp in enumerate(scoring_points)) or "(No rubric: grade the answer as a whole.)"


def score_subjective(q: dict, result: SubjectiveReviewResult) -> tuple[int, list[dict], bool]:
    """Turn the LLM's grade into (score, point_results, rubric_mismatch).

    The total is the sum of the per-point scores, each capped at its point value, so the
    LLM's own arithmetic is never trusted. If it returned a different number of points than
    the rubric has, its total is used (capped) and the question goes to the teacher.
    """
    points = q["scoring_points"]
    full = q["full_score"]
    if not points or len(result.point_results) != len(points):
        return max(0, min(result.total_score, full)), [p.model_dump() for p in result.point_results], bool(points)

    point_results = []
    for sp, pr in zip(points, result.point_results):
        earned = max(0, min(pr.earned_score, sp["score"]))
        point_results.append({
            "point_id":     sp.get("id", ""),
            "point_desc":   sp["desc"],             # From the rubric, not from the LLM
            "point_score":  sp["score"],
            "earned_score": earned,
            "earned":       earned == sp["score"],  # The frontend shows full points in green
            "evidence":     pr.evidence,
            "missing":      pr.missing,
        })
    return min(sum(p["earned_score"] for p in point_results), full), point_results, False


STRUCTURED_ATTEMPTS = 2       # One retry: the model sometimes returns no tool call or an invalid one


async def _structured_grade(agent_type: str, schema, prompt: str):
    """Structured LLM call with one retry when the reply is empty or fails validation."""
    last_error: Exception | None = None
    for attempt in range(1, STRUCTURED_ATTEMPTS + 1):
        try:
            result = await get_structured_llm(agent_type, schema).ainvoke([
                SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)])
            if result is not None:
                return result
            last_error = ValueError("the model returned no structured output")
        except Exception as e:      # Validation or parsing error; API errors are retried too
            last_error = e
        logger.warning("structured_grade.retry", agent_type=agent_type, attempt=attempt, error=str(last_error))
    raise last_error


async def _review_one_subjective(q: dict) -> dict:
    """Grade one short answer in two steps: a free-text analysis, then a structured grade."""
    rubric = _rubric_text(q["scoring_points"])
    answer = _wrap_answer(q["student_answer"], "(no answer)")

    # Step 1: think first. If it fails, grade without the analysis.
    analysis = ""
    try:
        think = await get_llm("exam_subjective", temperature=0).ainvoke([HumanMessage(content=SUBJECTIVE_THINK_PROMPT.format(
            question_content=q["content"], scoring_points=rubric, student_answer=answer))])
        analysis = _get_message_content(think).strip()
    except Exception as e:
        logger.warning("subjective_think.failed", question_no=q.get("question_no"), error=str(e))

    # Step 2: structured grade, with the analysis as extra context
    prompt = SUBJECTIVE_REVIEW_PROMPT.format(
        question_content=q["content"], scoring_points=rubric, full_score=q["full_score"],
        student_answer=answer,
        analysis=f"\nYour analysis before grading:\n{analysis}\n" if analysis else "",
    )
    result: SubjectiveReviewResult = await _structured_grade("exam_subjective", SubjectiveReviewResult, prompt)

    score, point_results, mismatch = score_subjective(q, result)
    confidence = max(0.0, min(result.confidence, 1.0))
    return {
        "question_id":    q["question_id"],
        "question_no":    q["question_no"],
        "question_type":  "short_answer",
        "knowledge_tag":  q.get("knowledge_tag", ""),
        "content":        q.get("content", ""),
        "student_answer": q["student_answer"],
        "score":          score,
        "full_score":     q["full_score"],
        "confidence":     confidence,
        "grader_instructions_found": result.grader_instructions_found,
        "needs_review":   confidence < REVIEW_CONFIDENCE or mismatch or result.grader_instructions_found,
        "ai_feedback":    result.overall_comment,
        "point_results":  point_results,
    }


async def _run_subjective_track(questions: list[dict]) -> list[dict]:
    """Grade short answers in groups of 3: parallel inside a group, groups one after another,
    so a long exam does not fire dozens of LLM calls at once."""
    results = []
    for i in range(0, len(questions), SUBJECTIVE_GROUP_SIZE):
        group = questions[i:i + SUBJECTIVE_GROUP_SIZE]
        outcomes = await asyncio.gather(*[_review_one_subjective(q) for q in group], return_exceptions=True)
        for q, outcome in zip(group, outcomes):
            if isinstance(outcome, Exception):
                # One failed question does not stop the batch: the teacher grades it by hand
                logger.warning("subjective_track.question_failed", question_id=q["question_id"], error=str(outcome))
                results.append(_fallback_result(q, "short_answer", "grading error"))
            else:
                results.append(outcome)
    return results


# ──────────────────────────────────────────────────────────────
# Track 3: LLM grading of code (correctness first, then quality)
# ──────────────────────────────────────────────────────────────

async def _review_one_code(q: dict) -> dict:
    """Grade one coding question against the reference solution in correct_answer."""
    prompt = CODE_REVIEW_PROMPT.format(
        question=q["content"],
        reference_solution=q.get("correct_answer") or "(none: judge the code on its own)",
        code=_wrap_answer(q["student_answer"], "(no code submitted)"),
        full_score=q["full_score"],
    )
    result: CodeReviewResult = await _structured_grade("exam_code", CodeReviewResult, prompt)

    confidence = max(0.0, min(result.confidence, 1.0))
    return {
        "question_id":      q["question_id"],
        "question_no":      q["question_no"],
        "question_type":    "code",
        "knowledge_tag":    q.get("knowledge_tag", ""),
        "content":          q.get("content", ""),
        "student_answer":   q["student_answer"],
        "score":            max(0, min(result.score, q["full_score"])),
        "full_score":       q["full_score"],
        "confidence":       confidence,
        "grader_instructions_found": result.grader_instructions_found,
        "needs_review":     confidence < REVIEW_CONFIDENCE or result.grader_instructions_found,
        "quality_feedback": result.feedback,
        "ai_feedback":      "\n".join(result.feedback),
    }


async def _run_code_track(questions: list[dict]) -> list[dict]:
    """Grade coding questions one by one (an exam has few of them)."""
    results = []
    for q in questions:
        try:
            results.append(await _review_one_code(q))
        except Exception as e:
            logger.warning("code_track.question_failed", question_id=q["question_id"], error=str(e))
            results.append(_fallback_result(q, "code", "grading error"))
    return results


# ──────────────────────────────────────────────────────────────
# Node 3: run_three_tracks: the three tracks run in parallel
# ──────────────────────────────────────────────────────────────

async def run_three_tracks_node(state: ExamState) -> dict:
    """Run rules, short-answer grading and code grading at the same time.

    asyncio.gather(return_exceptions=True) keeps one failing track from cancelling the others.
    The questions of a failed track get fallback results, so no question goes missing.
    """
    questions = state["parsed_questions"]
    tracks = {
        "objective":  ([q for q in questions if q["question_type"] in OBJECTIVE_TYPES], _run_objective_track),
        "subjective": ([q for q in questions if q["question_type"] == "short_answer"], _run_subjective_track),
        "code":       ([q for q in questions if q["question_type"] == "code"], _run_code_track),
    }
    logger.info("three_tracks.start", **{name: len(qs) for name, (qs, _) in tracks.items()})

    outcomes = await asyncio.gather(*[run(qs) for qs, run in tracks.values()], return_exceptions=True)

    results = {}
    for (name, (qs, _)), outcome in zip(tracks.items(), outcomes):
        if isinstance(outcome, Exception):
            logger.error("three_tracks.track_failed", track=name, error=str(outcome))
            outcome = [_fallback_result(q, q["question_type"], "track error") for q in qs]
        results[f"{name}_results"] = outcome

    logger.info("three_tracks.done", **{k: len(v) for k, v in results.items()})
    return results


# ──────────────────────────────────────────────────────────────
# Node 4: aggregate_results: the pre-review the teacher sees
# ──────────────────────────────────────────────────────────────

async def aggregate_results_node(state: ExamState) -> dict:
    """Merge the tracks, sort by question number, and total the scores."""
    by_question = sorted(
        state.get("objective_results", []) + state.get("subjective_results", []) + state.get("code_results", []),
        key=lambda r: r.get("question_no", 0),
    )
    total = sum(r.get("score", 0) for r in by_question)
    full = sum(r.get("full_score", 0) for r in by_question)
    summary = {
        "total_score":        total,
        "full_score":         full,
        "score_rate":         round(total / full, 4) if full else 0.0,
        "needs_review_count": sum(1 for r in by_question if r.get("needs_review")),
        "by_question":        by_question,
    }
    fallback_used = any(r.get("fallback") for r in by_question)
    logger.info("aggregate_results.done", total_score=total, full_score=full,
                needs_review=summary["needs_review_count"], fallback_used=fallback_used)
    return {"pre_review_summary": summary, "fallback_used": fallback_used}


# ──────────────────────────────────────────────────────────────
# Node 5: analyze_weak_points
# ──────────────────────────────────────────────────────────────

async def analyze_weak_points_node(state: ExamState) -> dict:
    """Find weak knowledge areas.

    Questions with a knowledge_tag are grouped by tag (exact counts). All questions with lost
    points also go to the LLM, which writes the suggestions and names areas for untagged ones.
    """
    all_results = state.get("pre_review_summary", {}).get("by_question", [])
    wrong = [r for r in all_results if r.get("score", 0) < r.get("full_score", 0)]

    if not wrong:
        return {"weak_points": [], "weak_points_summary": "Every answer earned full marks. Well done!"}

    # Path 1: group tagged questions by tag; total_count covers every question with the tag
    tagged: dict[str, dict] = {}
    for r in wrong:
        if tag := r.get("knowledge_tag"):
            wp = tagged.setdefault(tag, {"tag": tag, "wrong_count": 0, "question_nos": [], "suggestion": "",
                                         "total_count": sum(1 for a in all_results if a.get("knowledge_tag") == tag)})
            wp["wrong_count"] += 1
            wp["question_nos"].append(r["question_no"])

    # Path 2: the LLM sees every question with lost points
    llm_points, summary = [], ""
    wrong_desc = "\n".join(
        f"Question {r['question_no']} ({r['question_type']}, {r['score']}/{r['full_score']} points, "
        f"tag: {r.get('knowledge_tag') or 'none'})\n"
        f"  Question: {r.get('content', '')[:200]}\n"
        f"  Grader feedback: {r.get('ai_feedback', '')[:150]}"
        for r in wrong
    )
    try:
        report: WeakPointsReport = await get_structured_llm("exam_subjective", WeakPointsReport).ainvoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=WEAK_POINTS_ANALYSIS_PROMPT.format(wrong_questions=wrong_desc)),
        ])
        llm_points = [wp.model_dump() for wp in report.weak_points]
        summary = report.overall_summary
    except Exception as e:
        logger.warning("analyze_weak_points.llm_failed", error=str(e))
        summary = "Weak-point analysis failed; please review the lost points manually."

    # Merge: tagged counts win; the LLM adds suggestions and areas for untagged questions
    final = []
    for wp in llm_points:
        if wp["tag"] in tagged:
            tagged[wp["tag"]]["suggestion"] = wp["suggestion"]
        elif not any(wp["tag"] == f["tag"] for f in final):
            final.append(wp)
    for wp in tagged.values():
        wp["suggestion"] = wp["suggestion"] or f"Review the material on {wp['tag']}."
        final.append(wp)
    final.sort(key=lambda wp: wp.get("wrong_count", 0), reverse=True)

    logger.info("analyze_weak_points.done", weak_count=len(final))
    return {"weak_points": final, "weak_points_summary": summary}


# ──────────────────────────────────────────────────────────────
# Node 6: notify_teacher
# ──────────────────────────────────────────────────────────────

async def notify_teacher_node(state: ExamState) -> dict:
    """Move the submission to pending_review so it shows up in the teacher's queue."""
    async with AsyncSessionLocal() as session:
        async with session.begin():
            await session.execute(
                text("""
                    UPDATE exam_submissions SET status = 'pending_review', updated_at = NOW()
                    WHERE id = :submission_id AND tenant_id = :tenant_id
                """),
                {"submission_id": state["submission_id"], "tenant_id": state["tenant_id"]},
            )
    logger.info("notify_teacher.done", submission_id=state["submission_id"])
    return {"teacher_notified": True}


# ──────────────────────────────────────────────────────────────
# Node 7: teacher_review: human in the loop
# ──────────────────────────────────────────────────────────────

async def teacher_review_node(state: ExamState) -> dict:
    """Pause for the teacher.

    interrupt(payload) saves the state to the checkpointer and stops the graph. The teacher
    reads the pre-review through GET /review and resumes with POST /confirm, which calls
    ainvoke(Command(resume=decision)); interrupt() then returns that decision.
    """
    decision = interrupt({
        "submission_id":       state["submission_id"],
        "student_id":          state["student_id"],
        "pre_review_summary":  state.get("pre_review_summary", {}),
        "weak_points":         state.get("weak_points", []),
        "weak_points_summary": state.get("weak_points_summary", ""),
        "message":             "Check the AI pre-review and weak points, then approve or adjust the scores.",
    })
    logger.info("teacher_review.resumed", submission_id=state["submission_id"], action=decision.get("action"))
    return {"teacher_decision": decision}


# ──────────────────────────────────────────────────────────────
# Node 8: apply_teacher_decision
# ──────────────────────────────────────────────────────────────

async def apply_teacher_decision_node(state: ExamState) -> dict:
    """approve: the AI scores become final. modify: listed questions get the teacher's score and comment."""
    decision = state.get("teacher_decision") or {}
    modifications = decision.get("modifications", []) if decision.get("action") == "modify" else []

    final = [dict(r, final_score=r.get("score", 0))
             for r in state.get("pre_review_summary", {}).get("by_question", [])]
    by_id = {r["question_id"]: r for r in final}

    for mod in modifications:
        r = by_id.get(mod.get("question_id"))
        if r is None:
            continue
        if mod.get("new_score") is not None:
            # The API validates the range; clamp again so the DB never gets an impossible score
            r["teacher_score"] = r["final_score"] = max(0, min(int(mod["new_score"]), r["full_score"]))
        if mod.get("comment"):
            r["teacher_comment"] = mod["comment"]
        r["reviewed_by"] = decision.get("teacher_id", "")

    logger.info("apply_teacher_decision.done", action=decision.get("action"), modifications=len(modifications))
    return {"final_results": final}


# ──────────────────────────────────────────────────────────────
# Node 9: publish_results
# ──────────────────────────────────────────────────────────────

async def publish_results_node(state: ExamState) -> dict:
    """Write the final grades and publish them to the student.

    exam_reviews: delete then insert per question, so publishing twice leaves no duplicates.
    exam_submissions: status 'published' plus the weak points.
    """
    final = state.get("final_results", [])
    submission_id = state["submission_id"]
    teacher_id = (state.get("teacher_decision") or {}).get("teacher_id") or None
    weak_points = state.get("weak_points", [])

    async with AsyncSessionLocal() as session:
        async with session.begin():
            for r in final:
                await session.execute(
                    text("DELETE FROM exam_reviews WHERE submission_id = :sid AND question_id = :qid"),
                    {"sid": submission_id, "qid": r["question_id"]},
                )
                await session.execute(
                    text("""
                        INSERT INTO exam_reviews (
                            id, submission_id, question_id, question_type, knowledge_tag, student_answer,
                            ai_score, ai_feedback, ai_raw_result, teacher_score, teacher_comment,
                            final_score, needs_review, reviewed_by, reviewed_at
                        ) VALUES (
                            :id, :submission_id, :question_id, :question_type, :knowledge_tag, :student_answer,
                            :ai_score, :ai_feedback, :ai_raw_result, :teacher_score, :teacher_comment,
                            :final_score, :needs_review, :reviewed_by, NOW()
                        )
                    """),
                    {
                        "id":              str(uuid.uuid4()),
                        "submission_id":   submission_id,
                        "question_id":     r["question_id"],
                        "question_type":   r["question_type"],
                        "knowledge_tag":   r.get("knowledge_tag", ""),
                        "student_answer":  r.get("student_answer", ""),
                        "ai_score":        r.get("score", 0),
                        "ai_feedback":     r.get("ai_feedback", ""),
                        "ai_raw_result":   json.dumps(r),
                        "teacher_score":   r.get("teacher_score"),     # None: not changed by the teacher
                        "teacher_comment": r.get("teacher_comment"),
                        "final_score":     r.get("final_score", r.get("score", 0)),
                        "needs_review":    r.get("needs_review", False),
                        "reviewed_by":     teacher_id,
                    },
                )
            await session.execute(
                text("""
                    UPDATE exam_submissions
                    SET status = 'published', published_at = NOW(), updated_at = NOW(),
                        weak_points = :weak_points, weak_points_summary = :weak_points_summary
                    WHERE id = :submission_id AND tenant_id = :tenant_id
                """),
                {"submission_id": submission_id, "tenant_id": state["tenant_id"],
                 "weak_points": json.dumps(weak_points),
                 "weak_points_summary": state.get("weak_points_summary", "")},
            )

    total = sum(r.get("final_score", 0) for r in final)
    full = sum(r.get("full_score", 0) for r in final)
    logger.info("publish_results.done", submission_id=submission_id, final_score=total)
    return {
        "published": True,
        "structured_output": {
            "submission_id":       submission_id,
            "final_score":         total,
            "full_score":          full,
            "score_rate":          round(total / full, 4) if full else 0,
            "weak_points":         weak_points,
            "weak_points_summary": state.get("weak_points_summary", ""),
            "published":           True,
        },
    }
