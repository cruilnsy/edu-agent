# scripts/evaluation/exam_docx.py
# Builds .docx answer sheets from datasets/exam, in the layout the exam agent parses:
#   "Question N (P points) <question text>" ... "Answer: <answer>", code inside ``` fences.

import json
from pathlib import Path

from docx import Document

DATASET_DIR = Path(__file__).parent / "datasets" / "exam"


def load_exam() -> dict:
    return json.loads((DATASET_DIR / "exam.json").read_text(encoding="utf-8"))


def load_submissions() -> list[dict]:
    return json.loads((DATASET_DIR / "submissions.json").read_text(encoding="utf-8"))


def render_answer_sheet(exam: dict, answers: dict[str, str], path: Path) -> Path:
    """Write one student's answer sheet; answers are keyed by question number as a string."""
    doc = Document()
    doc.add_paragraph(exam["title"])
    doc.add_paragraph("Answer sheet")
    for q in exam["questions"]:
        first, *rest = q["content"].split("\n")
        doc.add_paragraph(f"Question {q['question_no']} ({q['score']} points) {first}")
        for line in rest:
            doc.add_paragraph(line)

        answer = answers.get(str(q["question_no"]), "")
        if q["question_type"] == "code" and answer:
            doc.add_paragraph("Answer:")
            doc.add_paragraph("```python")
            for line in answer.split("\n"):
                doc.add_paragraph(line)
            doc.add_paragraph("```")
        else:
            first_line, *more = answer.split("\n")
            doc.add_paragraph(f"Answer: {first_line}".rstrip())
            for line in more:
                doc.add_paragraph(line)
    doc.save(path)
    return path
