# tests/unit/test_resume_dataset.py
# The fictional resume set must render to readable PDFs and must not contain real contact details.

import json
import re

import pytest

from backend.agents.resume.nodes import _sync_extract_text
from scripts.evaluation.resume_pdf import DATASET_DIR, render_pdf, resume_sources

SOURCES = resume_sources()


def test_every_resume_has_a_label_entry():
    labels = json.loads((DATASET_DIR / "labels.json").read_text())
    assert [entry["id"] for entry in labels] == [p.stem for p in SOURCES]
    assert len(SOURCES) == 10


@pytest.mark.parametrize("source", SOURCES, ids=lambda p: p.stem)
def test_contact_details_are_fictional(source):
    text = source.read_text()
    for email in re.findall(r"[\w.+-]+@[\w-]+\.[\w.]+", text):
        assert email.endswith("@example.com"), email
    for phone in re.findall(r"\(\d{3}\)\s*\d{3}-\d{4}", text):
        assert phone.startswith("(555) 010-"), phone


@pytest.mark.parametrize("source", SOURCES, ids=lambda p: p.stem)
def test_renders_to_a_pdf_the_agent_can_read(source, tmp_path):
    pdf = render_pdf(source, tmp_path / f"{source.stem}.pdf")
    text = _sync_extract_text(str(pdf))["raw_text"]
    name = source.read_text().splitlines()[0].lstrip("# ").strip()
    assert name in text
    assert len(text) > 100
