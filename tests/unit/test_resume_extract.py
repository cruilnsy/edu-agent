# tests/unit/test_resume_extract.py
# PDF text extraction must read every page (up to MAX_PAGES) and cap the text once for all nodes.

import fitz
import pytest

from backend.agents.resume import nodes


def _make_pdf(path, pages: list[str]) -> str:
    doc = fitz.open()
    for body in pages:
        doc.new_page().insert_textbox(fitz.Rect(54, 54, 558, 738), body, fontsize=10)
    doc.save(path)
    doc.close()
    return str(path)


def test_text_from_every_page_is_extracted(tmp_path):
    pdf = _make_pdf(tmp_path / "two.pdf", ["Page one: Education", "Page two: Projects FastAPI"])
    result = nodes._sync_extract_text(pdf)
    assert result["page_count"] == 2
    assert "Page one" in result["raw_text"] and "Page two: Projects FastAPI" in result["raw_text"]


def test_pages_beyond_the_limit_are_skipped(tmp_path, monkeypatch):
    monkeypatch.setattr(nodes, "MAX_PAGES", 2)
    pdf = _make_pdf(tmp_path / "three.pdf", ["first", "second", "third"])
    result = nodes._sync_extract_text(pdf)
    assert result["page_count"] == 3                 # The real count is still reported
    assert "second" in result["raw_text"] and "third" not in result["raw_text"]


@pytest.mark.parametrize("limit, expect_capped", [(10_000, False), (50, True)])
async def test_extract_text_node_caps_text_once(tmp_path, monkeypatch, limit, expect_capped):
    monkeypatch.setattr(nodes, "MAX_RESUME_CHARS", limit)
    pdf = _make_pdf(tmp_path / "r.pdf", ["A" * 80 + " end of page one", "page two marker"])
    out = await nodes.extract_text_node({"pdf_local_path": pdf})
    assert len(out["raw_text"]) <= limit
    assert ("page two marker" not in out["raw_text"]) == expect_capped


def test_ligatures_are_expanded(tmp_path):
    from scripts.evaluation.resume_pdf import DATASET_DIR, render_pdf
    pdf = render_pdf(DATASET_DIR / "r01.md", tmp_path / "r01.pdf")   # The renderer emits ligatures
    raw_text = nodes._sync_extract_text(str(pdf))["raw_text"]
    assert not any(ch in raw_text for ch in "\ufb00\ufb01\ufb02\ufb03\ufb04")


def test_pages_are_joined_without_a_marker(tmp_path):
    pdf = _make_pdf(tmp_path / "two.pdf", ["first page", "second page"])
    assert nodes._sync_extract_text(pdf)["raw_text"] == "first page\n\nsecond page"


def test_pieces_on_one_row_are_merged(tmp_path):
    """Exporters split a header or a title with right-aligned dates into separate text lines."""
    doc = fitz.open()
    page = doc.new_page()
    for x, text in [(72, "jane@example.com"), (170, "|"), (180, "(555) 010-0000")]:
        page.insert_text((x, 72), text, fontsize=10)
    page.insert_text((72, 110), "Software Engineer, Acme Corp", fontsize=10)
    page.insert_text((450, 110), "Sep 2022 - Present", fontsize=10)
    page.insert_text((80, 125), "- Built a FastAPI reporting service used by every sales team in the company",
                     fontsize=10)
    doc.save(tmp_path / "row.pdf")
    lines = nodes._sync_extract_text(str(tmp_path / "row.pdf"))["raw_text"].splitlines()
    assert lines[0] == "jane@example.com | (555) 010-0000"
    assert lines[1].startswith("Software Engineer, Acme Corp") and lines[1].endswith("Sep 2022 - Present")


def test_two_column_layout_reads_left_then_right(tmp_path):
    doc = fitz.open()
    page = doc.new_page()                          # 612 pt wide; the gutter is at 306
    page.insert_text((72, 60), "Jane Doe - Backend Engineer - jane@example.com - (555) 010-0000", fontsize=11)
    for i in range(6):
        page.insert_text((40, 100 + i * 15), f"left {i}", fontsize=10)
        page.insert_text((330, 100 + i * 15), f"right {i}", fontsize=10)
    doc.save(tmp_path / "cols.pdf")
    text = nodes._sync_extract_text(str(tmp_path / "cols.pdf"))["raw_text"]
    assert text.startswith("Jane Doe")
    assert text.index("left 5") < text.index("right 0")
