# scripts/evaluation/resume_pdf.py
# Renders the Markdown resumes in datasets/resume/ to PDF, so the eval feeds the agent
# the same kind of file a student uploads. Only the sources are committed, not the PDFs.
#
# Supported Markdown: "# name", "## section", "### entry", "- bullet", plain lines, **bold**.

import html
import re
from pathlib import Path

import fitz   # PyMuPDF

DATASET_DIR = Path(__file__).resolve().parent / "datasets" / "resume"

_CSS = """
body { font-family: sans-serif; font-size: 10pt; line-height: 1.35; }
h1 { font-size: 18pt; margin: 0 0 2pt 0; }
h2 { font-size: 11.5pt; margin: 10pt 0 3pt 0; border-bottom: 0.5pt solid #888; }
h3 { font-size: 10.5pt; margin: 6pt 0 2pt 0; }
p  { margin: 0 0 3pt 0; }
ul { margin: 0 0 3pt 14pt; padding: 0; }
li { margin: 0 0 1.5pt 0; }
"""


def markdown_to_html(markdown: str) -> str:
    out, in_list = [], False
    for raw in markdown.splitlines():
        line = raw.strip()
        if not line:
            continue
        is_bullet = line.startswith("- ")
        if in_list and not is_bullet:
            out.append("</ul>")
            in_list = False
        content = _inline(line[2:] if is_bullet else line.lstrip("#").strip())
        if line.startswith("### "):
            out.append(f"<h3>{content}</h3>")
        elif line.startswith("## "):
            out.append(f"<h2>{content}</h2>")
        elif line.startswith("# "):
            out.append(f"<h1>{content}</h1>")
        elif is_bullet:
            if not in_list:
                out.append("<ul>")
                in_list = True
            out.append(f"<li>{content}</li>")
        else:
            out.append(f"<p>{content}</p>")
    if in_list:
        out.append("</ul>")
    return "\n".join(out)


def _inline(text: str) -> str:
    return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", html.escape(text))


def render_pdf(markdown_path: Path, pdf_path: Path) -> Path:
    """Lay the resume out on US Letter pages, adding pages as needed."""
    story = fitz.Story(html=markdown_to_html(markdown_path.read_text()), user_css=_CSS)
    page_rect = fitz.paper_rect("letter")
    content_rect = page_rect + (54, 54, -54, -54)   # 0.75 inch margins
    writer = fitz.DocumentWriter(str(pdf_path))
    more = True
    while more:
        device = writer.begin_page(page_rect)
        more, _ = story.place(content_rect)
        story.draw(device)
        writer.end_page()
    writer.close()
    return pdf_path


def resume_sources(ids: list[str] | None = None) -> list[Path]:
    paths = sorted(DATASET_DIR.glob("r*.md"))
    return [p for p in paths if ids is None or p.stem in ids]
