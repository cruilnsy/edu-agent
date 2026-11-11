# tests/integration/test_resume_api.py
# Resume API lifecycle with the review graph replaced by a stub, so no LLM is called.

import asyncio

import fitz
import pytest

import backend.api.v1.resume as resume_api

pytestmark = pytest.mark.integration

BASE = "/api/v1/resume"


class _StubGraph:
    """Stands in for the compiled graph; records the state it was given."""

    def __init__(self):
        self.states = []

    async def ainvoke(self, state):
        self.states.append(state)
        return state


@pytest.fixture
def stub_graph(monkeypatch):
    stub = _StubGraph()
    monkeypatch.setattr(resume_api, "_get_graph", lambda: stub)
    return stub


@pytest.fixture
def pdf_bytes() -> bytes:
    doc = fitz.open()
    doc.new_page().insert_text((50, 60), "Alex Example\nBackend Engineer\nSkills: Python, FastAPI")
    data = doc.tobytes()
    doc.close()
    return data


async def test_upload_get_list_delete(client, student_headers, stub_graph, pdf_bytes):
    resp = await client.post(f"{BASE}/upload", headers=student_headers,
                             files={"file": ("resume.pdf", pdf_bytes, "application/pdf")})
    assert resp.status_code == 202
    review_id = resp.json()["review_id"]
    await asyncio.sleep(0)                       # Let the background task run
    assert stub_graph.states[0]["review_id"] == review_id

    got = await client.get(f"{BASE}/reviews/{review_id}", headers=student_headers)
    assert got.status_code == 200 and got.json()["status"] == "processing"

    listed = await client.get(f"{BASE}/reviews", headers=student_headers)
    assert review_id in {item["review_id"] for item in listed.json()["items"]}

    assert (await client.delete(f"{BASE}/reviews/{review_id}", headers=student_headers)).status_code == 204
    assert (await client.get(f"{BASE}/reviews/{review_id}", headers=student_headers)).status_code == 404


async def test_other_users_cannot_see_a_review(client, student_headers, teacher_headers, stub_graph, pdf_bytes):
    resp = await client.post(f"{BASE}/upload", headers=student_headers,
                             files={"file": ("resume.pdf", pdf_bytes, "application/pdf")})
    review_id = resp.json()["review_id"]
    try:
        assert (await client.get(f"{BASE}/reviews/{review_id}", headers=teacher_headers)).status_code == 404
        assert (await client.delete(f"{BASE}/reviews/{review_id}", headers=teacher_headers)).status_code == 404
    finally:
        await client.delete(f"{BASE}/reviews/{review_id}", headers=student_headers)


@pytest.mark.parametrize("name, content, status", [
    ("notes.txt", b"hello", 400),                # Wrong extension
    ("fake.pdf", b"not a pdf at all", 400),      # Right extension, wrong content
    ("empty.pdf", b"", 400),
])
async def test_rejects_invalid_files(client, student_headers, stub_graph, name, content, status):
    resp = await client.post(f"{BASE}/upload", headers=student_headers,
                             files={"file": (name, content, "application/pdf")})
    assert resp.status_code == status
    assert stub_graph.states == []


async def test_upload_requires_a_token(client, stub_graph, pdf_bytes):
    resp = await client.post(f"{BASE}/upload", files={"file": ("r.pdf", pdf_bytes, "application/pdf")})
    assert resp.status_code in (401, 403)        # HTTPBearer answers 403 when the header is missing
