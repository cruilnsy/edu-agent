# backend/api/v1/resume.py

import asyncio
import json
import os
import tempfile
import threading
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy import text

from backend.agents.resume.graph import build_resume_graph
from backend.core.logger import get_logger
from backend.core.rate_limit import limit_resume_upload
from backend.dependencies import AsyncSessionLocal, get_current_user

router = APIRouter()
logger = get_logger(__name__)

MAX_PDF_SIZE = 20 * 1024 * 1024                  # 20 MB
RESUME_REVIEW_TIMEOUT_SECONDS = 15 * 60          # A review still processing after 15 min is marked failed
FAILED_MESSAGE = "The review failed. Please upload the resume again."

# ── One compiled graph per thread, to avoid sharing it across threads ──
_graph_local = threading.local()


def _get_graph():
    if not hasattr(_graph_local, "graph"):
        _graph_local.graph = build_resume_graph()
    return _graph_local.graph


# The event loop keeps only weak references to tasks, so hold strong ones here
# until each task finishes; otherwise a running review could be garbage-collected.
_background_tasks: set[asyncio.Task] = set()


async def _mark_review_failed(review_id: str, error_msg: str) -> None:
    """Mark a review as failed. Idempotent: only changes rows that are still processing."""
    async with AsyncSessionLocal() as session:
        await session.execute(
            text("""
                UPDATE resume_reviews
                SET status = 'failed', error_msg = :error_msg, updated_at = NOW()
                WHERE id = :review_id AND status = 'processing'
            """),
            {"review_id": review_id, "error_msg": error_msg[:1000]},
        )
        await session.commit()


@router.post("/upload", status_code=202, dependencies=[Depends(limit_resume_upload)])         # 202 Accepted: processing in the background
async def upload_resume(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user),
):
    """Upload a PDF resume, start the review in the background, and return its review_id."""
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    review_id  = str(uuid.uuid4())
    student_id = current_user["user_id"]
    tenant_id  = current_user["tenant_id"]

    # 1. Read and validate the file
    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="The uploaded file is empty")
    if len(file_bytes) > MAX_PDF_SIZE:
        raise HTTPException(status_code=413, detail="File too large; the limit is 20 MB")
    if not file_bytes.startswith(b"%PDF-"):       # The extension alone can be faked
        raise HTTPException(status_code=400, detail="The file is not a valid PDF")

    # Name comes from our UUID, never from the user's filename, so it cannot escape the temp dir
    tmp_path = os.path.join(tempfile.gettempdir(), f"{review_id}_upload.pdf")
    with open(tmp_path, "wb") as f:
        f.write(file_bytes)
    logger.info("upload_resume.file_saved", review_id=review_id, size=len(file_bytes))

    # 2. Create the review row (status=processing)
    async with AsyncSessionLocal() as session:
        await session.execute(
            text("""
                INSERT INTO resume_reviews (id, tenant_id, student_id, pdf_minio_path, status)
                VALUES (:id, :tenant_id, :student_id, :pdf_minio_path, 'processing')
            """),
            {"id": review_id, "tenant_id": tenant_id, "student_id": student_id,
             "pdf_minio_path": f"resumes/{student_id}/{review_id}.pdf"},
        )
        await session.commit()

    # 3. Run the graph in the background
    initial_state = {
        "messages": [], "student_id": student_id, "tenant_id": tenant_id,
        "review_id": review_id, "pdf_minio_path": "", "pdf_local_path": tmp_path,
        "raw_text": "", "page_count": 0, "structured": None,
        "dimension_scores": [], "weighted_score": 0.0, "issues": [],
        "summary": None, "fallback_used": False, "structured_output": None,
    }

    def _on_task_done(t: asyncio.Task):
        """Release the task, delete the temp file, and mark the review failed on error."""
        _background_tasks.discard(t)
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        failed = False
        if t.cancelled():                             # e.g. the server restarted
            failed = True
            logger.warning("resume.background_task_cancelled", review_id=review_id)
        elif (exc := t.exception()) is not None:
            failed = True                             # Details go to the log, not to the user
            logger.error("resume.background_task_failed", review_id=review_id,
                         error=str(exc), exc_info=exc)
        if failed:
            try:
                asyncio.get_running_loop().create_task(_mark_review_failed(review_id, FAILED_MESSAGE))
            except RuntimeError:
                pass                                  # Loop closed: the timeout check in get_review handles it

    task = asyncio.create_task(_get_graph().ainvoke(initial_state))
    _background_tasks.add(task)
    task.add_done_callback(_on_task_done)
    logger.info("upload_resume.task_started", review_id=review_id)

    return {
        "review_id": review_id, "status": "processing",
        "message": "Resume uploaded. The review usually takes 30-60 seconds.",
    }


@router.get("/reviews/{review_id}")
async def get_review(review_id: str, current_user: dict = Depends(get_current_user)):
    """Return the review status or result: processing / done / failed, or 404."""
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            text("""
                SELECT id, status, scores, issues, summary, error_msg, created_at, updated_at
                FROM resume_reviews
                WHERE id = :review_id AND student_id = :student_id
            """),
            {"review_id": review_id, "student_id": current_user["user_id"]},
        )
        row = result.mappings().fetchone()

    if not row:                                       # Missing, or owned by someone else
        raise HTTPException(status_code=404, detail="Review not found")

    if row["status"] == "processing":
        # Safety net for reviews stuck in processing after an interrupted task
        last_ts = row["updated_at"] or row["created_at"]
        if isinstance(last_ts, datetime):
            elapsed = (datetime.now(timezone.utc) - last_ts).total_seconds()
            if elapsed >= RESUME_REVIEW_TIMEOUT_SECONDS:
                timeout_msg = "The review timed out or was interrupted. Please upload again."
                await _mark_review_failed(review_id, timeout_msg)
                return {"review_id": review_id, "status": "failed", "error_msg": timeout_msg}
        return {"review_id": review_id, "status": "processing"}

    if row["status"] == "failed":
        return {"review_id": review_id, "status": "failed", "error_msg": row.get("error_msg") or FAILED_MESSAGE}

    # status == done. asyncpg already decodes JSONB into dict/list.
    def _as_json(value, empty):
        if value is None:
            return empty
        return value if isinstance(value, (dict, list)) else json.loads(value)

    scores_data = _as_json(row["scores"], {})
    return {
        "review_id": review_id, "status": "done",
        "weighted_score": scores_data.get("weighted_score", 0),
        "dimension_scores": scores_data.get("dimension_scores", []),
        "issues": _as_json(row["issues"], []),
        "summary": _as_json(row["summary"], {}),
    }


@router.delete("/reviews/{review_id}", status_code=204)
async def delete_review(review_id: str, current_user: dict = Depends(get_current_user)):
    """Delete a review. Filtering by student_id means users can delete only their own."""
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            text("""
                DELETE FROM resume_reviews
                WHERE id = :review_id AND student_id = :student_id
            """),
            {"review_id": review_id, "student_id": current_user["user_id"]},
        )
        await session.commit()
    if result.rowcount == 0:                          # Missing, or not theirs
        raise HTTPException(status_code=404, detail="Review not found")


@router.get("/reviews")
async def list_reviews(current_user: dict = Depends(get_current_user)):
    """List the current user's reviews, newest first (summary only)."""
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            text("""
                SELECT id, status, created_at,
                       (scores::jsonb ->> 'weighted_score')::float AS weighted_score
                FROM resume_reviews
                WHERE student_id = :student_id
                ORDER BY created_at DESC
                LIMIT 50
            """),
            {"student_id": current_user["user_id"]},
        )
        rows = result.mappings().all()

    items = [
        {"review_id": row["id"], "status": row["status"],
         "created_at": row["created_at"].isoformat() if row["created_at"] else None,
         "weighted_score": row["weighted_score"]}
        for row in rows
    ]
    return {"items": items, "total": len(items)}
