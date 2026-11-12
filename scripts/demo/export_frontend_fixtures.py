# scripts/demo/export_frontend_fixtures.py
# Records the API responses the read-only frontend demo (VITE_DEMO_MODE=true, deployed as a
# static site) serves instead of calling a backend. Loads the demo data first, then saves the
# real responses of the student pages, so the demo and the app share one data shape.
#
#     python -m scripts.demo.export_frontend_fixtures
#
# Rerun after seed_demo.py's data or any of these endpoints change. Like seed_demo.py it
# resets student01.

import asyncio
import json
from pathlib import Path

import httpx
from sqlalchemy import text

from backend.api.v1.auth import _create_access_token
from backend.dependencies import AsyncSessionLocal
from scripts.demo.seed_demo import STUDENT, seed

OUT_PATH = Path(__file__).resolve().parents[2] / "frontend" / "src" / "demo" / "fixtures.json"
DEMO_USER_ID = "demo-student"          # Replaces the local account id in every response


async def main() -> None:
    await seed()
    async with AsyncSessionLocal() as db:
        user = (await db.execute(text("SELECT id, role, tenant_id FROM users WHERE username = :u"),
                                 {"u": STUDENT})).mappings().one()
        teacher_ids = [str(r) for r in (await db.execute(text("SELECT id FROM users WHERE role <> 'student'"))).scalars()]
    user_id = str(user["id"])
    # A short-lived token signed locally, only to read this student's pages
    token = _create_access_token({"sub": user_id, "role": user["role"], "tenant_id": user["tenant_id"]}, 5)
    headers = {"Authorization": f"Bearer {token}"}

    from backend.main import app
    fixtures: dict[str, object] = {}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://demo/api/v1") as client:
        async def get(path: str):
            resp = await client.get(path, headers=headers)
            resp.raise_for_status()
            fixtures[f"GET {path}"] = resp.json()
            return resp.json()

        await get("/auth/me")
        for review in (await get("/resume/reviews"))["items"]:
            await get(f"/resume/reviews/{review['review_id']}")
        await get("/exam/exams")
        for submission in (await get("/exam/my-submissions"))["items"]:
            await get(f"/exam/my-submissions/{submission['submission_id']}")
        for session in (await get("/interview/sessions"))["items"]:
            if session["status"] == "finished":
                await get(f"/interview/sessions/{session['session_id']}/report")

    raw = json.dumps(fixtures, indent=2, ensure_ascii=False).replace(user_id, DEMO_USER_ID)
    for other in teacher_ids:
        if other in raw:
            raise SystemExit("A staff account id appears in the responses; check the endpoints")
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(raw + "\n", encoding="utf-8")
    print(f"wrote {len(fixtures)} responses to {OUT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
