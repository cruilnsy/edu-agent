# scripts/manual_tests/test_interview.py
# End-to-end mock interview against a running backend (port 8010), answered by a simulated
# candidate from datasets/interview/candidates.json:
#   start -> one streamed turn -> chat until finished -> report -> checks:
#   another user gets 404, a malformed id gets 400, someone else's resume review gets 404,
#   a finished interview cannot continue (409).
#
# Needs: python -m scripts.seed_interview_questions, and seed passwords in .env.local.
# Run from the project root:
#     python -m scripts.manual_tests.test_interview
#     python -m scripts.manual_tests.test_interview --candidate c05

import argparse
import asyncio
import json
import os
import sys
import uuid

import httpx
from dotenv import load_dotenv

load_dotenv(".env.local")

from scripts.evaluation.interview_sim import candidate_reply, load_candidates  # noqa: E402  (needs the env loaded first)

API_URL = "http://localhost:{port}/api/v1"    # Port 8000 is taken by another app on this machine
BASE_URL = LOGIN_URL = ""                      # Set in main() from --port
STUDENT = ("student01", os.environ["SEED_STUDENT_PASSWORD"])
TEACHER = ("teacher01", os.environ["SEED_TEACHER_PASSWORD"])
MAX_REPLIES = 40


def login(username: str, password: str) -> dict:
    resp = httpx.post(LOGIN_URL, json={"username": username, "password": password}, trust_env=False)
    resp.raise_for_status()
    return {"Authorization": f"Bearer {resp.json()['access_token']}"}


def section(title: str) -> None:
    print(f"\n{'=' * 60}\n  {title}\n{'=' * 60}")


def expect(resp: httpx.Response, code: int, what: str) -> None:
    print(f"  {what} -> {resp.status_code}")
    assert resp.status_code == code, resp.text


def stream_turn(headers: dict, session_id: str, message: str) -> dict:
    """Send one message to the SSE endpoint; print the tokens and return the done event."""
    done = None
    with httpx.stream("POST", f"{BASE_URL}/sessions/{session_id}/chat/stream", headers=headers,
                      json={"message": message}, timeout=60, trust_env=False) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line.startswith("data:") or not line[5:].strip():
                continue
            event = json.loads(line[5:])
            if event["type"] == "token":
                print(event["content"], end="", flush=True)
            elif event["type"] == "done":
                done = event
            elif event["type"] == "error":
                sys.exit(f"\nStream error: {event['message']}")
    print()
    assert done is not None, "no done event"
    return done


async def run() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", default="c02", help="Candidate id in datasets/interview/candidates.json")
    parser.add_argument("--port", type=int, default=8010)
    args = parser.parse_args()
    global BASE_URL, LOGIN_URL
    BASE_URL = API_URL.format(port=args.port) + "/interview"
    LOGIN_URL = API_URL.format(port=args.port) + "/auth/login"
    candidate = next(c for c in load_candidates() if c["id"] == args.candidate)
    student, teacher = login(*STUDENT), login(*TEACHER)

    section("1. Input checks")
    expect(httpx.post(f"{BASE_URL}/sessions", headers=student, trust_env=False,
                      json={"target_position": "AI Engineer", "resume_review_id": str(uuid.uuid4())}),
           404, "someone else's (or a missing) resume review")
    expect(httpx.post(f"{BASE_URL}/sessions", headers=student, trust_env=False, json={"target_position": " "}),
           422, "empty position")
    expect(httpx.post(f"{BASE_URL}/sessions/not-a-uuid/chat", headers=student, trust_env=False,
                      json={"message": "hi"}), 400, "malformed session id")

    section("2. Start")
    resp = httpx.post(f"{BASE_URL}/sessions", headers=student, json={"target_position": candidate["target_position"]},
                      timeout=60, trust_env=False)
    resp.raise_for_status()
    session_id, opening = resp.json()["session_id"], resp.json()["message"]
    print(f"session_id: {session_id}\nInterviewer: {opening}")
    dialogue = [("Interviewer", opening)]

    section("3. Another user cannot use the session")
    expect(httpx.post(f"{BASE_URL}/sessions/{session_id}/chat", headers=teacher, json={"message": "hi"},
                      trust_env=False), 404, "teacher /chat")
    expect(httpx.post(f"{BASE_URL}/sessions/{session_id}/chat/stream", headers=teacher, json={"message": "hi"},
                      trust_env=False), 404, "teacher /chat/stream")
    expect(httpx.get(f"{BASE_URL}/sessions/{session_id}/report", headers=teacher, trust_env=False),
           404, "teacher /report")
    expect(httpx.get(f"{BASE_URL}/sessions/{session_id}/report", headers=student, trust_env=False),
           409, "report before the end")

    section("4. Interview (first turn streamed)")
    result = None
    for reply_no in range(MAX_REPLIES):
        answer = await candidate_reply(candidate, dialogue, reply_no)
        print(f"\nCandidate: {answer}")
        if reply_no == 0:
            print("Interviewer (stream): ", end="")
            result = stream_turn(student, session_id, answer)
        else:
            resp = httpx.post(f"{BASE_URL}/sessions/{session_id}/chat", headers=student, json={"message": answer},
                              timeout=60, trust_env=False)
            resp.raise_for_status()
            result = resp.json()
            print(f"Interviewer [{result['current_stage']}]: {result['reply']}")
        dialogue += [("Candidate", answer), ("Interviewer", result["reply"])]
        if result["is_finished"]:
            break
    else:
        sys.exit("The interview did not finish")
    print(f"\nReport summary: {result['report_summary']}")

    section("5. Report, and no change after the end")
    report = httpx.get(f"{BASE_URL}/sessions/{session_id}/report", headers=student, trust_env=False)
    report.raise_for_status()
    body = report.json()
    print(f"  overall {body['overall_score']}, dimensions: "
          + ", ".join(f"{d['dimension']} {d['score']}" for d in body["dimensions"]))
    assert body["overall_score"] == result["report_summary"]["overall_score"] and not body["fallback"]
    expect(httpx.post(f"{BASE_URL}/sessions/{session_id}/chat", headers=student, json={"message": "one more"},
                      trust_env=False), 409, "chat after the end")
    sessions = httpx.get(f"{BASE_URL}/sessions", headers=student, trust_env=False).json()["items"]
    assert any(s["session_id"] == session_id and s["status"] == "finished" for s in sessions)
    print("\nEnd-to-end interview flow passed")


def main() -> None:
    # One event loop for the whole run: the cached LLM client is bound to the loop it first ran in
    asyncio.run(run())


if __name__ == "__main__":
    main()
