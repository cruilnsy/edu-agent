# scripts/manual_tests/test_qa.py
# End-to-end Q&A and unified assistant against a running backend:
#   /qa/chat: small talk, a knowledge base question, an uncovered question, a two-turn
#   follow-up and the session history; /qa/chat/stream; /chat/stream (unified assistant)
#   for a rule reply, a Q&A route, an agent page link and the job preparation plan.
# The answers depend on the LLM, so only the answer mode, sources and event order are checked.
# Retrieval and answer quality are measured by scripts/evaluation/eval_qa.py.
#
# Needs: the English knowledge base (scripts/build_knowledge_base.py) and seed passwords in .env.local.
# Run from the project root:
#     python -m scripts.manual_tests.test_qa
#     python -m scripts.manual_tests.test_qa --port 8011

import argparse
import json
import os
import uuid

import httpx
from dotenv import load_dotenv

load_dotenv(".env.local")

STUDENT = ("student01", os.environ["SEED_STUDENT_PASSWORD"])
COVERED = "What does a CORS preflight request check, and which HTTP method does it use?"
UNCOVERED = "How does Kubernetes horizontal pod autoscaling decide when to add pods?"


def section(title: str) -> None:
    print(f"\n{'=' * 60}\n  {title}\n{'=' * 60}")


class Client:
    def __init__(self, port: int):
        self.base = f"http://localhost:{port}/api/v1"
        resp = httpx.post(f"{self.base}/auth/login", json=dict(zip(("username", "password"), STUDENT)),
                          trust_env=False)
        resp.raise_for_status()
        self.headers = {"Authorization": f"Bearer {resp.json()['access_token']}"}

    def chat(self, message: str, session_id: str) -> dict:
        resp = httpx.post(f"{self.base}/qa/chat", headers=self.headers, timeout=120, trust_env=False,
                          json={"session_id": session_id, "message": message})
        resp.raise_for_status()
        result = resp.json()
        print(f"  [{result['answer_mode']}, confidence {result['confidence']:.2f}, {len(result['sources'])} sources] "
              f"{result['answer'][:160]!r}")
        return result

    def stream(self, path: str, body: dict) -> list[dict]:
        """All SSE events of one request; tokens are printed as they arrive."""
        events = []
        with httpx.stream("POST", f"{self.base}{path}", headers=self.headers, json=body, timeout=120,
                          trust_env=False) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if line.startswith("data:") and line[5:].strip():
                    event = json.loads(line[5:])
                    events.append(event)
                    if event["type"] == "token":
                        print(event["content"], end="", flush=True)
        print(f"\n  events: {[e['type'] for e in events if e['type'] != 'token']}")
        return events


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8010)
    args = parser.parse_args()
    client = Client(args.port)
    run = uuid.uuid4().hex[:8]                 # Fresh sessions on every run

    section("1. Small talk takes the general path")
    assert client.chat("Hi! What can you help me with?", f"qa-{run}-a")["answer_mode"] == "general"

    section("2. Knowledge base question")
    result = client.chat(COVERED, f"qa-{run}-b")
    assert result["answer_mode"] == "rag" and result["sources"], "expected a knowledge base answer with sources"

    section("3. Follow-up in the same session, and the history")
    follow = client.chat("And which response headers does the server send back?", f"qa-{run}-b")
    assert "access-control" in follow["answer"].lower(), "the follow-up lost the CORS context"
    history = httpx.get(f"{client.base}/qa/sessions/qa-{run}-b/history", headers=client.headers,
                        trust_env=False).json()
    print(f"  history: {history['total_turns']} turns")
    assert history["total_turns"] == 2

    section("4. Question the knowledge base does not cover")
    result = client.chat(UNCOVERED, f"qa-{run}-c")
    assert result["answer_mode"] != "rag", "an uncovered question should not take the knowledge base path"

    section("5. /qa/chat/stream")
    events = client.stream("/qa/chat/stream", {"session_id": f"qa-{run}-d", "message": COVERED})
    types = [e["type"] for e in events]
    assert types[0] == "progress" and "token" in types and types[-2:] == ["meta", "done"]

    section("6. Unified assistant")
    unified = f"unified_{run}"
    events = client.stream("/chat/stream", {"session_id": unified, "message": "hello"})
    assert [e["type"] for e in events] == ["token", "done"]
    for message, expected in [(COVERED, "qa"), ("Can you review my resume?", "resume"),
                              ("Help me get ready for job hunting.", "multi_agent")]:
        print(f"\n> {message}")
        events = client.stream("/chat/stream", {"session_id": unified, "message": message})
        decision = events[0]
        print(f"  routed to {decision['label']} ({decision['agent_display']}): {decision['reason']}")
        assert decision["type"] == "routing_decision" and decision["label"] == expected
        types = [e["type"] for e in events]
        assert {"qa": "meta", "resume": "guidance", "multi_agent": "pipeline_plan"}[expected] in types
    print("\nEnd-to-end Q&A and assistant flow passed")


if __name__ == "__main__":
    main()
