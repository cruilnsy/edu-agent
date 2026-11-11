# scripts/generate_classifier_data.py
# Generate English training data for the QA query classifier with the LLM.
#
#   specialized: questions about the knowledge base topics (scripts/knowledge/sources.json), answered with RAG
#   general:     everything else (small talk, other fields, other programming ecosystems), answered by the LLM
#
# The labels follow the topic, not whether the 12 pages happen to contain the answer: an in-topic question
# the pages do not cover still goes to RAG, and the reranker's minimum score decides what is usable.
# Language-specific questions about other ecosystems (Java, Go, Rust, ...) are general even when they
# mention HTTP or JWT; protocol-level questions without such an ecosystem are specialized.
#
# Usage: python -m scripts.generate_classifier_data            # writes backend/core/data/training_data.jsonl

import argparse
import asyncio
import json
import random
import re
from pathlib import Path

from langchain_core.messages import HumanMessage

from backend.core.llm_factory import get_llm

OUT_PATH = Path(__file__).resolve().parents[1] / "backend" / "core" / "data" / "training_data.jsonl"
PER_BATCH = 25
CONCURRENCY = 5

SPECIALIZED = [
    "Python language basics: types, functions, classes, decorators, generators, exceptions, typing",
    "Python asyncio: coroutines, tasks, gather, TaskGroup, the event loop, timeouts",
    "FastAPI and Python web backends: routing, dependencies, Pydantic models, middleware, testing",
    "Databases for backend apps: SQL, PostgreSQL, indexes, transactions, ORMs such as SQLAlchemy",
    "HTTP and REST API design: methods, status codes, headers, versioning, pagination",
    "Web authentication and security: JWT, OAuth2, cookies, sessions, CORS, password hashing, XSS, CSRF",
    "Docker and deploying web apps: Dockerfiles, images, multi-stage builds, docker compose",
    "JavaScript fundamentals: promises, async/await, closures, modules, the browser event loop",
    "TypeScript: everyday types, interfaces, type aliases, generics, narrowing, utility types",
    "React: components, props, state, hooks, effects, custom hooks, rendering",
    "Frontend basics: HTML, CSS, the DOM, fetch, forms, Vite and bundlers",
    "LLM and transformer fundamentals: tokens, embeddings, attention, encoder vs decoder models",
    "RAG and prompt engineering: chunking, vector search, reranking, prompt techniques, hallucination",
    "AI agents and frameworks: tool calling, LangChain, LangGraph state graphs, memory, fine-tuning basics",
]

GENERAL = [
    "Greetings, small talk, thanks, and questions about the assistant itself",
    "Date, time, weather, news, and current events",
    "Everyday life: health, food, cooking, travel, relationships, hobbies",
    "School math and arithmetic: algebra, geometry, percentages, unit conversions",
    "Science, history, geography, and general trivia",
    "Career and job search: resumes, interview tips, salary negotiation, choosing a career",
    "Other programming languages: Java and Spring, C and C++, Go, Rust, C# and .NET, PHP, Ruby",
    "Mobile, game, and embedded development: Swift and iOS, Kotlin and Android, Unity, Arduino",
    "Computer use and IT support: Windows, macOS, printers, Wi-Fi, Excel, Word, keyboard shortcuts",
    "Language-agnostic algorithms and CS theory: Big-O, sorting, graphs, dynamic programming",
    "Business, marketing, personal finance, investing, and cryptocurrency",
    "Writing, translation, grammar, and learning foreign languages",
]

PROMPT = """Write {n} different questions that a student might type into a learning assistant's chat box.
Topic: {topic}

Vary them: short keyword queries and full sentences, beginner and advanced, "how do I", "why",
"what is the difference", error messages pasted in, casual tone, some lowercase or with small typos.
Do not number them. Output one question per line and nothing else."""


async def generate(llm, topic: str, n: int, semaphore: asyncio.Semaphore) -> list[str]:
    async with semaphore:
        resp = await llm.ainvoke([HumanMessage(content=PROMPT.format(n=n, topic=topic))])
    text = resp.text if hasattr(resp, "text") and not callable(resp.text) else str(resp.content)
    lines = [re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip() for line in text.splitlines()]
    return [line for line in lines if 5 <= len(line) <= 300]


async def main_async(batches: int, seed: int) -> None:
    llm = get_llm("qa", temperature=1.0)
    semaphore = asyncio.Semaphore(CONCURRENCY)
    jobs = [(label, topic) for label, topics in (("specialized", SPECIALIZED), ("general", GENERAL))
            for topic in topics for _ in range(batches)]
    results = await asyncio.gather(*(generate(llm, topic, PER_BATCH, semaphore) for _, topic in jobs))

    seen, rows = set(), []
    for (label, _), questions in zip(jobs, results):
        for q in questions:
            key = re.sub(r"\W+", " ", q.lower()).strip()
            if key not in seen:
                seen.add(key)
                rows.append({"text": q, "label": label})
    random.Random(seed).shuffle(rows)

    OUT_PATH.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    counts = {label: sum(r["label"] == label for r in rows) for label in ("specialized", "general")}
    print(f"wrote {len(rows)} rows to {OUT_PATH} {counts}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate classifier training data.")
    parser.add_argument("--batches", type=int, default=2, help=f"Batches of {PER_BATCH} per topic")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    asyncio.run(main_async(args.batches, args.seed))


if __name__ == "__main__":
    main()
