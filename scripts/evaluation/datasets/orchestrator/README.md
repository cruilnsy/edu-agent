# Orchestrator routing eval set

`routing.jsonl` has one message per line: `id`, `text`, and the gold `label`. The labels are what `backend/core/orchestrator.route_message` should return.

| Label | Meaning | Answered by |
|---|---|---|
| hello, thanks, bye, identity, capability | Social or meta message | Rules: a fixed reply, no LLM call |
| qa | A question to answer now, including general questions about resumes, exams or interviews | The Q&A agent, inside the chat |
| exam | Submit an exam for grading, or see exam results | A link to the Exam page |
| resume | Review the user's own resume | A link to the Resume page |
| interview | Practise a mock interview | A link to the Interview page |
| multi_agent | End-to-end job preparation, or both a resume review and an interview | The two-step plan |
| clarify | Too vague to choose | A question back to the user |

Judgment calls are made by the definitions in `ROUTE_SYSTEM`:
- "How should I prepare for a system design interview?" is `qa` (a question), not `interview` (a practice session).
- "I'm stuck." is `clarify`: there is no topic to answer.
- A message with thanks plus a question ("Thanks, and how do indexes work?") is `qa`, not `thanks`.

Run with `python -m scripts.evaluation.eval_orchestrator`. The eval reports:
- accuracy and macro F1 over all labels;
- per-label recall and the confusion matrix;
- rule hits on agent messages (target 0);
- the fallback rate;
- P95 latency and tokens of the LLM-routed messages.
