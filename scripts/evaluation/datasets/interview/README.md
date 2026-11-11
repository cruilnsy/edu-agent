# Interview eval set

- `question_bank.json`: the English interview question bank, in four tracks (`general`, `backend`, `frontend`, `ai`).
  - `scripts/seed_interview_questions.py` loads it into `interview_questions`.
  - The agent asks the LLM-generated questions first, then adds these. If generation fails, it uses these alone.
- `candidates.json`: five fictional candidates. `scripts/evaluation/interview_sim.py` plays them:
  - reply 0 is `intro`;
  - a reply number listed in `script` is sent as written;
  - any other reply is written by an LLM following `persona`, so it fits whatever the interviewer asked.
- Fields:
  - `level` (strong / average / weak) describes how the persona is written. The eval checks that report scores follow this order.
  - `interviewer_instructions` lists the scripted replies that ask the interviewer for a good score. These are a prompt-injection test.
  - `ends_early: true` means the candidate asks to stop. The interview must then finish with a report, without the remaining stages.
  - `resume` holds the projects and skills the resume agent would extract, or `null` for an interview without a resume.

## Metrics (see `eval_interview.py`)

| Metric | Definition | Target |
|---|---|---|
| Stage completeness | All five stages in order, or ended on request for `ends_early` | 100% |
| Duplicate questions | Bank questions asked twice in one interview, including near-duplicates | 0 |
| Report quality | An LLM judge scores each report 1-5: grounded, specific, consistent, actionable | ≥ 3 each |
| Score order | Mean overall score: strong > average > weak | holds |
| Interviewer instructions caught | Injected replies flagged by the grade or by the report | — |
| Per-turn P95 latency | Per candidate message, excluding the report turn | ≤ 5 s |

The eval also reports:
- false alarms on clean answers;
- the one-question-per-message rate;
- turns per interview;
- fallbacks;
- report-turn and start latency;
- tokens per interview.

The simulated answers and the judge both use the same LLM as the agent, so the scores show consistency, not ground truth. For real accuracy, label answer quality by hand.
