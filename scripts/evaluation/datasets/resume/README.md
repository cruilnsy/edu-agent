# Resume eval set

Ten fictional resumes (`r01.md` - `r10.md`) of mixed quality, all aimed at a Python backend role.
Every name, company, school, email (`example.com`) and phone number (`555-01xx`) is made up.
`tests/unit/test_resume_dataset.py` fails if a real-looking email or phone number appears.

## Labeling (do this before looking at any agent output)

Fill in `labels.json` for each resume:

| Field | Meaning |
|---|---|
| `expert_score` | Overall score from 0 to 100, as a hiring manager for a Python backend role |
| `expert_issues` | The problems you would tell the candidate to fix, one short sentence each (aim for 3-6) |
| `notes` | Optional reasoning |

Rules:

1. Label every resume **before** running the eval. Seeing the agent's answer first biases the labels.
2. Score each resume on its own merits; do not rank them against each other first.
3. Write issues as problems, not fixes (e.g. "No metrics in the project bullets").

### Current status

`labels.json` has been re-labeled from a hiring-manager perspective for a Python backend role
(scores + issue lists for r01-r10). Re-run `eval_resume.py` to get Score MAE and Issue Recall
against these labels.

## Metrics (see `eval_resume.py`)

- **Score MAE**: mean |agent score - expert score|. Target <= 8.
- **Issue Recall**: share of expert issues that the agent also found, matched by an LLM judge. Target >= 0.7.
- **Score spread**: max - min of the agent score across repeated runs of the same resume. Target <= 5.
- **P95 latency**, **fallback rate**, **tokens and cost** per review.
