# backend/agents/resume/prompts.py
# Resume review prompts for the US tech job market.
# Placeholders must match what nodes.py passes; tests/unit/test_resume_prompts.py checks them.
# Dimension prompts receive {focus} {structured_summary} {resume_text} {today} {page_count}.

SYSTEM_PROMPT = """You are a senior technical recruiter and engineering hiring manager at a US tech company.
You review resumes for software engineering roles, from new grad to senior, and know US resume
conventions, how ATS (applicant tracking systems) filter candidates, and how recruiters skim a
resume in 6-10 seconds.
Be direct and specific: point to the exact problem and how to fix it. No vague praise.
Quote the resume verbatim when you refer to it.
The structured summary includes facts computed in code: total years of experience and whether
roles are in order. Trust them over your own date arithmetic. "N+ years" is a floor: a total at
or above N is accurate. Do not mention it at all: not as an inconsistency, not as underselling,
and do not suggest raising the number.
Judge only what the resume says. Never infer or mention the candidate's nationality, ethnicity,
immigration or visa status, gender, age or religion from their name, school, location or anything
else, and never tie a suggestion to such a guess. A neutral, optional suggestion to add a
work-authorization line ("if applicable, add ...") is allowed, as long as it assumes nothing about
the candidate's status."""

_TODAY_RULE = ('Today is {today}. Any date before today is in the past, not the future. '
               '"Present" means up to today.')
_PAGE_RULE = ("The PDF has {page_count} page(s). Page breaks are not marked in the text; judge "
              "length from this page count, never by estimating from the text.")


EXTRACT_STRUCTURED_PROMPT = """Extract structured information from this resume.

[Resume]
{resume_text}

Rules:
- Keep all text verbatim; do not rewrite, shorten, or translate bullets.
- List each technology separately (e.g. "FastAPI", "PostgreSQL").
- Normalize dates to "Mon YYYY - Mon YYYY" (e.g. "Sep 2022 - Dec 2025"). Keep "Present" as is;
  keep year-only dates as written.
- Put the Summary or Objective section in self_intro.
- Use an empty string for missing fields, never "N/A" or "Unknown".
- Highlights: only sentences with concrete numbers (e.g. "reduced p95 latency by 40%")."""


def _dimension_prompt(title: str, checklist: str, rubric: str, context: str = "") -> str:
    """Shared frame for the six dimension reviews; only the checklist and rubric differ."""
    context = f"{context}\n\n" if context else ""
    return f"""Review this resume on **{title}**.

Focus: {{focus}}

[Structured summary]
{{structured_summary}}

[Resume]
{{resume_text}}

{context}What to check:
{checklist}

Score 0-100:
{rubric}"""


DIMENSION_REVIEW_PROMPTS = {
    "project_depth": _dimension_prompt(
        "experience and project depth",
        """- Each bullet shows what was done, how (key technical decisions), and the impact.
- Personal ownership is clear (Led / Designed / Built), not just what the team did.
- Scale and complexity are visible: users, data volume, system size, cross-team work.
- Senior roles show ownership, architecture decisions, and mentoring. For new grads, side or
  open-source projects link to GitHub or a live demo.""",
        """- 90-100: every role shows clear technical decisions, ownership, and measurable impact
- 70-89: most roles show ownership and impact; a few bullets are generic
- 50-69: mostly duties ("Responsible for..."); depth and impact are hard to see
- 30-49: a list of tasks with no sign of technical difficulty
- 0-29: very thin, or unrelated to the target role"""),

    "tech_match": _dimension_prompt(
        "technical fit",
        """- Skills match the keywords common in job descriptions for the target role (ATS filters on them).
- Every listed skill is backed by real use in Experience or Projects.
- Skills are grouped (Languages / Frameworks / Cloud & DevOps / Databases) for fast scanning.
- No proficiency bars or ratings; no filler such as Microsoft Office or soft skills.
- No outdated or off-target technologies diluting the core stack.""",
        """- 90-100: core skills match the role closely and are all backed by experience
- 70-89: main skills match; most are backed by experience
- 50-69: partial match, or a long skill list with little evidence
- 30-49: weak match, or many irrelevant technologies
- 0-29: almost no relevant experience"""),

    "expression": _dimension_prompt(
        "writing",
        """- Bullets start with strong action verbs (Built, Reduced, Migrated), not "Responsible for",
  "Worked on", or "Helped with".
- Past tense for past roles, present tense for the current role.
- No first-person pronouns (I, my, we); concise bullets of one or two lines, not paragraphs.
- No spelling or grammar errors; technology names cased correctly (JavaScript, PostgreSQL).
- No filler buzzwords (hard-working, team player, passionate, detail-oriented).""",
        """- 90-100: strong verbs, consistent tense, concise, error-free
- 70-89: clean overall; occasional weak verb or tense slip
- 50-69: many weak verbs or buzzwords, mixed tenses, some errors
- 30-49: frequent errors or awkward phrasing that hurts credibility
- 0-29: hard to understand"""),

    "structure": _dimension_prompt(
        "structure and formatting",
        """- Length: one page under ~10 years of experience; two pages at most for senior candidates.
- Roles in reverse-chronological order by end date. This is checked in code: use the "Role order"
  line in the structured summary and do not judge the order yourself. Education first for new
  grads; Experience first otherwise.
- Header: name, email, phone, city/state, LinkedIn/GitHub.
- No photo, age or date of birth, gender, marital status, national origin, or ID numbers; many
  US employers discard resumes that include them. A work-authorization line ("U.S. Citizen",
  "Green Card holder", "Authorized to work in the U.S.; no sponsorship required") is allowed and
  often helps; never flag it as protected information. No "References available upon request".
- ATS-friendly: no tables, multiple columns, text boxes, or icons; key details not in
  headers/footers; standard section titles (Experience, Education, Skills, Projects).
- The strongest content sits in the top half of the first page.""",
        """- 90-100: right length, logical order, ATS-friendly; highlights stand out at a glance
- 70-89: sound structure with minor formatting or length issues
- 50-69: wrong length, poor section order, or ATS parsing risk
- 30-49: disorganized, key content buried, or includes personal details that should not be there
- 0-29: no clear structure""",
        context=f"[Length] {_PAGE_RULE}"),

    "quantification": _dimension_prompt(
        "quantified impact",
        """- Bullets use numbers to show impact, ideally in the XYZ form: "Accomplished X, as measured
  by Y, by doing Z".
- Typical metrics: latency or throughput, cost saved, revenue or conversion, users, data volume,
  time saved, release frequency, defect rate.
- Numbers have a baseline or scope (from what to what, across what), not a bare percentage.
- Output counts ("built 7 widgets") are not impact; call out the difference.""",
        """- 90-100: nearly every bullet has a concrete number tied to business or technical impact
- 70-89: main roles are quantified; some bullets are qualitative only
- 50-69: a few numbers; mostly qualitative
- 30-49: almost no numbers
- 0-29: no numbers at all"""),

    "authenticity": _dimension_prompt(
        "credibility and consistency",
        """- Technical depth and title (Junior / Senior / Staff) match the years of experience.
- Years claimed in the Summary (e.g. "8+ years"): compare them with the total in the structured
  summary, which is computed from the role dates; do not redo the date arithmetic yourself.
  "N+ years" is a floor, so any total at or above N is accurate (8+ with a 9.6-year total is fine).
  Flag only a clear overstatement (the claim exceeds the total by more than about a year), never
  an understatement.
- The timeline is coherent: no implausible overlap between full-time roles. Gaps over six months
  may deserve a short note, but gaps are common in the US and are not a serious problem. A gap of
  over a year that is still ongoing (nothing on the resume up to today) matters more to recruiters.
- Achievements are plausible for the role (e.g. a junior engineer did not single-handedly
  architect an entire platform).""",
        """- 90-100: title, years, and depth line up; timeline is clear
- 70-89: consistent overall; a few claims are vague
- 50-69: some claims outrun the experience shown, or the timeline is unclear
- 30-49: several claims conflict with years or title
- 0-29: contradictory; actual level cannot be judged""",
        context=f"[Timeline] {_TODAY_RULE}"),
}


DIAGNOSE_ISSUES_PROMPT = f"""Produce an itemized list of problems in this resume, for a candidate applying to US tech roles.

[Structured summary]
{{structured_summary}}

[Resume]
{{resume_text}}

[Issues found per dimension (reference; add any that are missing)]
{{raw_issues}}

[Length] {_PAGE_RULE}

Requirements:
1. Locate each issue precisely (e.g. "Experience - Acme Corp - bullet 2").
2. Priority:
   - high: likely to fail the initial screen, e.g. bullets without impact or numbers, missing core
     keywords for the target role, badly wrong length, ATS parsing problems, photo or age included.
     Never an employment gap, however long (see the timeline rules)
   - medium: weakens the resume, e.g. weak verbs, mixed tenses, skills with no evidence,
     unclear ownership
   - low: polish, e.g. formatting details, missing GitHub/LinkedIn link, certifications
3. Timeline rules (to avoid false alarms):
   - {_TODAY_RULE}
   - A project dated within a job's dates belongs to that job; that is normal.
   - Several projects overlapping one job is normal.
   - An employment gap that has ended is at most low priority. A gap of over a year that is still
     ongoing (nothing on the resume up to today) may be medium, never high.
   - For years of experience, use the total in the structured summary; do not recompute it.
     "N+ years" is a floor; flag a claim only if it clearly exceeds the total, never because it
     is lower than the total.
   - For role order, use the "Role order" line in the structured summary; flag order only if it
     says NOT.
4. Do not flag a work-authorization line ("U.S. Citizen", "Authorized to work in the U.S.");
   it is allowed and helps the candidate.
5. Make each suggestion actionable; where possible, include a rewritten bullet.
6. Merge duplicates and similar issues.
7. Return 5-15 issues."""


GENERATE_SUMMARY_PROMPT = """Write the overall assessment of this resume review.

[Candidate summary]
{structured_summary}

[Dimension scores]
{scores_summary}

[Overall score] {weighted_score} / 100

[High-priority issues]
{high_issues}

[Target role] {target_position}

Write:
1. Highlights: 2-3 real strengths in the resume, backed by numbers where possible.
2. Core improvements: the 2-3 most important changes, drawn mainly from the high-priority issues.
3. Overall comment: 1-2 sentences, as a recruiter would put it; neither inflated nor discouraging.
4. Fit: one sentence on how well the skills and experience fit a {target_position} role."""


DIAGNOSE_THINK_PROMPT = f"""Before writing the issue list, step back and assess the resume as a whole.

[Dimension scores]
{{dimension_scores_summary}}

[Issues found so far]
{{raw_issues}}

[Timeline] {_TODAY_RULE}
If an issue calls a date before today "in the future", it is a false alarm; ignore it.

In 5-8 sentences:
1. What are the one or two weaknesses most likely to fail the initial screen?
2. Do the issues across dimensions share a root cause?
3. Is there a gap between the skills claimed and the skills shown in the experience?
4. Which issues hurt the first impression in a 6-10 second skim, and should be high priority?

Output the analysis only, with no heading or label."""
