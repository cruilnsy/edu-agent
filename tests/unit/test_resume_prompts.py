# tests/unit/test_resume_prompts.py
# Prompt templates must format with the arguments the nodes pass, and the timeline prompts
# must carry today's date.

from backend.agents.resume import nodes
from backend.agents.resume.prompts import (
    DIAGNOSE_ISSUES_PROMPT, DIAGNOSE_THINK_PROMPT, DIMENSION_REVIEW_PROMPTS,
)

TODAY = "2030-01-15"


def test_every_dimension_has_a_prompt_that_formats():
    assert set(DIMENSION_REVIEW_PROMPTS) == {d["key"] for d in nodes.SIX_DIMENSIONS}
    for key, template in DIMENSION_REVIEW_PROMPTS.items():
        filled = template.format(resume_text="R", structured_summary="S", focus="F",
                                 today=TODAY, page_count=2)
        assert "{" not in filled, key


def test_timeline_prompts_include_today():
    authenticity = DIMENSION_REVIEW_PROMPTS["authenticity"].format(
        resume_text="R", structured_summary="S", focus="F", today=TODAY, page_count=2)
    diagnose = DIAGNOSE_ISSUES_PROMPT.format(
        resume_text="R", structured_summary="S", raw_issues="I", today=TODAY, page_count=2)
    think = DIAGNOSE_THINK_PROMPT.format(dimension_scores_summary="D", raw_issues="I", today=TODAY)
    for filled in (authenticity, diagnose, think):
        assert TODAY in filled


def test_today_is_an_iso_date():
    assert len(nodes._today()) == 10 and nodes._today()[4] == "-"


def test_total_experience_merges_overlaps_and_reads_common_formats():
    from datetime import date
    today = date(2026, 11, 4)
    roles = [
        {"duration": "Sep 2022 - Dec 2025"},          # 40 months
        {"duration": "2020.05 - 2021.12"},            # 20 months
        {"duration": "06/2020 – 05/2021"},            # Inside the role above: adds nothing
        {"duration": "Jan 2026 - Present"},           # 11 months up to Nov 2026
        {"duration": "unknown"},                      # Unreadable: skipped
    ]
    assert nodes._total_experience_years(roles, today) == (round(71 / 12, 1), 4)


def test_structured_summary_states_the_computed_total():
    summary = nodes._build_structured_summary(
        {"work_experience": [{"duration": "Jan 2015 - Dec 2024"}]})
    assert "10.0 years in total" in summary


def test_role_order_is_checked_by_end_date():
    from datetime import date
    today = date(2026, 11, 4)
    ordered = [{"company": "A", "duration": "Jan 2026 - Present"},
               {"company": "B", "duration": "May 2020 - Dec 2021"},
               {"company": "C", "duration": "Jun 2020 - May 2021"}]   # Starts later, ends earlier
    assert "correct" in nodes._role_order_note(ordered, today)
    swapped = [ordered[0], ordered[2], ordered[1]]
    note = nodes._role_order_note(swapped, today)
    assert note.startswith("Role order: NOT") and '"C"' in note and '"B"' in note


def test_system_prompt_forbids_guessing_personal_traits():
    from backend.agents.resume.prompts import SYSTEM_PROMPT
    assert "Never infer or mention the candidate's nationality" in SYSTEM_PROMPT


def test_optional_work_authorization_suggestion_is_allowed():
    from backend.agents.resume.prompts import SYSTEM_PROMPT
    assert "optional suggestion to add a\nwork-authorization line" in SYSTEM_PROMPT


def test_only_a_long_ongoing_gap_may_be_medium():
    from backend.agents.resume.prompts import DIAGNOSE_ISSUES_PROMPT
    assert "gap that has ended is at most low priority" in DIAGNOSE_ISSUES_PROMPT
    assert "may be medium, never high" in DIAGNOSE_ISSUES_PROMPT
    assert "Never an employment gap, however long" in DIAGNOSE_ISSUES_PROMPT
