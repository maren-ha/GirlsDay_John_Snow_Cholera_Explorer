import pytest

from src.guided_mode import (
    GUIDED_STEPS,
    build_default_guided_answers,
    build_default_session_state,
    backfill_session_state_defaults,
    count_completed_steps,
    get_current_guided_step,
    get_next_step_id,
    get_previous_step_id,
    get_step_by_id,
    is_step_complete,
    update_guided_answer,
)


def test_guided_steps_define_the_structured_step_order():
    assert [step["step_id"] for step in GUIDED_STEPS] == [
        "overview",
        "distribution",
        "comparison",
        "hypothesis",
        "stats",
        "conclusion",
    ]

    for step in GUIDED_STEPS:
        assert {"step_id", "title_msg_id", "prompt_msg_id", "fields", "completion_rule", "report_section"} <= set(step)
        assert isinstance(step["fields"], list)
        assert step["fields"]
        for field in step["fields"]:
            assert {"field_id", "label_msg_id", "input_type"} <= set(field)


def test_build_default_guided_answers_uses_the_field_id_contract():
    defaults = build_default_guided_answers()

    assert defaults == {
        "overview_observation": "",
        "distribution_observation": "",
        "comparison_observation": "",
        "hypothesis": "",
        "stats_interpretation": "",
        "conclusion": "",
    }


def test_build_default_session_state_matches_the_shared_contract():
    defaults = build_default_session_state()

    assert defaults == {
        "language": "de",
        "guided_mode_enabled": False,
        "guided_step": "overview",
        "student_name": "",
        "group_name": "",
        "guided_answers": build_default_guided_answers(),
        "selected_plots": [],
        "report_ready_state": {"last_error": None, "last_export_language": None},
    }


def test_get_current_guided_step_reads_session_state_with_overview_fallback():
    assert get_current_guided_step({}) == "overview"
    assert get_current_guided_step({"guided_step": "comparison"}) == "comparison"
    assert get_current_guided_step({"guided_step": "not-a-step"}) == "overview"
    assert get_current_guided_step(None) == "overview"


def test_step_navigation_helpers_follow_the_defined_order():
    assert get_step_by_id("overview")["step_id"] == "overview"
    assert get_step_by_id("missing") is None
    assert get_next_step_id("overview") == "distribution"
    assert get_next_step_id("conclusion") is None
    assert get_previous_step_id("distribution") == "overview"
    assert get_previous_step_id("overview") is None


def test_completion_helpers_use_the_field_id_contract():
    answers = build_default_guided_answers()
    step = get_step_by_id("overview")

    assert not is_step_complete(step, answers)
    assert count_completed_steps(answers) == 0

    update_guided_answer({"guided_answers": answers}, "overview_observation", "A strong clue")

    assert is_step_complete(step, answers)
    assert count_completed_steps(answers) == 1


def test_update_guided_answer_persists_into_the_nested_session_state():
    session_state = {"guided_answers": build_default_guided_answers()}

    update_guided_answer(session_state, "hypothesis", "Pump exposure matters")

    assert session_state["guided_answers"]["hypothesis"] == "Pump exposure matters"
    assert session_state["guided_answers"]["overview_observation"] == ""


def test_backfill_session_state_defaults_backfills_missing_nested_defaults_without_overwriting_values():
    session_state = {
        "language": "en",
        "guided_mode_enabled": True,
        "guided_answers": {
            "overview_observation": "Already written",
            "hypothesis": "Keep this",
        },
        "report_ready_state": {
            "last_error": "previous problem",
        },
    }

    backfill_session_state_defaults(session_state)

    assert session_state["language"] == "en"
    assert session_state["guided_mode_enabled"] is True
    assert session_state["guided_step"] == "overview"
    assert session_state["guided_answers"] == {
        "overview_observation": "Already written",
        "distribution_observation": "",
        "comparison_observation": "",
        "hypothesis": "Keep this",
        "stats_interpretation": "",
        "conclusion": "",
    }
    assert session_state["report_ready_state"] == {
        "last_error": "previous problem",
        "last_export_language": None,
    }


def test_get_current_guided_step_does_not_swallow_arbitrary_lookup_errors():
    class ExplodingSessionState:
        def get(self, key, default=None):
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        get_current_guided_step(ExplodingSessionState())
