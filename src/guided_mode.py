from copy import deepcopy

from src.i18n import get_default_language


GUIDED_STEPS = [
    {
        "step_id": "overview",
        "title_msg_id": "guided.overview.title",
        "prompt_msg_id": "guided.overview.prompt",
        "fields": [
            {
                "field_id": "overview_observation",
                "label_msg_id": "guided.overview.observation_label",
                "input_type": "textarea",
            }
        ],
        "completion_rule": {"type": "all_fields_filled", "field_ids": ["overview_observation"]},
        "report_section": "overview",
    },
    {
        "step_id": "distribution",
        "title_msg_id": "guided.distribution.title",
        "prompt_msg_id": "guided.distribution.prompt",
        "fields": [
            {
                "field_id": "distribution_observation",
                "label_msg_id": "guided.distribution.observation_label",
                "input_type": "textarea",
            }
        ],
        "completion_rule": {"type": "all_fields_filled", "field_ids": ["distribution_observation"]},
        "report_section": "distribution",
    },
    {
        "step_id": "comparison",
        "title_msg_id": "guided.comparison.title",
        "prompt_msg_id": "guided.comparison.prompt",
        "fields": [
            {
                "field_id": "comparison_observation",
                "label_msg_id": "guided.comparison.observation_label",
                "input_type": "textarea",
            }
        ],
        "completion_rule": {"type": "all_fields_filled", "field_ids": ["comparison_observation"]},
        "report_section": "comparison",
    },
    {
        "step_id": "hypothesis",
        "title_msg_id": "guided.hypothesis.title",
        "prompt_msg_id": "guided.hypothesis.prompt",
        "fields": [
            {
                "field_id": "hypothesis",
                "label_msg_id": "guided.hypothesis.field_label",
                "input_type": "textarea",
            }
        ],
        "completion_rule": {"type": "all_fields_filled", "field_ids": ["hypothesis"]},
        "report_section": "hypothesis",
    },
    {
        "step_id": "stats",
        "title_msg_id": "guided.stats.title",
        "prompt_msg_id": "guided.stats.prompt",
        "fields": [
            {
                "field_id": "stats_interpretation",
                "label_msg_id": "guided.stats.interpretation_label",
                "input_type": "textarea",
            }
        ],
        "completion_rule": {"type": "all_fields_filled", "field_ids": ["stats_interpretation"]},
        "report_section": "stats",
    },
    {
        "step_id": "conclusion",
        "title_msg_id": "guided.conclusion.title",
        "prompt_msg_id": "guided.conclusion.prompt",
        "fields": [
            {
                "field_id": "conclusion",
                "label_msg_id": "guided.conclusion.field_label",
                "input_type": "textarea",
            }
        ],
        "completion_rule": {"type": "all_fields_filled", "field_ids": ["conclusion"]},
        "report_section": "conclusion",
    },
]

GUIDED_STEP_IDS = tuple(step["step_id"] for step in GUIDED_STEPS)
GUIDED_STEP_BY_ID = {step["step_id"]: step for step in GUIDED_STEPS}
GUIDED_FIELD_IDS = tuple(field["field_id"] for step in GUIDED_STEPS for field in step["fields"])
DEFAULT_GUIDED_STEP_ID = GUIDED_STEP_IDS[0]


def build_default_guided_answers():
    return {field_id: "" for field_id in GUIDED_FIELD_IDS}


def get_step_by_id(step_id):
    return GUIDED_STEP_BY_ID.get(step_id)


def _normalize_step_id(step_id):
    if step_id in GUIDED_STEP_BY_ID:
        return step_id
    return DEFAULT_GUIDED_STEP_ID


def get_next_step_id(step_id):
    normalized_step_id = _normalize_step_id(step_id)
    current_index = GUIDED_STEP_IDS.index(normalized_step_id)
    if current_index >= len(GUIDED_STEP_IDS) - 1:
        return None
    return GUIDED_STEP_IDS[current_index + 1]


def get_previous_step_id(step_id):
    normalized_step_id = _normalize_step_id(step_id)
    current_index = GUIDED_STEP_IDS.index(normalized_step_id)
    if current_index <= 0:
        return None
    return GUIDED_STEP_IDS[current_index - 1]


def get_current_guided_step(session_state):
    if session_state is None:
        return DEFAULT_GUIDED_STEP_ID

    getter = getattr(session_state, "get", None)
    if callable(getter):
        try:
            guided_step = getter("guided_step", DEFAULT_GUIDED_STEP_ID)
        except (KeyError, TypeError):
            guided_step = DEFAULT_GUIDED_STEP_ID
    else:
        try:
            guided_step = session_state["guided_step"]
        except (KeyError, TypeError):
            guided_step = DEFAULT_GUIDED_STEP_ID

    return guided_step if guided_step in GUIDED_STEP_IDS else DEFAULT_GUIDED_STEP_ID


def is_step_complete(step, guided_answers):
    step_dict = get_step_by_id(step) if isinstance(step, str) else step
    if not step_dict:
        return False

    completion_rule = step_dict.get("completion_rule", {})
    if completion_rule.get("type") != "all_fields_filled":
        return False

    guided_answers = guided_answers or {}
    for field_id in completion_rule.get("field_ids", []):
        value = guided_answers.get(field_id, "")
        if value is None or not str(value).strip():
            return False
    return True


def count_completed_steps(guided_answers):
    return sum(1 for step in GUIDED_STEPS if is_step_complete(step, guided_answers))


def update_guided_answer(session_state, field_id, value):
    if session_state is None:
        return None

    if "guided_answers" not in session_state or not isinstance(session_state["guided_answers"], dict):
        session_state["guided_answers"] = build_default_guided_answers()

    session_state["guided_answers"][field_id] = value
    return session_state


def build_default_session_state():
    return {
        "language": get_default_language(),
        "guided_mode_enabled": False,
        "guided_step": DEFAULT_GUIDED_STEP_ID,
        "student_name": "",
        "group_name": "",
        "guided_answers": build_default_guided_answers(),
        "selected_plots": [],
        "report_ready_state": {"last_error": None, "last_export_language": None},
    }


def backfill_session_state_defaults(session_state, defaults=None):
    if session_state is None:
        return None

    defaults = build_default_session_state() if defaults is None else defaults
    for key, value in defaults.items():
        if key not in session_state:
            session_state[key] = deepcopy(value)
            continue

        if isinstance(value, dict) and isinstance(session_state[key], dict):
            for nested_key, nested_value in value.items():
                if nested_key not in session_state[key]:
                    session_state[key][nested_key] = deepcopy(nested_value)

    return session_state
