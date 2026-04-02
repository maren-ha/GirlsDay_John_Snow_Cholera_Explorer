from copy import deepcopy
from hashlib import sha1
from io import BytesIO
import json

from src.guided_mode import GUIDED_STEPS
from src.i18n import translate


MAX_REPORT_PLOTS = 4
SUPPORTED_REPORT_PLOT_TYPES = ("distribution", "heatmap", "scatter")
REQUIRED_SELECTED_PLOT_KEYS = (
    "plot_id",
    "plot_type",
    "title",
    "caption",
    "parameters",
    "image_bytes",
    "mime_type",
)
REPORT_PLOT_COPY_KEYS = {
    "distribution": {
        "title_msg_id": "distributions.title",
        "caption_msg_id": "distributions.caption",
    },
    "heatmap": {
        "title_msg_id": "heatmap.subtitle",
        "caption_msg_id": "heatmap.caption",
    },
    "scatter": {
        "title_msg_id": "scatter.subtitle",
        "caption_msg_id": "scatter.caption",
    },
}


def _ensure_supported_plot_type(plot_type):
    if plot_type not in SUPPORTED_REPORT_PLOT_TYPES:
        raise ValueError(f"Unsupported report plot type: {plot_type}")
    return plot_type


def build_report_plot_id(plot_type, parameters):
    plot_type = _ensure_supported_plot_type(plot_type)
    normalized_parameters = deepcopy(parameters) if isinstance(parameters, dict) else {}
    serialized_parameters = json.dumps(
        normalized_parameters,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )
    digest = sha1(serialized_parameters.encode("utf-8")).hexdigest()[:12]
    return f"{plot_type}:{digest}"


def figure_to_png_bytes(fig):
    buffer = BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight")
    return buffer.getvalue()


def build_report_plot_entry(plot_id, plot_type, title, caption, parameters, fig, mime_type="image/png"):
    plot_type = _ensure_supported_plot_type(plot_type)
    normalized_parameters = deepcopy(parameters) if isinstance(parameters, dict) else {}
    return {
        "plot_id": str(plot_id),
        "plot_type": plot_type,
        "title": title,
        "caption": caption,
        "parameters": normalized_parameters,
        "image_bytes": figure_to_png_bytes(fig),
        "mime_type": mime_type,
    }


def _normalize_selected_plot(plot):
    if not isinstance(plot, dict):
        raise ValueError("Selected plot must be a mapping.")

    missing_keys = [key for key in REQUIRED_SELECTED_PLOT_KEYS if key not in plot]
    if missing_keys:
        raise ValueError(f"Selected plot is missing required keys: {', '.join(missing_keys)}")

    plot_type = _ensure_supported_plot_type(plot["plot_type"])
    image_bytes = plot["image_bytes"]
    if not isinstance(image_bytes, (bytes, bytearray)):
        raise ValueError("Selected plot image_bytes must be bytes.")

    mime_type = plot["mime_type"]
    if mime_type != "image/png":
        raise ValueError("Selected plot mime_type must be image/png.")

    parameters = plot["parameters"]
    if not isinstance(parameters, dict):
        raise ValueError("Selected plot parameters must be a mapping.")

    return {
        "plot_id": str(plot["plot_id"]),
        "plot_type": plot_type,
        "title": plot["title"],
        "caption": plot["caption"],
        "parameters": deepcopy(parameters),
        "image_bytes": bytes(image_bytes),
        "mime_type": mime_type,
    }


def localize_selected_plot(plot, language):
    normalized_plot = _normalize_selected_plot(plot)
    copy_keys = REPORT_PLOT_COPY_KEYS.get(normalized_plot["plot_type"], {})
    title_msg_id = copy_keys.get("title_msg_id")
    caption_msg_id = copy_keys.get("caption_msg_id")
    plot_type_label = translate(f"report.plot_type.{normalized_plot['plot_type']}", language)

    localized_plot = deepcopy(normalized_plot)
    localized_plot["title_text"] = (
        translate(title_msg_id, language) if title_msg_id else normalized_plot["title"]
    )
    localized_plot["caption_text"] = (
        translate(caption_msg_id, language) if caption_msg_id else normalized_plot["caption"]
    )
    localized_plot["plot_type_label"] = plot_type_label
    return localized_plot


def normalize_selected_plots(selected_plots):
    normalized_plots = []
    for plot in selected_plots or []:
        normalized_plots = upsert_selected_plot(normalized_plots, plot)
    return normalized_plots


def upsert_selected_plot(selected_plots, selected_plot):
    normalized_plot = _normalize_selected_plot(selected_plot)
    updated_plots = []
    replaced = False

    for plot in selected_plots or []:
        normalized_existing = _normalize_selected_plot(plot)
        if normalized_existing["plot_id"] == normalized_plot["plot_id"]:
            updated_plots.append(normalized_plot)
            replaced = True
        else:
            updated_plots.append(normalized_existing)

    if not replaced:
        if len(updated_plots) >= MAX_REPORT_PLOTS:
            raise ValueError(f"Cannot select more than the maximum of {MAX_REPORT_PLOTS} report plots.")
        updated_plots.append(normalized_plot)

    return updated_plots


def remove_selected_plot(selected_plots, plot_id):
    return [
        _normalize_selected_plot(plot)
        for plot in (selected_plots or [])
        if plot.get("plot_id") != plot_id
    ]


def replace_selected_plot(selected_plots, plot_id, replacement_plot):
    normalized_replacement = _normalize_selected_plot(replacement_plot)
    updated_plots = []
    replaced = False

    for plot in selected_plots or []:
        normalized_existing = _normalize_selected_plot(plot)
        if normalized_existing["plot_id"] == plot_id:
            updated_plots.append(normalized_replacement)
            replaced = True
        else:
            updated_plots.append(normalized_existing)

    if not replaced:
        raise ValueError(f"Selected plot {plot_id!r} was not found.")

    if len(updated_plots) > MAX_REPORT_PLOTS:
        raise ValueError(f"Cannot select more than the maximum of {MAX_REPORT_PLOTS} report plots.")

    replacement_id = normalized_replacement["plot_id"]
    if sum(1 for plot in updated_plots if plot["plot_id"] == replacement_id) > 1:
        raise ValueError(f"Selected plot {replacement_id!r} is already saved.")

    return updated_plots


def build_report_payload(language, guided_answers, selected_plots, student_name, group_name):
    normalized_guided_answers = deepcopy(guided_answers) if isinstance(guided_answers, dict) else {}
    normalized_selected_plots = normalize_selected_plots(selected_plots)
    localized_selected_plots = [localize_selected_plot(plot, language) for plot in normalized_selected_plots]
    report_sections = []

    for step in GUIDED_STEPS:
        field_ids = [field["field_id"] for field in step["fields"]]
        answers = {field_id: normalized_guided_answers.get(field_id, "") for field_id in field_ids}
        fields = []
        field_labels = {}

        for field in step["fields"]:
            field_answer = answers.get(field["field_id"], "")
            field_label = translate(field["label_msg_id"], language)
            field_labels[field["field_id"]] = field_label
            fields.append(
                {
                    "field_id": field["field_id"],
                    "label_msg_id": field["label_msg_id"],
                    "label_text": field_label,
                    "answer": field_answer,
                }
            )

        report_sections.append(
            {
                "step_id": step["step_id"],
                "title_msg_id": step["title_msg_id"],
                "title_text": translate(step["title_msg_id"], language),
                "prompt_msg_id": step["prompt_msg_id"],
                "prompt_text": translate(step["prompt_msg_id"], language),
                "report_section": step["report_section"],
                "field_ids": field_ids,
                "field_labels": field_labels,
                "fields": fields,
                "answers": answers,
                "answer": answers[field_ids[0]] if len(field_ids) == 1 else answers,
            }
        )

    return {
        "language": language,
        "student_name": student_name or "",
        "group_name": group_name or "",
        "guided_answers": normalized_guided_answers,
        "report_sections": report_sections,
        "selected_plots": localized_selected_plots,
        "selected_plot_count": len(localized_selected_plots),
        "selected_plot_ids": [plot["plot_id"] for plot in localized_selected_plots],
    }
