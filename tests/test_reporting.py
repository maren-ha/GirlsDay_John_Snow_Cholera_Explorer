import base64

import pytest

from src.guided_mode import build_default_guided_answers
from src.i18n import translate
from src.report_pdf import render_report_pdf, render_report_pdf_safe
from src.reporting import (
    MAX_REPORT_PLOTS,
    SUPPORTED_REPORT_PLOT_TYPES,
    build_report_payload,
    build_report_plot_entry,
    figure_to_png_bytes,
    remove_selected_plot,
    replace_selected_plot,
    upsert_selected_plot,
)


class FakeFigure:
    def __init__(self):
        self.savefig_calls = []

    def savefig(self, buffer, **kwargs):
        self.savefig_calls.append(kwargs)
        buffer.write(b"\x89PNG\r\n\x1a\nfake-image")


_VALID_PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO2s7wAAAABJRU5ErkJggg=="
)


def _build_plot_entry(plot_id, plot_type="distribution"):
    fig = FakeFigure()
    return build_report_plot_entry(
        plot_id=plot_id,
        plot_type=plot_type,
        title=f"Title {plot_id}",
        caption=f"Caption {plot_id}",
        parameters={"plot_id": plot_id, "value": 1},
        fig=fig,
    )


def _build_pdf_plot_entry(plot_id, plot_type="distribution"):
    return {
        "plot_id": plot_id,
        "plot_type": plot_type,
        "title": f"Title {plot_id}",
        "caption": f"Caption {plot_id}",
        "parameters": {"plot_id": plot_id, "value": 1},
        "image_bytes": _VALID_PNG_BYTES,
        "mime_type": "image/png",
    }


def _build_non_ascii_pdf_plot_entry(plot_id):
    entry = _build_pdf_plot_entry(plot_id)
    entry["title"] = "Ausgewählte Grafik"
    entry["caption"] = "Hinweis für Schüler:innen in Köln"
    return entry


def test_figure_to_png_bytes_returns_png_bytes():
    fig = FakeFigure()
    png_bytes = figure_to_png_bytes(fig)

    assert isinstance(png_bytes, bytes)
    assert png_bytes.startswith(b"\x89PNG\r\n\x1a\n")
    assert fig.savefig_calls == [{"format": "png", "bbox_inches": "tight"}]


def test_build_report_plot_entry_has_the_expected_contract_shape():
    entry = _build_plot_entry("distribution:age")

    assert set(entry) == {
        "plot_id",
        "plot_type",
        "title",
        "caption",
        "parameters",
        "image_bytes",
        "mime_type",
    }
    assert entry["plot_id"] == "distribution:age"
    assert entry["plot_type"] == "distribution"
    assert entry["title"] == "Title distribution:age"
    assert entry["caption"] == "Caption distribution:age"
    assert entry["parameters"] == {"plot_id": "distribution:age", "value": 1}
    assert isinstance(entry["image_bytes"], bytes)
    assert entry["mime_type"] == "image/png"


def test_build_report_plot_entry_rejects_unsupported_plot_types():
    fig = FakeFigure()
    with pytest.raises(ValueError, match="Unsupported report plot type"):
        build_report_plot_entry(
            plot_id="boxplot:age",
            plot_type="boxplot",
            title="Title",
            caption="Caption",
            parameters={},
            fig=fig,
        )


def test_upsert_selected_plot_enforces_the_selection_limit_and_replaces_matching_ids():
    selected_plots = []
    for index in range(MAX_REPORT_PLOTS):
        selected_plots = upsert_selected_plot(selected_plots, _build_plot_entry(f"plot-{index}"))

    assert len(selected_plots) == MAX_REPORT_PLOTS

    with pytest.raises(ValueError, match="maximum of 4"):
        upsert_selected_plot(selected_plots, _build_plot_entry("plot-overflow"))

    replacement = _build_plot_entry("plot-2")
    replacement["title"] = "Updated title"
    updated = upsert_selected_plot(selected_plots, replacement)

    assert len(updated) == MAX_REPORT_PLOTS
    assert [plot["plot_id"] for plot in updated].count("plot-2") == 1
    assert next(plot for plot in updated if plot["plot_id"] == "plot-2")["title"] == "Updated title"


def test_remove_selected_plot_drops_the_requested_item():
    selected_plots = [_build_plot_entry("plot-1"), _build_plot_entry("plot-2")]

    updated = remove_selected_plot(selected_plots, "plot-1")

    assert [plot["plot_id"] for plot in updated] == ["plot-2"]


def test_replace_selected_plot_swaps_the_requested_item_without_changing_the_total_count():
    selected_plots = [_build_plot_entry("plot-1"), _build_plot_entry("plot-2")]
    replacement = _build_plot_entry("plot-3")
    replacement["title"] = "Replacement title"

    updated = replace_selected_plot(selected_plots, "plot-1", replacement)

    assert len(updated) == 2
    assert [plot["plot_id"] for plot in updated] == ["plot-3", "plot-2"]
    assert updated[0]["title"] == "Replacement title"


def test_replace_selected_plot_rejects_duplicate_plot_ids():
    selected_plots = [_build_plot_entry("plot-1"), _build_plot_entry("plot-2")]

    with pytest.raises(ValueError, match="already saved"):
        replace_selected_plot(selected_plots, "plot-1", _build_plot_entry("plot-2"))


def test_build_report_payload_collects_guided_answers_and_selected_plots():
    selected_plot = _build_plot_entry("distribution:age")
    guided_answers = build_default_guided_answers()
    guided_answers["overview_observation"] = "The data looks clustered around the pumps."

    payload = build_report_payload(
        language="en",
        guided_answers=guided_answers,
        selected_plots=[selected_plot],
        student_name="Ava",
        group_name="Team River",
    )

    assert payload["language"] == "en"
    assert payload["student_name"] == "Ava"
    assert payload["group_name"] == "Team River"
    assert payload["guided_answers"] == guided_answers
    assert payload["selected_plot_count"] == 1
    assert payload["selected_plots"][0]["plot_id"] == "distribution:age"
    assert payload["selected_plots"][0]["caption"] == "Caption distribution:age"
    assert payload["selected_plots"][0]["mime_type"] == "image/png"
    assert payload["report_sections"][0]["step_id"] == "overview"
    assert payload["report_sections"][0]["answer"] == "The data looks clustered around the pumps."
    assert payload["report_sections"][0]["title_text"] == translate("guided.overview.title", "en")
    assert payload["report_sections"][0]["prompt_text"] == translate("guided.overview.prompt", "en")
    assert payload["report_sections"][0]["fields"][0]["label_text"] == translate(
        "guided.overview.observation_label", "en"
    )
    assert payload["report_sections"][0]["field_labels"] == {
        "overview_observation": translate("guided.overview.observation_label", "en")
    }
    assert payload["report_sections"][0]["report_section"] == "overview"
    assert len(payload["report_sections"]) >= 1
    assert set(SUPPORTED_REPORT_PLOT_TYPES) == {"distribution", "heatmap", "scatter"}


def test_build_report_payload_allows_zero_selected_plots_and_keeps_translated_field_labels():
    guided_answers = build_default_guided_answers()
    guided_answers["overview_observation"] = "Die Quelle liegt nahe bei der Pumpe."

    payload = build_report_payload(
        language="de",
        guided_answers=guided_answers,
        selected_plots=[],
        student_name="Jörg Müller",
        group_name="Gruppe 7",
    )

    assert payload["selected_plot_count"] == 0
    assert payload["selected_plots"] == []
    assert payload["report_sections"][0]["title_text"] == translate("guided.overview.title", "de")
    assert payload["report_sections"][0]["prompt_text"] == translate("guided.overview.prompt", "de")
    assert payload["report_sections"][0]["fields"][0]["label_text"] == translate(
        "guided.overview.observation_label", "de"
    )
    assert payload["report_sections"][0]["field_labels"] == {
        "overview_observation": translate("guided.overview.observation_label", "de")
    }


def test_build_report_payload_preserves_selected_plot_caption_metadata():
    selected_plot = _build_non_ascii_pdf_plot_entry("distribution:age")

    payload = build_report_payload(
        language="de",
        guided_answers=build_default_guided_answers(),
        selected_plots=[selected_plot],
        student_name="Jörg Müller",
        group_name="Gruppe 7",
    )

    assert payload["selected_plot_count"] == 1
    assert payload["selected_plots"][0]["caption"] == "Hinweis für Schüler:innen in Köln"
    assert payload["selected_plots"][0]["title"] == "Ausgewählte Grafik"


def test_build_report_payload_relocalizes_saved_plot_copy_for_the_current_language():
    selected_plot = _build_non_ascii_pdf_plot_entry("distribution:age")

    payload = build_report_payload(
        language="en",
        guided_answers=build_default_guided_answers(),
        selected_plots=[selected_plot],
        student_name="Jörg Müller",
        group_name="Gruppe 7",
    )

    assert payload["selected_plots"][0]["title"] == "Ausgewählte Grafik"
    assert payload["selected_plots"][0]["caption"] == "Hinweis für Schüler:innen in Köln"
    assert payload["selected_plots"][0]["title_text"] == translate("distributions.title", "en")
    assert payload["selected_plots"][0]["caption_text"] == translate("distributions.caption", "en")
    assert payload["selected_plots"][0]["plot_type_label"] == translate("report.plot_type.distribution", "en")


def test_render_report_pdf_returns_pdf_bytes():
    payload = build_report_payload(
        language="en",
        guided_answers=build_default_guided_answers(),
        selected_plots=[],
        student_name="Ava",
        group_name="Team River",
    )

    pdf_bytes = render_report_pdf(payload)

    assert isinstance(pdf_bytes, bytes)
    assert pdf_bytes.startswith(b"%PDF")


def test_render_report_pdf_uses_the_current_language_heading():
    payload = build_report_payload(
        language="de",
        guided_answers=build_default_guided_answers(),
        selected_plots=[],
        student_name="Ava",
        group_name="Team River",
    )

    pdf_bytes = render_report_pdf(payload)

    assert translate("report.pdf.heading", "de").encode("ascii") in pdf_bytes


def test_render_report_pdf_uses_language_specific_pdf_labels_and_empty_state():
    payload = build_report_payload(
        language="de",
        guided_answers=build_default_guided_answers(),
        selected_plots=[],
        student_name="Ava",
        group_name="Team River",
    )

    pdf_bytes = render_report_pdf(payload)

    assert b"Sch\\374lername:" in pdf_bytes
    assert b"Ausgew\\344hlte Grafiken" in pdf_bytes
    assert b"Gef\\374hrte Antworten" in pdf_bytes
    assert b"Noch keine Grafiken ausgew\\344hlt." in pdf_bytes


def test_render_report_pdf_uses_current_language_plot_type_labels():
    payload = build_report_payload(
        language="de",
        guided_answers=build_default_guided_answers(),
        selected_plots=[_build_pdf_plot_entry("distribution:age")],
        student_name="Ava",
        group_name="Team River",
    )

    pdf_bytes = render_report_pdf(payload)

    assert b"Verteilung" in pdf_bytes
    assert b"(distribution)" not in pdf_bytes


def test_render_report_pdf_uses_translated_field_labels_instead_of_raw_field_ids():
    guided_answers = build_default_guided_answers()
    guided_answers["overview_observation"] = "The data looks clustered around the pumps."
    payload = build_report_payload(
        language="en",
        guided_answers=guided_answers,
        selected_plots=[],
        student_name="Ava",
        group_name="Team River",
    )

    pdf_bytes = render_report_pdf(payload)

    assert b"Your first observation" in pdf_bytes
    assert b"overview_observation" not in pdf_bytes


def test_render_report_pdf_preserves_non_ascii_german_user_text_where_practical():
    guided_answers = build_default_guided_answers()
    guided_answers["overview_observation"] = "Schüler:innen sehen Köln nahe der Pumpe."
    payload = build_report_payload(
        language="de",
        guided_answers=guided_answers,
        selected_plots=[_build_non_ascii_pdf_plot_entry("distribution:age")],
        student_name="Jörg Müller",
        group_name="Gruppe 7",
    )

    pdf_bytes = render_report_pdf(payload)

    assert b"Sch\\374lername: J\\366rg M\\374ller" in pdf_bytes
    assert b"Sch\\374ler:innen sehen K\\366ln nahe der Pumpe." in pdf_bytes


def test_render_report_pdf_includes_student_and_group_lines_when_both_are_present():
    payload = build_report_payload(
        language="en",
        guided_answers=build_default_guided_answers(),
        selected_plots=[_build_pdf_plot_entry("distribution:age")],
        student_name="Ava",
        group_name="Team River",
    )

    pdf_bytes = render_report_pdf(payload)

    assert b"Student: Ava" in pdf_bytes
    assert b"Group: Team River" in pdf_bytes


def test_render_report_pdf_omits_blank_student_or_group_lines():
    payload = build_report_payload(
        language="en",
        guided_answers=build_default_guided_answers(),
        selected_plots=[],
        student_name="Ava",
        group_name="",
    )

    pdf_bytes = render_report_pdf(payload)

    assert b"Student: Ava" in pdf_bytes
    assert b"Group:" not in pdf_bytes


def test_render_report_pdf_embeds_selected_plot_images():
    payload = build_report_payload(
        language="en",
        guided_answers=build_default_guided_answers(),
        selected_plots=[_build_pdf_plot_entry("distribution:age")],
        student_name="Ava",
        group_name="Team River",
    )

    pdf_bytes = render_report_pdf(payload)

    assert b"/Subtype /Image" in pdf_bytes
    assert translate("distributions.title", "en").encode("ascii") in pdf_bytes


def test_render_report_pdf_safe_surfaces_render_failures_without_mutating_payload(monkeypatch):
    payload = build_report_payload(
        language="de",
        guided_answers=build_default_guided_answers(),
        selected_plots=[_build_pdf_plot_entry("distribution:age")],
        student_name="Ava",
        group_name="Team River",
    )
    original_payload = {
        "selected_plots": [dict(plot) for plot in payload["selected_plots"]],
        "report_sections": [dict(section) for section in payload["report_sections"]],
    }

    monkeypatch.setattr(
        "src.report_pdf.render_report_pdf",
        lambda value: (_ for _ in ()).throw(ValueError("bad image bytes")),
    )

    pdf_bytes, error_message = render_report_pdf_safe(payload)

    assert pdf_bytes is None
    assert error_message == "bad image bytes"
    assert payload["selected_plots"] == original_payload["selected_plots"]
    assert payload["report_sections"][0]["field_labels"] == original_payload["report_sections"][0]["field_labels"]


def test_build_report_payload_raises_for_malformed_selected_plots_before_pdf_rendering():
    with pytest.raises(ValueError, match="image_bytes must be bytes"):
        build_report_payload(
            language="de",
            guided_answers=build_default_guided_answers(),
            selected_plots=[
                {
                    "plot_id": "distribution:age",
                    "plot_type": "distribution",
                    "title": "Title",
                    "caption": "Caption",
                    "parameters": {},
                    "image_bytes": "not-bytes",
                    "mime_type": "image/png",
                }
            ],
            student_name="Ava",
            group_name="Team River",
        )
