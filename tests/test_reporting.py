import pytest

from src.guided_mode import build_default_guided_answers
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
    assert payload["selected_plots"][0]["mime_type"] == "image/png"
    assert payload["report_sections"][0]["step_id"] == "overview"
    assert payload["report_sections"][0]["answer"] == "The data looks clustered around the pumps."
    assert payload["report_sections"][0]["report_section"] == "overview"
    assert len(payload["report_sections"]) >= 1
    assert set(SUPPORTED_REPORT_PLOT_TYPES) == {"distribution", "heatmap", "scatter"}
