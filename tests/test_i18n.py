from src.guided_mode import GUIDED_STEPS
from src.i18n import STRINGS, get_default_language, translate


def test_translate_returns_german_app_title():
    assert translate("app.title", "de") == "Epidemiologischer Daten-Explorer — London 1854"


def test_unknown_language_falls_back_to_german_without_claiming_complete_translation_coverage():
    assert translate("app.subtitle", "fr") == translate("app.subtitle", "de")
    assert translate("app.subtitle", "fr") == "Eine kurze Einführung in den Ausbruch von 1854."


def test_guided_strings_cover_the_sidebar_and_step_copy_in_both_languages():
    sidebar_keys = {
        "guided.sidebar.title",
        "guided.sidebar.enabled",
        "guided.sidebar.disabled",
        "guided.sidebar.student_name",
        "guided.sidebar.group_name",
        "guided.sidebar.current_step",
        "guided.sidebar.progress",
        "guided.sidebar.completed_steps",
        "guided.sidebar.previous",
        "guided.sidebar.next",
        "guided.sidebar.complete",
        "guided.sidebar.incomplete",
        "guided.sidebar.empty_state",
    }

    step_keys = set()
    for step in GUIDED_STEPS:
        step_keys.add(step["title_msg_id"])
        step_keys.add(step["prompt_msg_id"])
        for field in step["fields"]:
            step_keys.add(field["label_msg_id"])

    for language in ("de", "en"):
        for key in sidebar_keys | step_keys:
            assert key in STRINGS[language]


def test_default_language_is_german():
    assert get_default_language() == "de"


def test_report_export_copy_uses_actual_german_labels():
    assert STRINGS["de"]["report.pdf.student_label"] == "Schülername:"
    assert STRINGS["de"]["report.pdf.selected_plots"] == "Ausgewählte Grafiken"
    assert STRINGS["de"]["report.pdf.guided_responses"] == "Geführte Antworten"
    assert STRINGS["de"]["report.pdf.no_plots_selected"] == "Noch keine Grafiken ausgewählt."


def test_visual_refresh_copy_is_translated_in_both_languages():
    assert STRINGS["de"]["app.hero.eyebrow"] == "Klassenraum-Explorer"
    assert STRINGS["de"]["app.hero.chip.report"] == "PDF-Bericht"
    assert STRINGS["en"]["app.subtitle"] == "A short introduction to the 1854 outbreak."
    assert STRINGS["en"]["overview.missing_chart.title"] == "Missing values in the current slice"
