from src.guided_mode import GUIDED_STEPS
from src.i18n import STRINGS, get_default_language, translate


def test_translate_returns_german_app_title():
    assert translate("app.title", "de") == "Epidemiologischer Daten-Explorer — London 1854"


def test_unknown_language_falls_back_to_german_without_claiming_complete_translation_coverage():
    assert translate("app.subtitle", "fr") == translate("app.subtitle", "de")
    assert translate("app.subtitle", "fr") == "Du bist Wissenschaftlerin in London, 1854."


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
    assert "app.hero.eyebrow" not in STRINGS["de"]
    assert "app.hero.eyebrow" not in STRINGS["en"]
    assert STRINGS["de"]["app.hero.chip.report"] == "PDF-Bericht"
    assert STRINGS["en"]["app.subtitle"] == "You are a scientist in London, 1854."
    assert STRINGS["en"]["overview.missing_chart.title"] == "Missing values by column"


def test_german_plot_copy_uses_student_friendly_language():
    german_copy = "\n".join(
        STRINGS["de"][key]
        for key in (
            "overview.missing_chart.title",
            "overview.missing_chart.caption",
            "overview.missing_count_label",
            "tab.distributions",
            "distributions.subtitle",
            "distributions.caption",
            "distributions.title",
            "distributions.discuss",
            "tab.scatter",
            "scatter.subtitle",
            "scatter.caption",
            "scatter.title",
            "scatter.discuss",
        )
    )

    assert STRINGS["de"]["tab.distributions"] == "Balkendiagramm"
    assert STRINGS["de"]["tab.scatter"] == "Punktdiagramm"
    assert "Jitter" not in german_copy
    assert "möglicher Risikofaktoren" not in german_copy
    assert "Fein aufgelöster Blick" not in german_copy
    assert "aktuellen Ausschnitt" not in german_copy


def test_intro_copy_uses_standalone_scientist_story_without_multi_task_framing():
    for language in ("de", "en"):
        intro = STRINGS[language]["app.intro"]
        focus = STRINGS[language]["app.focus"]
        assert "scientist" in intro.lower() or "wissenschaft" in intro.lower()
        assert "symptoms" in intro.lower() or "symptome" in intro.lower()
        assert "bad air" in intro.lower() or "schlechte luft" in intro.lower()
        assert "fear" in intro.lower() or "angst" in intro.lower()
        assert "four tasks" not in intro.lower()
        assert "vier aufgaben" not in intro.lower()
        assert "this app focuses" not in focus.lower()
        assert "diese app konzentriert" not in focus.lower()

    german_story = f"{STRINGS['de']['app.subtitle']}\n{STRINGS['de']['app.intro']}"
    assert "Wissenschaftlerin" in german_story
    assert "Wissenschaftler." not in german_story
    assert "Wissenschaftler und" not in german_story

    english_intro = STRINGS["en"]["app.intro"]
    assert "how many people live in each household" in english_intro
    assert "diet" in english_intro
    assert "households, distances" not in english_intro


def test_histogram_control_copy_describes_bin_count_not_bin_width():
    assert "sidebar.bin_width" not in STRINGS["de"]
    assert "sidebar.bin_width" not in STRINGS["en"]
    assert STRINGS["de"]["sidebar.bin_count"] == "Anzahl der Gruppen (für Zahlen)"
    assert STRINGS["en"]["sidebar.bin_count"] == "Number of groups (numeric variables)"

    joined_copy = "\n".join(
        value for language in STRINGS.values() for value in language.values() if isinstance(value, str)
    ).lower()
    assert "bin-breite" not in joined_copy
    assert "bin width" not in joined_copy


def test_student_facing_copy_uses_simple_statistical_test_language():
    joined_copy = "\n".join(
        value for language in STRINGS.values() for value in language.values() if isinstance(value, str)
    ).lower()

    assert "correlation" not in joined_copy
    assert "causation" not in joined_copy
    assert "korrelation" not in joined_copy
    assert "kausal" not in joined_copy
    assert "statistischen test" in joined_copy
    assert "statistical test" in joined_copy
