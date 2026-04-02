from pathlib import Path

from src.i18n import STRINGS


ROOT = Path(__file__).resolve().parents[1]


def test_english_notebook_uses_canonical_raw_vegetable_column_name():
    source = (ROOT / "src" / "explore_data.py").read_text()
    assert "Raw Vegetable Consumption" in source


def test_english_notebook_uses_shared_normalization_helper():
    source = (ROOT / "src" / "explore_data.py").read_text()
    assert "normalize_dataframe" in source


def test_streamlit_app_uses_canonical_raw_vegetable_column_name():
    source = (ROOT / "app" / "streamlit_app.py").read_text()
    assert '"Raw Vegetable Consumption"' in source


def test_streamlit_app_applies_sidebar_filters_across_multiple_tabs():
    source = (ROOT / "app" / "streamlit_app.py").read_text()
    assert source.count("apply_filters(df)") >= 4


def test_streamlit_app_no_longer_mentions_t_test():
    source = (ROOT / "app" / "streamlit_app.py").read_text()
    assert "t-test" not in source


def test_streamlit_app_reports_filtered_rows():
    source = (ROOT / "app" / "streamlit_app.py").read_text()
    assert 'translate("common.filtered_rows"' in source
    assert 'st.metric(' in source


def test_streamlit_app_defines_a_language_control():
    source = (ROOT / "app" / "streamlit_app.py").read_text()
    assert 'key="language"' in source
    assert 'translate("sidebar.language"' in source
    assert 'st.session_state["language"]' in source


def test_streamlit_app_defines_guided_mode_sidebar_controls():
    source = (ROOT / "app" / "streamlit_app.py").read_text()
    assert 'guided_mode_enabled' in source
    assert 'guided_step' in source
    assert 'guided_answers' in source
    assert 'student_name' in source
    assert 'group_name' in source
    assert 'get_previous_step_id' in source
    assert 'get_next_step_id' in source
    assert 'update_guided_answer' in source
    assert 'count_completed_steps' in source
    assert 'is_step_complete' in source
    assert 'st.sidebar.text_area' in source or 'st.text_area' in source
    assert 'st.sidebar.button' in source


def test_streamlit_app_defines_report_selection_controls():
    source = (ROOT / "app" / "streamlit_app.py").read_text()
    assert 'selected_plots' in source
    assert 'MAX_REPORT_PLOTS' in source
    assert 'build_report_plot_entry' in source
    assert 'build_report_plot_id' in source
    assert 'render_report_sidebar' in source
    assert 'translate("report.sidebar.title"' in source
    assert 'translate("report.controls.add"' in source
    assert 'translate("report.controls.update"' in source
    assert 'translate("report.controls.replace_button"' in source
    assert 'translate("report.sidebar.remove"' in source
    assert 'st.sidebar.error(' in source


def test_streamlit_app_loader_does_not_silently_swallow_all_errors():
    source = (ROOT / "app" / "streamlit_app.py").read_text()
    assert "except Exception" not in source
    assert "pd.errors.ParserError" in source
    assert "pd.errors.EmptyDataError" in source


def test_streamlit_app_translates_current_scope_visible_labels():
    source = (ROOT / "app" / "streamlit_app.py").read_text()
    assert "get_display_column_label" in source
    assert "localize_category_labels" in source
    assert "localize_dataframe_columns" in source
    assert "format_column_option" in source
    assert "format_axis_label" in source
    assert "translate(\"stats.table.feature\"" in source
    assert "translate(\"stats.table.log_odds\"" in source
    assert "translate(\"stats.table.odds_ratio\"" in source


def test_i18n_exposes_german_and_english_catalogs():
    assert set(STRINGS) == {"de", "en"}
    assert "app.title" in STRINGS["de"]
    assert "app.title" in STRINGS["en"]
    assert "app.intro" in STRINGS["de"]
    assert "app.intro" in STRINGS["en"]


def test_requirements_include_seaborn():
    requirements = (ROOT / "requirements.txt").read_text()
    assert "seaborn" in requirements.splitlines()


def test_readme_uses_correct_license_filename():
    readme = (ROOT / "README.md").read_text()
    assert "LICENSE.md" not in readme
    assert "`LICENSE`" in readme


def test_worksheet_matches_stats_sections():
    worksheet = (ROOT / "materials" / "Worksheet.md").read_text()
    assert "Chi-squared" in worksheet
    assert "Logistic regression" in worksheet
