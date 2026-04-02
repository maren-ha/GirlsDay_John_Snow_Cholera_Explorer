from pathlib import Path


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
    assert 'st.metric("Filtered rows"' in source


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
