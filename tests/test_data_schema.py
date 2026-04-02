from src.data_schema import (
    DATA_DIR,
    canonicalize_columns,
    canonicalize_value,
    get_display_column_label,
    get_display_label,
    normalize_dataframe,
)


def test_canonicalize_columns_maps_english_raw_vegetable_column():
    columns = ["Age", "Raw Vegetable Consumption", "Health Status"]
    renamed = canonicalize_columns(columns)
    assert "Raw Vegetable Consumption" in renamed


def test_canonicalize_columns_maps_german_columns_to_canonical_names():
    columns = ["Alter", "Geschlecht", "Rohkost-Konsum", "Gesundheitsstatus"]
    renamed = canonicalize_columns(columns)
    assert renamed == ["Age", "Gender", "Raw Vegetable Consumption", "Health Status"]


def test_data_dir_points_to_tracked_repo_data_folder():
    assert DATA_DIR.name == "data"
    assert DATA_DIR.exists()


def test_german_dataset_path_is_tracked_in_data_dir():
    assert (DATA_DIR / "cholera_datensatz_de.csv").exists()


def test_canonicalize_value_maps_german_gender():
    assert canonicalize_value("Gender", "Weiblich") == "Female"


def test_canonicalize_value_maps_german_pump_name():
    assert canonicalize_value("Nearest Pump", "Pumpe B") == "Pump B"


def test_canonicalize_value_maps_german_health_status():
    assert canonicalize_value("Health Status", "Schwere Krankheit") == "Severe Illness"


def test_normalize_dataframe_maps_german_occupation_values_for_english_display():
    class FakeSeries(list):
        def replace(self, replacements):
            return FakeSeries([replacements.get(value, value) for value in self])

    class FakeFrame:
        def __init__(self, data):
            self._data = {key: list(values) for key, values in data.items()}
            self._columns = list(data.keys())

        @property
        def columns(self):
            return self._columns

        @columns.setter
        def columns(self, new_columns):
            new_columns = list(new_columns)
            self._data = {
                new_columns[index]: self._data[old_column]
                for index, old_column in enumerate(self._columns)
            }
            self._columns = new_columns

        def copy(self):
            return FakeFrame(self._data)

        def __getitem__(self, key):
            return FakeSeries(self._data[key])

        def __setitem__(self, key, value):
            self._data[key] = list(value)

    df = FakeFrame({"Beruf": ["Arbeiter", "Schreiber"], "Alter": [12, 34]})
    normalized = normalize_dataframe(df)

    assert list(normalized.columns) == ["Occupation", "Age"]
    assert list(normalized["Occupation"]) == ["Laborer", "Clerk"]
    assert get_display_label("Occupation", normalized["Occupation"][0], "en") == "Laborer"
    assert get_display_label("Occupation", normalized["Occupation"][1], "en") == "Clerk"


def test_display_label_uses_explicit_english_mappings_for_student_facing_categories():
    assert get_display_label("Occupation", "Laborer", "en") == "Laborer"
    assert get_display_label("Nearest Pump", "Pump B", "en") == "Pump B"
    assert get_display_label("Health Status", "Severe Illness", "en") == "Severe Illness"


def test_display_label_falls_back_to_german_when_language_label_is_missing():
    assert get_display_label("Occupation", "Laborer", "fr") == "Arbeiter"
    assert get_display_label("Nearest Pump", "Pump A", "fr") == "Pumpe A"


def test_display_column_label_falls_back_to_german_when_language_is_missing():
    assert get_display_column_label("Age", "fr") == "Alter"
    assert get_display_column_label("Nearest Pump", "fr") == "Nächste Pumpe"
