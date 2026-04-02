from src.data_schema import DATA_DIR, canonicalize_columns, canonicalize_value


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
