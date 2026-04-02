from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"


CANONICAL_COLUMN_MAP = {
    "Alter": "Age",
    "Geschlecht": "Gender",
    "Beruf": "Occupation",
    "Haushaltsgröße": "Household Size",
    "Rohkost-Konsum": "Raw Vegetable Consumption",
    "Nächstgelegene Pumpe": "Nearest Pump",
    "Wohnort X": "Home Location X",
    "Wohnort Y": "Home Location Y",
    "Gesundheitsstatus": "Health Status",
    "Entfernung zu Pumpe A": "Distance to Pump A",
    "Entfernung zu Pumpe B": "Distance to Pump B",
    "Entfernung zu Pumpe C": "Distance to Pump C",
    "Entfernung zu Pumpe D": "Distance to Pump D",
}

VALUE_NORMALIZATION_MAP = {
    "Gender": {
        "Männlich": "Male",
        "Weiblich": "Female",
    },
    "Nearest Pump": {
        "Pumpe A": "Pump A",
        "Pumpe B": "Pump B",
        "Pumpe C": "Pump C",
        "Pumpe D": "Pump D",
    },
    "Raw Vegetable Consumption": {
        "Oft": "Often",
        "Manchmal": "Sometimes",
        "Selten": "Rarely",
    },
    "Health Status": {
        "Tod": "Death",
        "Schwere Krankheit": "Severe Illness",
        "Leichte Krankheit": "Mild Illness",
        "Keine Krankheit": "No Illness",
    },
}


def canonicalize_columns(columns):
    return [CANONICAL_COLUMN_MAP.get(column, column) for column in columns]


def canonicalize_value(column, value):
    return VALUE_NORMALIZATION_MAP.get(column, {}).get(value, value)


def normalize_dataframe(df):
    normalized = df.copy()
    normalized.columns = canonicalize_columns(normalized.columns)
    for column, replacements in VALUE_NORMALIZATION_MAP.items():
        if column in normalized.columns:
            normalized[column] = normalized[column].replace(replacements)
    return normalized
