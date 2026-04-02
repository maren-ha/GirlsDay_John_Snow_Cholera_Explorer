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
    "Occupation": {
        "Arbeiter": "Laborer",
        "Diener": "Servant",
        "Hausfrau": "Housewife",
        "Händler": "Merchant",
        "Rentner": "Retiree",
        "Schreiber": "Clerk",
        "Schüler": "Student",
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

DISPLAY_COLUMN_LABEL_MAP = {
    "de": {
        "Age": "Alter",
        "Gender": "Geschlecht",
        "Occupation": "Beruf",
        "Household Size": "Haushaltsgröße",
        "Raw Vegetable Consumption": "Rohkost",
        "Nearest Pump": "Nächste Pumpe",
        "Home Location X": "Wohnort X",
        "Home Location Y": "Wohnort Y",
        "Health Status": "Gesundheitszustand",
        "Distance to Pump A": "Entfernung zu Pumpe A",
        "Distance to Pump B": "Entfernung zu Pumpe B",
        "Distance to Pump C": "Entfernung zu Pumpe C",
        "Distance to Pump D": "Entfernung zu Pumpe D",
        "Household Size Category": "Haushaltsgröße (Kategorie)",
    },
    "en": {
        "Age": "Age",
        "Gender": "Gender",
        "Occupation": "Occupation",
        "Household Size": "Household Size",
        "Raw Vegetable Consumption": "Raw Vegetable Consumption",
        "Nearest Pump": "Nearest Pump",
        "Home Location X": "Home Location X",
        "Home Location Y": "Home Location Y",
        "Health Status": "Health Status",
        "Distance to Pump A": "Distance to Pump A",
        "Distance to Pump B": "Distance to Pump B",
        "Distance to Pump C": "Distance to Pump C",
        "Distance to Pump D": "Distance to Pump D",
        "Household Size Category": "Household Size Category",
    },
}

DISPLAY_VALUE_LABEL_MAP = {
    "de": {
        "Gender": {
            "Male": "Männlich",
            "Female": "Weiblich",
        },
        "Occupation": {
            "Laborer": "Arbeiter",
            "Servant": "Diener",
            "Housewife": "Hausfrau",
            "Merchant": "Händler",
            "Retiree": "Rentner",
            "Clerk": "Schreiber",
            "Student": "Schüler",
        },
        "Nearest Pump": {
            "Pump A": "Pumpe A",
            "Pump B": "Pumpe B",
            "Pump C": "Pumpe C",
            "Pump D": "Pumpe D",
        },
        "Raw Vegetable Consumption": {
            "Often": "Oft",
            "Sometimes": "Manchmal",
            "Rarely": "Selten",
        },
        "Health Status": {
            "Death": "Tod",
            "Severe Illness": "Schwere Krankheit",
            "Mild Illness": "Leichte Krankheit",
            "No Illness": "Keine Krankheit",
        },
        "Household Size Category": {
            "Unknown": "Unbekannt",
        },
    },
    "en": {
        "Gender": {
            "Male": "Male",
            "Female": "Female",
        },
        "Occupation": {
            "Laborer": "Laborer",
            "Servant": "Servant",
            "Housewife": "Housewife",
            "Merchant": "Merchant",
            "Retiree": "Retiree",
            "Clerk": "Clerk",
            "Student": "Student",
        },
        "Nearest Pump": {
            "Pump A": "Pump A",
            "Pump B": "Pump B",
            "Pump C": "Pump C",
            "Pump D": "Pump D",
        },
        "Raw Vegetable Consumption": {
            "Often": "Often",
            "Sometimes": "Sometimes",
            "Rarely": "Rarely",
        },
        "Health Status": {
            "Death": "Death",
            "Severe Illness": "Severe Illness",
            "Mild Illness": "Mild Illness",
            "No Illness": "No Illness",
        },
        "Household Size Category": {
            "Unknown": "Unknown",
        },
    },
}


def canonicalize_columns(columns):
    return [CANONICAL_COLUMN_MAP.get(column, column) for column in columns]


def canonicalize_value(column, value):
    return VALUE_NORMALIZATION_MAP.get(column, {}).get(value, value)


def get_display_label(category, value, language):
    if value is None:
        return value
    language_map = DISPLAY_VALUE_LABEL_MAP.get(language, {})
    category_map = language_map.get(category)
    if isinstance(category_map, dict):
        return category_map.get(value, value)
    german_map = DISPLAY_VALUE_LABEL_MAP.get("de", {}).get(category, {})
    if isinstance(german_map, dict):
        return german_map.get(value, value)
    return value


def get_display_column_label(column, language):
    language_map = DISPLAY_COLUMN_LABEL_MAP.get(language, {})
    if column in language_map:
        return language_map[column]
    german_map = DISPLAY_COLUMN_LABEL_MAP.get("de", {})
    if column in german_map:
        return german_map[column]
    english_map = DISPLAY_COLUMN_LABEL_MAP.get("en", {})
    return english_map.get(column, column)


def localize_column_labels(columns, language):
    return [get_display_column_label(column, language) for column in columns]


def localize_category_labels(category, values, language):
    return [get_display_label(category, value, language) for value in values]


def localize_dataframe_columns(df, language):
    localized = df.copy()
    localized.columns = localize_column_labels(localized.columns, language)
    return localized


def normalize_dataframe(df):
    normalized = df.copy()
    normalized.columns = canonicalize_columns(normalized.columns)
    for column, replacements in VALUE_NORMALIZATION_MAP.items():
        if column in normalized.columns:
            normalized[column] = normalized[column].replace(replacements)
    return normalized
