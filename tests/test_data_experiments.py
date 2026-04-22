import pandas as pd

from src.data_experiments import apply_random_missingness, prepare_jittered_scatter_values


def test_apply_random_missingness_is_deterministic_and_preserves_original_frame():
    df = pd.DataFrame(
        {
            "Age": [10, 20, 30, 40],
            "Gender": ["Female", "Male", "Female", "Male"],
            "ID": [1, 2, 3, 4],
        }
    )

    result = apply_random_missingness(df, percent=50, seed=7, exclude_columns={"ID"})
    repeated = apply_random_missingness(df, percent=50, seed=7, exclude_columns={"ID"})

    assert df.isna().sum().sum() == 0
    assert result.equals(repeated)
    assert result["ID"].equals(df["ID"])
    assert result[["Age", "Gender"]].isna().sum().sum() > 0


def test_apply_random_missingness_zero_percent_leaves_data_unchanged():
    df = pd.DataFrame({"Age": [10, 20], "Gender": ["Female", "Male"]})

    result = apply_random_missingness(df, percent=0, seed=7)

    assert result.equals(df)


def test_prepare_jittered_scatter_values_supports_categorical_axes():
    df = pd.DataFrame(
        {
            "Gender": ["Female", "Male", "Female", "Male"],
            "Nearest Pump": ["Pump A", "Pump A", "Pump B", "Pump B"],
        }
    )

    prepared = prepare_jittered_scatter_values(
        df,
        x_column="Gender",
        y_column="Nearest Pump",
        seed=3,
    )

    assert prepared["x_tick_labels"] == ["Female", "Male"]
    assert prepared["y_tick_labels"] == ["Pump A", "Pump B"]
    assert len(prepared["x_values"]) == len(df)
    assert len(prepared["y_values"]) == len(df)
    assert prepared["x_values"].nunique() > 2
    assert prepared["y_values"].nunique() > 2
