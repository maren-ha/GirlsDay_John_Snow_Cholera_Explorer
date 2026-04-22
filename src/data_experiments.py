import numpy as np
import pandas as pd


def apply_random_missingness(df, percent, seed=0, exclude_columns=None):
    if percent <= 0:
        return df.copy()

    result = df.copy()
    excluded = set(exclude_columns or ())
    columns = [column for column in result.columns if column not in excluded]
    if not columns:
        return result

    probability = min(float(percent), 100.0) / 100.0
    rng = np.random.default_rng(seed)
    mask = rng.random((len(result), len(columns))) < probability
    for column_index, column in enumerate(columns):
        result[column] = result[column].astype("object").mask(mask[:, column_index])
    return result


def _is_numeric(series):
    return pd.api.types.is_numeric_dtype(series)


def _encode_axis(series, seed, jitter_width):
    clean_series = series.dropna()
    if _is_numeric(clean_series):
        return pd.to_numeric(series, errors="coerce"), None

    categories = list(pd.unique(clean_series))
    category_positions = {category: index for index, category in enumerate(categories)}
    base_values = series.map(category_positions).astype(float)
    rng = np.random.default_rng(seed)
    jitter = rng.uniform(-jitter_width, jitter_width, size=len(series))
    return base_values + jitter, categories


def prepare_jittered_scatter_values(df, x_column, y_column, seed=0, jitter_width=0.18):
    x_values, x_categories = _encode_axis(df[x_column], seed, jitter_width)
    y_values, y_categories = _encode_axis(df[y_column], seed + 1, jitter_width)

    return {
        "x_values": x_values,
        "y_values": y_values,
        "x_tick_positions": list(range(len(x_categories))) if x_categories is not None else None,
        "x_tick_labels": x_categories,
        "y_tick_positions": list(range(len(y_categories))) if y_categories is not None else None,
        "y_tick_labels": y_categories,
    }
