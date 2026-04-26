from __future__ import annotations

import re
from typing import Any

import pandas as pd


PROMINENCE_LABEL_WEIGHTS = {
    "very high": 1.75,
    "high": 1.0,
    "moderate": 0.25,
    "medium": 0.25,
    "low": -0.85,
    "very low": -1.45,
    "none": -2.1,
}


def get_prominence_columns(df: pd.DataFrame | None) -> list[str]:
    if df is None or df.empty:
        return []
    return [
        col
        for col in df.columns
        if isinstance(col, str) and col.startswith("Prominence")
    ]


def normalize_selected_prominence_column(
    selected_column: str | None,
    available_columns: list[str] | None,
) -> str:
    selected = str(selected_column or "").strip()
    if not selected:
        return ""
    available = [str(col).strip() for col in (available_columns or []) if str(col).strip()]
    return selected if selected in available else ""


def get_prominence_column_preview(
    df: pd.DataFrame | None,
    *,
    columns: list[str] | None = None,
    limit: int = 8,
) -> dict[str, list[str]]:
    if df is None or df.empty:
        return {}

    target_columns = columns or get_prominence_columns(df)
    preview: dict[str, list[str]] = {}
    for col in target_columns:
        if col not in df.columns:
            continue
        values = (
            df[col]
            .fillna("")
            .astype(str)
            .str.strip()
        )
        unique_values = [value for value in values.drop_duplicates().tolist() if value]
        preview[str(col)] = unique_values[:limit]
    return preview


def _normalize_prominence_label(value: Any) -> str:
    text = str(value or "").strip().casefold()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def prominence_value_weight(value: Any) -> float:
    if value is None:
        return 0.0

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        numeric = float(value)
        return max(-1.5, min(1.5, numeric))

    normalized = _normalize_prominence_label(value)
    if not normalized:
        return 0.0

    if normalized in PROMINENCE_LABEL_WEIGHTS:
        return PROMINENCE_LABEL_WEIGHTS[normalized]

    compact = normalized.replace(" ", "")
    alias_map = {
        "veryhigh": "very high",
        "high": "high",
        "moderate": "moderate",
        "medium": "moderate",
        "low": "low",
        "verylow": "very low",
        "none": "none",
    }
    if compact in alias_map:
        return PROMINENCE_LABEL_WEIGHTS[alias_map[compact]]

    return 0.0


def get_prominence_weight_series(
    df: pd.DataFrame,
    selected_column: str | None,
) -> pd.Series:
    if df is None or df.empty:
        return pd.Series(dtype="float64")

    selected = str(selected_column or "").strip()
    if not selected or selected not in df.columns:
        return pd.Series(0.0, index=df.index, dtype="float64")

    series = df[selected]
    if pd.api.types.is_numeric_dtype(series):
        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.notna().any():
            min_val = float(numeric.min())
            max_val = float(numeric.max())
            if max_val > min_val:
                scaled = ((numeric - min_val) / (max_val - min_val)) * 3.0 - 1.5
                return scaled.fillna(0.0).clip(lower=-1.5, upper=1.5)
            return pd.Series(0.0, index=df.index, dtype="float64")

    return series.apply(prominence_value_weight).astype(float)
