from __future__ import annotations

import pandas as pd


def top_x_by_mentions(df: pd.DataFrame, column: str, top_n: int = 10) -> pd.DataFrame:
    """
    Return top values in a column by summed Mentions,
    falling back to row counts if Mentions is unavailable.
    """
    if column not in df.columns or df.empty:
        return pd.DataFrame()

    working = df.copy()
    working[column] = working[column].fillna("").astype(str).str.strip()
    working = working[working[column] != ""]

    if working.empty:
        return pd.DataFrame()

    if "Mentions" in working.columns:
        return (
            working.groupby(column, dropna=False)["Mentions"]
            .sum()
            .sort_values(ascending=False)
            .head(top_n)
            .reset_index()
        )

    return (
        working[column]
        .value_counts()
        .head(top_n)
        .rename_axis(column)
        .reset_index(name="Count")
    )