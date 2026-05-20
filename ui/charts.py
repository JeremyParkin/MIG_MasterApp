# charts.py
from __future__ import annotations

import importlib
from typing import Any

import pandas as pd


_ADAPTIVE_AXIS_EXPR = """
    datum.value >= 1e9 ? (((datum.value / 1e9) % 1) == 0 ? format(datum.value / 1e9, '.0f') : format(datum.value / 1e9, '.1f')) + ' B' :
    datum.value >= 1e6 ? (((datum.value / 1e6) % 1) == 0 ? format(datum.value / 1e6, '.0f') : format(datum.value / 1e6, '.1f')) + ' M' :
    datum.value >= 1e3 ? (((datum.value / 1e3) % 1) == 0 ? format(datum.value / 1e3, '.0f') : format(datum.value / 1e3, '.1f')) + ' K' :
    format(datum.value, ',')
"""


def get_adaptive_axis_expr() -> str:
    """Return a shared Altair/Vega labelExpr for compact numeric axis labels."""
    return _ADAPTIVE_AXIS_EXPR


def format_compact_number(value: float | int | object) -> str:
    """Format a number compactly without creating duplicate rounded tick labels."""
    try:
        n = float(value)
    except Exception:
        return str(value)

    abs_n = abs(n)
    if abs_n >= 1_000_000_000:
        scaled = n / 1_000_000_000
        suffix = "B"
    elif abs_n >= 1_000_000:
        scaled = n / 1_000_000
        suffix = "M"
    elif abs_n >= 1_000:
        scaled = n / 1_000
        suffix = "K"
    else:
        return str(int(n)) if n.is_integer() else str(round(n, 1))

    rounded = round(scaled)
    if abs(scaled - rounded) < 1e-9:
        return f"{rounded:.0f}{suffix}"
    return f"{scaled:.1f}{suffix}"


def build_time_series_area_chart(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str = "",
    height: int = 250,
) -> Any | None:
    """
    Build an Altair area chart with adaptive y-axis labels like:
    12 K, 3.4 M, 1.2 B.
    """
    try:
        alt = importlib.import_module("altair")
    except Exception:
        return None

    return (
        alt.Chart(df)
        .mark_area()
        .encode(
            x=alt.X(
                f"{x_col}:T",
                title="Date",
            ),
            y=alt.Y(
                f"{y_col}:Q",
                title=y_col,
                axis=alt.Axis(labelExpr=_ADAPTIVE_AXIS_EXPR),
            ),
            tooltip=[
                alt.Tooltip(f"{x_col}:T", title="Date"),
                alt.Tooltip(f"{y_col}:Q", title=y_col, format=","),
            ],
        )
        .properties(
            title=title,
            height=height,
        )
    )
