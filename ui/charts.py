# charts.py
from __future__ import annotations

import altair as alt
import pandas as pd


_ADAPTIVE_AXIS_EXPR = """
    datum.value >= 1e9 ? format(datum.value / 1e9, '.0f') + ' B' :
    datum.value >= 1e6 ? format(datum.value / 1e6, '.0f') + ' M' :
    datum.value >= 1e3 ? format(datum.value / 1e3, '.0f') + ' K' :
    format(datum.value, ',')
"""


def build_time_series_area_chart(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str = "",
    height: int = 250,
) -> alt.Chart:
    """
    Build an Altair area chart with adaptive y-axis labels like:
    12 K, 3.4 M, 20 B.
    """
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