# missing_authors.py

from __future__ import annotations

import numpy as np
import pandas as pd


HIDDEN_FLAGS = {"Good Outlet", "Aggregator"}


def init_missing_authors_state(session_state) -> None:
    """Initialize session keys used by the missing authors page."""
    if "last_author_fix" not in session_state:
        session_state.last_author_fix = None

    if "auth_reviewed_count" not in session_state:
        session_state.auth_reviewed_count = 0

    if "auth_skip_counter" not in session_state:
        session_state.auth_skip_counter = 0


def prepare_author_working_df(
    df_traditional: pd.DataFrame,
    excluded_flags: list[str] | None = None,
) -> pd.DataFrame:
    """Build working dataframe for the missing-author workflow."""
    df = df_traditional.copy()

    if "Author" in df.columns:
        df["Author"] = df["Author"].replace("", np.nan)

    excluded_flags = excluded_flags or []

    if excluded_flags and "Coverage Flags" in df.columns:
        df = df[~df["Coverage Flags"].fillna("").isin(excluded_flags)].copy()

    return df


def get_available_visible_flags(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return visible flag options and sensible default exclusions."""
    if "Coverage Flags" not in df.columns:
        return [], []

    available_flags = sorted(
        [
            f
            for f in df["Coverage Flags"].fillna("").astype(str).unique().tolist()
            if f.strip()
        ]
    )

    default_excluded_flags = [
        f
        for f in [
            "Newswire?",
            "Market Report Spam?",
            "Stocks / Financials?",
            "Advertorial?",
            "User-Generated",
        ]
        if f in available_flags
    ]

    visible_flags = [f for f in available_flags if f not in HIDDEN_FLAGS]
    visible_defaults = [f for f in default_excluded_flags if f not in HIDDEN_FLAGS]

    return visible_flags, visible_defaults


def build_fixable_headline_table(df: pd.DataFrame) -> pd.DataFrame:
    """Return headline-level table of fixable missing-author cases."""
    required = {"Headline", "Mentions", "Author"}
    if df is None or df.empty or not required.issubset(df.columns):
        return pd.DataFrame(columns=["Headline", "Total", "Known", "Missing"])

    headline_table = df[["Headline", "Mentions", "Author"]].copy()
    headline_table = headline_table.groupby("Headline").count()
    headline_table["Missing"] = headline_table["Mentions"] - headline_table["Author"]
    headline_table = headline_table[
        (headline_table["Author"] > 0) & (headline_table["Missing"] > 0)
    ].sort_values("Missing", ascending=False).reset_index()

    headline_table.rename(
        columns={"Author": "Known", "Mentions": "Total"},
        inplace=True,
        errors="raise",
    )

    return headline_table


def fixable_headline_stats(
    df: pd.DataFrame,
    counter: int,
    primary: str = "Headline",
    secondary: str = "Author",
) -> dict:
    """Return stats on how many author fields can be fixed."""
    if df is None or df.empty or primary not in df.columns or secondary not in df.columns:
        return {
            "total": 0,
            "total_known": 0,
            "percent_known": "0%",
            "fixable": 0,
            "fixable_headline_count": 0,
            "remaining": 0,
            "percent_knowable": "0%",
        }

    total = df["Mentions"].count() if "Mentions" in df.columns else len(df)

    if total == 0:
        return {
            "total": 0,
            "total_known": 0,
            "percent_known": "0%",
            "fixable": 0,
            "fixable_headline_count": 0,
            "remaining": 0,
            "percent_knowable": "0%",
        }

    headline_stats = pd.pivot_table(
        df,
        index=primary,
        values=["Mentions", secondary],
        aggfunc="count",
    )

    headline_stats["Missing"] = headline_stats["Mentions"] - headline_stats[secondary]
    missing = headline_stats["Missing"].sum()

    headline_stats = headline_stats[headline_stats[secondary] > 0]
    headline_stats = headline_stats[headline_stats["Missing"] > 0]

    fixable = headline_stats["Missing"].sum()
    fixable_headline_count = headline_stats["Missing"].count()
    remaining = max(fixable_headline_count - counter, 0)

    total_known = total - missing
    percent_known = "{:.0%}".format(total_known / total) if total > 0 else "0%"
    percent_knowable = (
        "{:.0%}".format((total - (missing - fixable)) / total) if total > 0 else "0%"
    )

    return {
        "total": total,
        "total_known": total_known,
        "percent_known": percent_known,
        "fixable": fixable,
        "fixable_headline_count": fixable_headline_count,
        "remaining": remaining,
        "percent_knowable": percent_knowable,
    }


def get_headline_authors(df: pd.DataFrame, headline_text: str) -> pd.DataFrame:
    """
    Return known authors already associated with a headline,
    along with how many times each appears.
    """
    if df is None or df.empty or "Headline" not in df.columns:
        return pd.DataFrame(columns=["Possible Author(s)", "Count"])

    working = df[df["Headline"] == headline_text].copy()

    if "Author" not in working.columns:
        return pd.DataFrame(columns=["Possible Author(s)", "Count"])

    working["Author"] = working["Author"].replace("", np.nan)
    working = working[working["Author"].notna()].copy()
    working["Author"] = working["Author"].astype(str).str.strip()
    working = working[working["Author"] != ""]

    if working.empty:
        return pd.DataFrame(columns=["Possible Author(s)", "Count"])

    author_counts = (
        working.groupby("Author", dropna=False)
        .size()
        .reset_index(name="Count")
        .sort_values(["Count", "Author"], ascending=[False, True])
        .reset_index(drop=True)
    )

    author_counts = author_counts.rename(columns={"Author": "Possible Author(s)"})
    return author_counts


def get_possible_authors(df: pd.DataFrame, headline_text: str) -> list[str]:
    """Return possible author strings for a headline."""
    authors_df = get_headline_authors(df, headline_text)

    if authors_df.empty or "Possible Author(s)" not in authors_df.columns:
        return [""]

    possibles = authors_df["Possible Author(s)"].dropna().astype(str).tolist()
    return possibles if possibles else [""]


def apply_author_fix(
    df: pd.DataFrame,
    headline_text: str,
    new_author: str,
) -> pd.DataFrame:
    """Apply author update to rows with matching headline and missing/blank author."""
    updated = df.copy()

    if "Headline" not in updated.columns or "Author" not in updated.columns:
        return updated

    author_blank_mask = (
        updated["Author"].isna()
        | updated["Author"].astype(str).str.strip().eq("")
    )

    match_mask = (updated["Headline"] == headline_text) & author_blank_mask
    updated.loc[match_mask, "Author"] = new_author

    return updated


def build_last_author_fix_payload(
    df: pd.DataFrame,
    headline_text: str,
    previous_reviewed_count: int,
) -> dict:
    """Capture enough info to undo the next author fix."""
    matching_rows = df.index[df["Headline"] == headline_text].tolist()

    return {
        "headline": headline_text,
        "row_indexes": matching_rows,
        "previous_authors": df.loc[matching_rows, "Author"].copy(),
        "previous_reviewed_count": previous_reviewed_count,
    }


def undo_last_author_fix(session_state) -> None:
    """
    Undo the most recent author fix applied on this page.
    Restores only the previously affected rows' Author values.
    """
    last_fix = session_state.get("last_author_fix")

    if not last_fix:
        return

    row_indexes = last_fix.get("row_indexes", [])
    previous_authors = last_fix.get("previous_authors")
    previous_reviewed_count = last_fix.get("previous_reviewed_count", 0)

    if not row_indexes or previous_authors is None:
        session_state.last_author_fix = None
        return

    valid_row_indexes = [
        idx for idx in row_indexes if idx in session_state.df_traditional.index
    ]

    if len(valid_row_indexes) == len(previous_authors):
        session_state.df_traditional.loc[valid_row_indexes, "Author"] = previous_authors.values

    session_state.auth_reviewed_count = previous_reviewed_count
    session_state.last_author_fix = None