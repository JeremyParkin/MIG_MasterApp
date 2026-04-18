# missing_authors.py

from __future__ import annotations

import math
import re

import numpy as np
import pandas as pd


HIDDEN_FLAGS = {"Good Outlet", "Aggregator"}


def _required_known_author_count(total_rows: int) -> int:
    total_rows = max(int(total_rows or 0), 0)
    if total_rows <= 10:
        return 1
    if total_rows <= 20:
        return 2
    return max(2, math.ceil(total_rows * 0.10))


def _is_quality_author_string(value: object) -> bool:
    text = str(value or "").strip()
    if not text:
        return False

    lowered = text.lower()
    if any(marker in lowered for marker in ["http://", "https://", "www.", "@"]):
        return False

    words = [word for word in re.split(r"\s+", text) if word]
    if not 2 <= len(words) <= 5:
        return False

    alpha_chars = sum(1 for ch in text if ch.isalpha())
    digit_chars = sum(1 for ch in text if ch.isdigit())
    punct_chars = sum(1 for ch in text if not ch.isalnum() and not ch.isspace())
    total_chars = sum(1 for ch in text if not ch.isspace())

    if total_chars == 0:
        return False

    if digit_chars > 2:
        return False

    if punct_chars > max(2, total_chars * 0.15):
        return False

    if (alpha_chars / total_chars) < 0.65:
        return False

    return True


def _split_camel_case(text: str) -> str:
    return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text)


def _author_name_tokens(value: object) -> list[str]:
    text = _split_camel_case(str(value or "").strip())
    text = re.sub(r"[^A-Za-z\s]", " ", text)
    tokens = [token.lower() for token in re.split(r"\s+", text) if token]
    return tokens


def _author_cluster_key(value: object) -> str:
    tokens = _author_name_tokens(value)
    if len(tokens) >= 2:
        return " ".join(tokens[:2])
    return " ".join(tokens)


def _split_multi_author_name(value: object) -> list[str]:
    text = str(value or "").strip()
    if not text:
        return []

    parts = re.split(r"\s+(?:and|&)\s+", text, flags=re.IGNORECASE)
    parts = [part.strip() for part in parts if part.strip()]
    return parts if len(parts) > 1 else []


def _headline_author_signal(df: pd.DataFrame, headline_text: str) -> dict:
    working = df[df["Headline"] == headline_text].copy()
    total = len(working)
    if total == 0 or "Author" not in working.columns:
        return {
            "total": total,
            "known": 0,
            "missing": total,
            "required_known": _required_known_author_count(total),
            "best_cluster_size": 0,
            "best_cluster_key": "",
            "passes": False,
        }

    working["Author"] = working["Author"].replace("", np.nan)
    working = working[working["Author"].notna()].copy()
    working["Author"] = working["Author"].astype(str).str.strip()
    working = working[working["Author"] != ""]
    working = working[working["Author"].apply(_is_quality_author_string)].copy()

    known = len(working)
    missing = max(total - known, 0)
    required_known = _required_known_author_count(total)

    if working.empty:
        return {
            "total": total,
            "known": 0,
            "missing": missing,
            "required_known": required_known,
            "best_cluster_size": 0,
            "best_cluster_key": "",
            "passes": False,
        }

    working["_cluster_key"] = working["Author"].apply(_author_cluster_key)
    cluster_counts = (
        working[working["_cluster_key"] != ""]
        .groupby("_cluster_key")
        .size()
        .sort_values(ascending=False)
    )

    if cluster_counts.empty:
        best_cluster_size = 0
        best_cluster_key = ""
    else:
        best_cluster_key = str(cluster_counts.index[0])
        best_cluster_size = int(cluster_counts.iloc[0])

    has_strong_cluster = (
        best_cluster_size >= required_known
        or (best_cluster_size >= 2 and best_cluster_size == known)
    )
    passes = known >= required_known and missing > 0 and has_strong_cluster

    return {
        "total": total,
        "known": known,
        "missing": missing,
        "required_known": required_known,
        "best_cluster_size": best_cluster_size,
        "best_cluster_key": best_cluster_key,
        "passes": passes,
    }


def _build_headline_signal_table(df: pd.DataFrame) -> pd.DataFrame:
    """Compute headline-level author-fix signals in one grouped pass."""
    required = {"Headline", "Author"}
    if df is None or df.empty or not required.issubset(df.columns):
        return pd.DataFrame(
            columns=[
                "Headline",
                "Total",
                "Known",
                "Missing",
                "Required_Known",
                "Best_Cluster_Size",
                "Best_Cluster_Key",
                "Passes",
            ]
        )

    working = df.copy()
    working["Headline"] = working["Headline"].fillna("").astype(str).str.strip()
    working = working[working["Headline"] != ""].copy()
    if working.empty:
        return pd.DataFrame(
            columns=[
                "Headline",
                "Total",
                "Known",
                "Missing",
                "Required_Known",
                "Best_Cluster_Size",
                "Best_Cluster_Key",
                "Passes",
            ]
        )

    total_by_headline = (
        working.groupby("Headline")
        .size()
        .rename("Total")
        .reset_index()
    )

    quality = working.copy()
    quality["Author"] = quality["Author"].replace("", np.nan)
    quality = quality[quality["Author"].notna()].copy()
    quality["Author"] = quality["Author"].astype(str).str.strip()
    quality = quality[quality["Author"] != ""].copy()
    quality = quality[quality["Author"].apply(_is_quality_author_string)].copy()

    if quality.empty:
        signal_table = total_by_headline.copy()
        signal_table["Known"] = 0
        signal_table["Missing"] = signal_table["Total"]
        signal_table["Required_Known"] = signal_table["Total"].apply(_required_known_author_count)
        signal_table["Best_Cluster_Size"] = 0
        signal_table["Best_Cluster_Key"] = ""
        signal_table["Passes"] = False
        return signal_table

    quality["_cluster_key"] = quality["Author"].apply(_author_cluster_key)
    quality = quality[quality["_cluster_key"] != ""].copy()

    if quality.empty:
        signal_table = total_by_headline.copy()
        signal_table["Known"] = 0
        signal_table["Missing"] = signal_table["Total"]
        signal_table["Required_Known"] = signal_table["Total"].apply(_required_known_author_count)
        signal_table["Best_Cluster_Size"] = 0
        signal_table["Best_Cluster_Key"] = ""
        signal_table["Passes"] = False
        return signal_table

    known_by_headline = (
        quality.groupby("Headline")
        .size()
        .rename("Known")
        .reset_index()
    )

    cluster_counts = (
        quality.groupby(["Headline", "_cluster_key"])
        .size()
        .rename("Cluster_Count")
        .reset_index()
        .sort_values(["Headline", "Cluster_Count", "_cluster_key"], ascending=[True, False, True])
    )

    best_cluster = (
        cluster_counts
        .drop_duplicates(subset=["Headline"], keep="first")
        .rename(columns={"_cluster_key": "Best_Cluster_Key", "Cluster_Count": "Best_Cluster_Size"})
    )

    signal_table = total_by_headline.merge(known_by_headline, on="Headline", how="left")
    signal_table = signal_table.merge(
        best_cluster[["Headline", "Best_Cluster_Key", "Best_Cluster_Size"]],
        on="Headline",
        how="left",
    )

    signal_table["Known"] = signal_table["Known"].fillna(0).astype(int)
    signal_table["Best_Cluster_Size"] = signal_table["Best_Cluster_Size"].fillna(0).astype(int)
    signal_table["Best_Cluster_Key"] = signal_table["Best_Cluster_Key"].fillna("")
    signal_table["Missing"] = (signal_table["Total"] - signal_table["Known"]).clip(lower=0).astype(int)
    signal_table["Required_Known"] = signal_table["Total"].apply(_required_known_author_count)

    has_strong_cluster = (
        (signal_table["Best_Cluster_Size"] >= signal_table["Required_Known"])
        | (
            (signal_table["Best_Cluster_Size"] >= 2)
            & (signal_table["Best_Cluster_Size"] == signal_table["Known"])
        )
    )

    signal_table["Passes"] = (
        (signal_table["Known"] >= signal_table["Required_Known"])
        & (signal_table["Missing"] > 0)
        & has_strong_cluster
    )

    return signal_table


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
            "Press Release",
            "Market Report Spam",
            "Financial Outlet",
            "Advertorial",
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

    signal_table = _build_headline_signal_table(df)
    fixable = signal_table[signal_table["Passes"]].copy()

    if fixable.empty:
        return pd.DataFrame(columns=["Headline", "Total", "Known", "Missing"])

    return (
        fixable[["Headline", "Total", "Known", "Missing"]]
        .sort_values(["Missing", "Total"], ascending=False)
        .reset_index(drop=True)
    )


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

    signal_table = _build_headline_signal_table(df.rename(columns={primary: "Headline", secondary: "Author"}))
    missing = int(signal_table["Missing"].sum()) if not signal_table.empty else 0

    fixable_rows = signal_table[signal_table["Passes"]] if not signal_table.empty else signal_table
    fixable = int(fixable_rows["Missing"].sum()) if not fixable_rows.empty else 0
    fixable_headline_count = int(len(fixable_rows))
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
        return pd.DataFrame(columns=["Possible Author(s)", "Count", "In Signal"])

    working = df[df["Headline"] == headline_text].copy()

    if "Author" not in working.columns:
        return pd.DataFrame(columns=["Possible Author(s)", "Count", "In Signal"])

    working["Author"] = working["Author"].replace("", np.nan)
    working = working[working["Author"].notna()].copy()
    working["Author"] = working["Author"].astype(str).str.strip()
    working = working[working["Author"] != ""]
    working = working[working["Author"].apply(_is_quality_author_string)].copy()
    if working.empty:
        return pd.DataFrame(columns=["Possible Author(s)", "Count", "In Signal"])

    working["_cluster_key"] = working["Author"].apply(_author_cluster_key)
    working = working[working["_cluster_key"] != ""].copy()

    if working.empty:
        return pd.DataFrame(columns=["Possible Author(s)", "Count", "In Signal"])

    cluster_counts = (
        working.groupby("_cluster_key")
        .size()
        .sort_values(ascending=False)
    )
    if cluster_counts.empty:
        return pd.DataFrame(columns=["Possible Author(s)", "Count", "In Signal"])

    best_cluster_key = str(cluster_counts.index[0])
    author_counts = (
        working.groupby("Author", dropna=False)
        .size()
        .reset_index(name="Count")
        .sort_values(["Count", "Author"], ascending=[False, True])
        .reset_index(drop=True)
    )
    author_cluster_map = (
        working[["Author", "_cluster_key"]]
        .drop_duplicates(subset=["Author"], keep="first")
        .set_index("Author")["_cluster_key"]
        .to_dict()
    )
    signal_names = set(
        working.loc[working["_cluster_key"] == best_cluster_key, "Author"]
        .dropna()
        .astype(str)
        .tolist()
    )

    expanded_signal_names = set(signal_names)
    for signal_name in list(signal_names):
        for split_name in _split_multi_author_name(signal_name):
            if split_name:
                expanded_signal_names.add(split_name)

    author_counts["In Signal"] = author_counts["Author"].map(
        lambda name: (
            author_cluster_map.get(name, "") == best_cluster_key
            or str(name) in expanded_signal_names
        )
    )
    author_counts = author_counts.rename(columns={"Author": "Possible Author(s)"})
    return author_counts


def get_possible_authors(df: pd.DataFrame, headline_text: str) -> list[str]:
    """Return possible author strings for a headline."""
    authors_df = get_headline_authors(df, headline_text)

    if authors_df.empty or "Possible Author(s)" not in authors_df.columns:
        return [""]

    if "In Signal" in authors_df.columns:
        authors_df = authors_df.sort_values(
            ["In Signal", "Count", "Possible Author(s)"],
            ascending=[False, False, True],
        ).reset_index(drop=True)

    possibles = authors_df["Possible Author(s)"].dropna().astype(str).tolist()
    if not possibles:
        return [""]

    possible_set = set(possibles)
    reordered: list[str] = []

    for name in possibles:
        split_names = _split_multi_author_name(name)
        if not split_names:
            continue
        first_name = split_names[0]
        if first_name in possible_set and first_name not in reordered:
            reordered.append(first_name)

    for name in possibles:
        if name not in reordered:
            reordered.append(name)

    possibles = reordered
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
