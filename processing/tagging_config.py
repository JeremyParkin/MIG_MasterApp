# tagging_config.py
from __future__ import annotations

import math
from typing import Literal

import pandas as pd


SampleMode = Literal["full", "representative", "custom", "reuse_other_sample"]

DEFAULT_MAX_FULL_ROWS = 2000
DEFAULT_EXCLUDED_COVERAGE_FLAGS = [
    "Press Release",
    "Market Report Spam",
    "Financial Outlet",
    "Advertorial",
    "User-Generated",
]


def init_tagging_config_state(session_state) -> None:
    defaults = {
        "tagging_config_step": False,
        "tagging_sample_mode": "representative",
        "tagging_sample_size": None,
        "tagging_full_override": False,
        "df_tagging_rows": pd.DataFrame(),
        "df_tagging_unique": pd.DataFrame(),
        "tagging_elapsed_time": 0.0,
        "tagging_excluded_flags": [],
    }

    for key, value in defaults.items():
        if key not in session_state:
            session_state[key] = value


def calculate_representative_sample_size(
    population_size: int,
    confidence_level: float = 0.95,
    margin_of_error: float = 0.05,
    p: float = 0.5,
) -> int:
    """
    Standard finite population sample size estimate.
    """
    if population_size <= 0:
        return 0

    z = 1.96 if confidence_level == 0.95 else 1.96

    numerator = population_size * (z ** 2) * p * (1 - p)
    denominator = (margin_of_error ** 2) * (population_size - 1) + (z ** 2) * p * (1 - p)

    return max(1, math.ceil(numerator / denominator))


def get_tagging_source_rows(df_traditional: pd.DataFrame) -> pd.DataFrame:
    """
    Start from cleaned traditional data with canonical Group ID already assigned.
    Social is intentionally excluded for now.
    """
    if df_traditional is None or df_traditional.empty:
        return pd.DataFrame()

    df = df_traditional.copy()

    if "Type" in df.columns:
        df = df[
            ~df["Type"].isin(
                ["FACEBOOK", "INSTAGRAM", "X", "TWITTER", "LINKEDIN", "TIKTOK", "YOUTUBE", "REDDIT", "BLUESKY"]
            )
        ].copy()

    return df.reset_index(drop=True)


def get_available_coverage_flags(df_rows: pd.DataFrame) -> tuple[list[str], list[str]]:
    if df_rows is None or df_rows.empty or "Coverage Flags" not in df_rows.columns:
        return [], []

    available_flags = sorted(
        [
            f
            for f in df_rows["Coverage Flags"].fillna("").astype(str).unique().tolist()
            if f.strip()
        ]
    )
    default_flags = [f for f in DEFAULT_EXCLUDED_COVERAGE_FLAGS if f in available_flags]
    return available_flags, default_flags


def apply_coverage_flag_exclusions(
    df_rows: pd.DataFrame,
    excluded_flags: list[str] | None = None,
) -> pd.DataFrame:
    if df_rows is None or df_rows.empty:
        return pd.DataFrame()

    excluded_flags = excluded_flags or []
    if not excluded_flags or "Coverage Flags" not in df_rows.columns:
        return df_rows.copy().reset_index(drop=True)

    return df_rows[~df_rows["Coverage Flags"].fillna("").isin(excluded_flags)].copy().reset_index(drop=True)


def sample_tagging_rows(
    df_rows: pd.DataFrame,
    sample_mode: SampleMode,
    custom_sample_size: int | None = None,
    max_full_rows: int = DEFAULT_MAX_FULL_ROWS,
    full_override: bool = False,
    random_state: int = 1,
) -> tuple[pd.DataFrame, int]:
    """
    Sample row-level mentions. Group IDs are preserved from the original cleaned dataset.
    """
    if df_rows is None or df_rows.empty:
        return pd.DataFrame(), 0

    population_size = len(df_rows)

    if sample_mode == "full":
        if population_size > max_full_rows and not full_override:
            effective_n = max_full_rows
            sampled = df_rows.sample(n=effective_n, random_state=random_state).reset_index(drop=True)
            return sampled, effective_n
        return df_rows.copy().reset_index(drop=True), population_size

    if sample_mode == "representative":
        effective_n = calculate_representative_sample_size(population_size)
        effective_n = min(effective_n, population_size)
        sampled = df_rows.sample(n=effective_n, random_state=random_state).reset_index(drop=True)
        return sampled, effective_n

    if sample_mode == "custom":
        effective_n = int(custom_sample_size or 0)
        effective_n = max(1, min(effective_n, population_size))
        sampled = df_rows.sample(n=effective_n, random_state=random_state).reset_index(drop=True)
        return sampled, effective_n

    return df_rows.copy().reset_index(drop=True), population_size


def build_unique_story_table_from_existing_groups(df_rows: pd.DataFrame) -> pd.DataFrame:
    """
    Build one-row-per-group table using the Prime Example row.
    """
    if df_rows is None or df_rows.empty:
        return pd.DataFrame()

    if "Group ID" not in df_rows.columns:
        raise ValueError("Group ID missing.")

    if "Prime Example" not in df_rows.columns:
        raise ValueError("Prime Example missing.")

    working = df_rows.copy()

    metrics = working.groupby("Group ID").agg({
        col: "sum" for col in ["Mentions", "Impressions", "Effective Reach"]
        if col in working.columns
    }).reset_index()

    group_counts = working.groupby("Group ID").size().reset_index(name="Group Count")
    metrics = metrics.merge(group_counts, on="Group ID", how="left")

    prime_rows = working[working["Prime Example"] == 1].copy()
    prime_rows = prime_rows.drop_duplicates(subset=["Group ID"], keep="first")

    prime_rows = prime_rows.drop(
        columns=["Mentions", "Impressions", "Effective Reach", "Group Count"],
        errors="ignore",
    )

    unique = prime_rows.merge(metrics, on="Group ID", how="left")

    return unique.reset_index(drop=True)


def ensure_prime_rows_in_sample(
    sampled_rows: pd.DataFrame,
    full_rows: pd.DataFrame,
) -> pd.DataFrame:
    """
    Ensure that for every Group ID represented in the sample,
    the Prime Example row is present. If not, swap it in.
    """
    if sampled_rows.empty or "Group ID" not in sampled_rows.columns:
        return sampled_rows

    if "Prime Example" not in sampled_rows.columns:
        raise ValueError("Prime Example column missing from sample. Run Basic Cleaning first.")

    if full_rows.empty or "Group ID" not in full_rows.columns or "Prime Example" not in full_rows.columns:
        raise ValueError("Prime Example column missing from full source rows. Run Basic Cleaning first.")

    working = sampled_rows.copy()
    group_ids = working["Group ID"].dropna().unique()

    for gid in group_ids:
        group = working[working["Group ID"] == gid]

        if (group["Prime Example"] == 1).any():
            continue

        full_group = full_rows[full_rows["Group ID"] == gid]
        prime_row = full_group[full_group["Prime Example"] == 1]

        if prime_row.empty:
            continue

        idx_to_replace = group.index[0]
        working.loc[idx_to_replace] = prime_row.iloc[0]

    return working.reset_index(drop=True)


def prepare_tagging_datasets(
    df_traditional: pd.DataFrame,
    sample_mode: SampleMode,
    excluded_flags: list[str] | None = None,
    custom_sample_size: int | None = None,
    max_full_rows: int = DEFAULT_MAX_FULL_ROWS,
    full_override: bool = False,
    random_state: int = 1,
    reused_rows: pd.DataFrame | None = None,
) -> dict:
    """
    Build tagging-specific datasets:
    - df_tagging_rows: sampled row-level rows with original Group ID
    - df_tagging_unique: one row per original Group ID represented in the sample
    """
    source_rows = get_tagging_source_rows(df_traditional)
    source_rows = apply_coverage_flag_exclusions(source_rows, excluded_flags)

    if sample_mode == "reuse_other_sample":
        if reused_rows is None or reused_rows.empty:
            raise ValueError("No reusable sentiment sample found.")
        sampled_rows = reused_rows.copy().reset_index(drop=True)
        effective_sample_size = len(sampled_rows)
    else:
        sampled_rows, effective_sample_size = sample_tagging_rows(
            source_rows,
            sample_mode=sample_mode,
            custom_sample_size=custom_sample_size,
            max_full_rows=max_full_rows,
            full_override=full_override,
            random_state=random_state,
        )

    sampled_rows = ensure_prime_rows_in_sample(sampled_rows, source_rows)

    if not sampled_rows.empty and "Group ID" not in sampled_rows.columns:
        raise ValueError("Sampled tagging rows do not contain Group ID. Standard Cleaning must run first.")

    unique_rows = build_unique_story_table_from_existing_groups(sampled_rows)

    if "Tag_Processed" not in unique_rows.columns:
        unique_rows["Tag_Processed"] = False

    for col in ["AI Tags", "AI Tag Rationale"]:
        if col not in unique_rows.columns:
            unique_rows[col] = ""

    return {
        "df_tagging_rows": sampled_rows.reset_index(drop=True),
        "df_tagging_unique": unique_rows.reset_index(drop=True),
        "population_size": len(source_rows),
        "sample_size_used": effective_sample_size,
        "unique_story_count": len(unique_rows),
    }


def reset_tagging_config_state(session_state) -> None:
    session_state.tagging_config_step = False
    session_state.tagging_sample_mode = "representative"
    session_state.tagging_sample_size = None
    session_state.tagging_full_override = False
    session_state.df_tagging_rows = pd.DataFrame()
    session_state.df_tagging_unique = pd.DataFrame()
    session_state.tagging_elapsed_time = 0.0
    session_state.tagging_excluded_flags = []
    for key in [
        "tagging_review_idx",
        "tagging_review_mode",
        "tagging_pre_review_n",
        "tagging_review_low_conf_threshold",
        "tagging_pre_review_message",
        "__last_tagging_pre_review_summary__",
    ]:
        session_state.pop(key, None)


def get_reusable_sentiment_sample(session_state) -> pd.DataFrame:
    df = session_state.get("df_sentiment_rows", pd.DataFrame())
    if isinstance(df, pd.DataFrame) and not df.empty:
        return df.copy()
    return pd.DataFrame()
