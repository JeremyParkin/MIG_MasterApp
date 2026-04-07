from __future__ import annotations

import math
from typing import Literal

import pandas as pd


SampleMode = Literal["full", "representative", "custom"]


DEFAULT_MAX_FULL_ROWS = 2000


def init_tagging_config_state(session_state) -> None:
    defaults = {
        "tagging_config_step": False,
        "tagging_sample_mode": "representative",
        "tagging_sample_size": None,
        "tagging_full_override": False,
        "df_tagging_rows": pd.DataFrame(),
        "df_tagging_grouped_rows": pd.DataFrame(),
        "df_tagging_unique": pd.DataFrame(),
        "tagging_elapsed_time": 0.0,
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
    Build one-row-per-group table from sampled rows using the EXISTING canonical Group ID.
    This does not recluster.
    """
    if df_rows is None or df_rows.empty:
        return pd.DataFrame()

    if "Group ID" not in df_rows.columns:
        raise ValueError("Group ID not found. Basic Cleaning must assign canonical groups before tagging.")

    working = df_rows.copy()

    agg_dict = {}

    # First non-null / first row style fields
    first_pref_cols = [
        "Headline",
        "Date",
        "Outlet",
        "Example Outlet",
        "URL",
        "Example URL",
        "Snippet",
        "Example Snippet",
        "Type",
        "Language",
        "Country",
        "Prov/State",
    ]
    for col in first_pref_cols:
        if col in working.columns:
            agg_dict[col] = "first"

    # Sum metrics
    if "Mentions" in working.columns:
        agg_dict["Mentions"] = "sum"
    if "Impressions" in working.columns:
        agg_dict["Impressions"] = "sum"
    if "Effective Reach" in working.columns:
        agg_dict["Effective Reach"] = "sum"

    unique_rows = (
        working.groupby("Group ID", as_index=False)
        .agg(agg_dict)
        .copy()
    )

    # Ensure exemplar fields exist
    if "Example Outlet" not in unique_rows.columns and "Outlet" in unique_rows.columns:
        unique_rows["Example Outlet"] = unique_rows["Outlet"]

    if "Example URL" not in unique_rows.columns and "URL" in unique_rows.columns:
        unique_rows["Example URL"] = unique_rows["URL"]

    if "Example Snippet" not in unique_rows.columns and "Snippet" in unique_rows.columns:
        unique_rows["Example Snippet"] = unique_rows["Snippet"]

    if "Group Count" not in unique_rows.columns:
        group_counts = working.groupby("Group ID").size().reset_index(name="Group Count")
        unique_rows = unique_rows.merge(group_counts, on="Group ID", how="left")

    return unique_rows.reset_index(drop=True)


def prepare_tagging_datasets(
    df_traditional: pd.DataFrame,
    sample_mode: SampleMode,
    custom_sample_size: int | None = None,
    max_full_rows: int = DEFAULT_MAX_FULL_ROWS,
    full_override: bool = False,
    random_state: int = 1,
) -> dict:
    """
    Build tagging-specific datasets:
    - df_tagging_rows: sampled row-level rows with original Group ID
    - df_tagging_grouped_rows: same as sampled rows (kept for naming consistency)
    - df_tagging_unique: one row per original Group ID represented in the sample
    """
    source_rows = get_tagging_source_rows(df_traditional)
    sampled_rows, effective_sample_size = sample_tagging_rows(
        source_rows,
        sample_mode=sample_mode,
        custom_sample_size=custom_sample_size,
        max_full_rows=max_full_rows,
        full_override=full_override,
        random_state=random_state,
    )

    if not sampled_rows.empty and "Group ID" not in sampled_rows.columns:
        raise ValueError("Sampled tagging rows do not contain Group ID. Standard Cleaning must run first.")

    grouped_rows = sampled_rows.copy()
    unique_rows = build_unique_story_table_from_existing_groups(grouped_rows)

    if "Tag_Processed" not in unique_rows.columns:
        unique_rows["Tag_Processed"] = False

    for col in ["AI Tag", "AI Tags", "AI Tag Rationale"]:
        if col not in unique_rows.columns:
            unique_rows[col] = ""

    return {
        "df_tagging_rows": sampled_rows.reset_index(drop=True),
        "df_tagging_grouped_rows": grouped_rows.reset_index(drop=True),
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
    session_state.df_tagging_grouped_rows = pd.DataFrame()
    session_state.df_tagging_unique = pd.DataFrame()
    session_state.tagging_elapsed_time = 0.0