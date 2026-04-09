# basic_cleaning_pipeline.py

from __future__ import annotations

import pandas as pd

from processing.effective_reach import (
    apply_effective_reach_social,
    apply_effective_reach_traditional,
)

from processing.story_grouping import (
    cluster_by_media_type,
    build_unique_story_table,
    mark_prime_examples,
)
from processing.standard_cleaning import run_standard_cleaning


def run_basic_cleaning_pipeline(
    df: pd.DataFrame,
    merge_online: bool = True,
    drop_dupes: bool = True,
    add_coverage_flags: bool = True,
    similarity_threshold: float = 0.935,
    max_batch_size: int = 1800,
) -> dict:
    cleaning_results = run_standard_cleaning(
        df=df,
        merge_online=merge_online,
        drop_dupes=drop_dupes,
        add_flags=add_coverage_flags,
    )

    df_traditional = apply_effective_reach_traditional(cleaning_results["df_traditional"])
    df_social = apply_effective_reach_social(cleaning_results["df_social"])
    df_dupes = cleaning_results["df_dupes"]


    df_ai_grouped = cluster_by_media_type(
        df_traditional.copy(),
        similarity_threshold=similarity_threshold,
        max_batch_size=max_batch_size,
    )
    df_ai_grouped = mark_prime_examples(df_ai_grouped)
    df_ai_unique = build_unique_story_table(df_ai_grouped)

    return {
        "df_traditional": df_traditional,
        "df_social": df_social,
        "df_dupes": df_dupes,
        "df_ai_grouped": df_ai_grouped,
        "df_ai_unique": df_ai_unique,
    }