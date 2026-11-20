from __future__ import annotations

import json

import pandas as pd

from processing.analysis_context import (
    apply_session_coverage_flag_policy,
    build_dataset_scope_cache_key,
    get_analysis_context_payload,
    get_qualitative_coverage_flag_exclusions,
    get_qualitative_coverage_keep_keys,
)
from processing.rapid_insights import build_analysis_context_fingerprint
from processing.sentiment_config import get_sentiment_source_rows
from processing.story_grouping import mark_prime_examples


def rapid_scope_signature(session_state) -> tuple:
    start, end, media, flags, keep = build_dataset_scope_cache_key(session_state)
    return (
        start, end, tuple(sorted(set(media))), tuple(sorted(set(flags))), keep,
        tuple(sorted(set(get_qualitative_coverage_flag_exclusions(session_state)))),
        tuple(sorted(get_qualitative_coverage_keep_keys(session_state))),
    )


def scoped_rapid_source_rows(session_state) -> pd.DataFrame:
    return apply_session_coverage_flag_policy(
        get_sentiment_source_rows(session_state.get("df_traditional")),
        session_state,
        get_qualitative_coverage_flag_exclusions(session_state),
    )


def legacy_rapid_scope_changed(prepared_context: str, current_context: str) -> bool:
    """Older Rapid snapshots may still record scope in their Analysis Context fingerprint."""
    try:
        prepared = json.loads(prepared_context)
        current = json.loads(current_context)
    except ValueError:
        return False
    if not isinstance(prepared, dict) or not isinstance(current, dict):
        return False
    for field in ("dataset_start_date", "dataset_end_date"):
        if field in prepared and prepared[field] != current.get(field):
            return True
    for field in (
        "dataset_media_types", "dataset_excluded_flags", "dataset_exclusion_keep_keys",
        "qualitative_excluded_flags", "qualitative_exclusion_keep_keys",
    ):
        if field in prepared and set(prepared[field] or []) != set(current.get(field) or []):
            return True
    return False


def _article_keys(rows: pd.DataFrame) -> pd.MultiIndex:
    # Group ID alone cannot distinguish an excluded prime from an eligible syndicated copy.
    fields = ["Group ID", "URL", "Headline", "Outlet", "Type", "Date", "Coverage Flags", "Story Family Flags"]
    keys = rows.reindex(columns=fields).astype("string").fillna("")
    return pd.MultiIndex.from_frame(keys)


def rapid_sample_rows_in_scope(canonical_rows: pd.DataFrame, sample_rows: pd.DataFrame) -> pd.Series:
    return pd.Series(_article_keys(sample_rows).isin(_article_keys(canonical_rows)), index=sample_rows.index)


def rapid_sample_matches_scope(
    session_state,
    *,
    current_context: str | None = None,
    source_rows: pd.DataFrame | None = None,
) -> bool:
    prepared_scope = session_state.get("rapid_sample_scope_signature")
    if prepared_scope is not None:
        if prepared_scope != rapid_scope_signature(session_state):
            return False
    else:
        prepared_context = session_state.get("jev_analysis_context_fingerprint")
        if prepared_context:
            current_context = current_context or build_analysis_context_fingerprint(
                get_analysis_context_payload(session_state)
            )
            if legacy_rapid_scope_changed(prepared_context, current_context):
                return False

    saved_frames = [
        session_state.get(key)
        for key in ("df_jev_sentiment_rows", "df_jev_sentiment_grouped_rows", "df_jev_sentiment_unique")
    ]
    saved_frames = [frame for frame in saved_frames if isinstance(frame, pd.DataFrame) and not frame.empty]
    traditional = session_state.get("df_traditional")
    if not saved_frames or not isinstance(traditional, pd.DataFrame):
        return True
    scoped = scoped_rapid_source_rows(session_state) if source_rows is None else source_rows
    return len(scoped) >= len(traditional) or all(
        rapid_sample_rows_in_scope(scoped, frame).all() for frame in saved_frames
    )


def scoped_rapid_reuse_rows(source_rows: pd.DataFrame, previous_sample: pd.DataFrame) -> pd.DataFrame:
    """Reuse membership while taking text and grouping metadata from eligible canonical rows."""
    return source_rows.loc[rapid_sample_rows_in_scope(previous_sample, source_rows)].reset_index(drop=True)


def ensure_scoped_rapid_primes(source_rows: pd.DataFrame) -> pd.DataFrame:
    """Keep canonical primes where eligible and choose an eligible prime where one was excluded."""
    if source_rows.empty:
        return source_rows
    prime_mask = source_rows["Prime Example"].eq(1)
    with_prime = source_rows.loc[prime_mask, "Group ID"]
    needs_prime = ~source_rows["Group ID"].isin(with_prime)
    if not needs_prime.any():
        return source_rows
    out = source_rows.copy()
    out.loc[needs_prime, "Prime Example"] = mark_prime_examples(
        source_rows.loc[needs_prime]
    )["Prime Example"]
    return out
