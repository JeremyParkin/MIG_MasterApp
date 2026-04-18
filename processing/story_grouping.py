# story_grouping.py


from __future__ import annotations

import math
import re
import string
import time
from typing import List

import pandas as pd
from scipy import sparse
from scipy.sparse import csgraph
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

PUNCT_TRANSLATOR = str.maketrans("", "", string.punctuation)
CLUSTER_SOURCE_ID_COL = "__cluster_source_row__"

def normalize_text(text: str) -> str:
    text = str(text).lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = text.translate(PUNCT_TRANSLATOR)
    return text


def remove_extra_spaces(text: str) -> str:
    text = str(text).strip()
    return re.sub(r"\s+", " ", text)


def clean_snippet(snippet: str) -> str:
    snippet = str(snippet)
    if snippet.startswith(">>>"):
        return snippet[3:]
    if snippet.startswith(">>"):
        return snippet[2:]
    return snippet


def ensure_published_date(df: pd.DataFrame) -> pd.DataFrame:
    if "Published Date" in df.columns:
        published = pd.to_datetime(df["Published Date"], errors="coerce")
    elif "Date" in df.columns:
        published = pd.to_datetime(df["Date"], errors="coerce")
    else:
        published = pd.Series(pd.NaT, index=df.index)

    df = df.copy()
    df["Published Date"] = published.dt.strftime("%Y-%m-%d")
    return df


def ensure_published_date_fast(df: pd.DataFrame) -> pd.DataFrame:
    return ensure_published_date(df)


def preprocess_media_df_fast(media_df: pd.DataFrame) -> pd.DataFrame:
    out = media_df.copy()
    out["Headline"] = (
        out["Headline"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )
    out["Snippet"] = (
        out["Snippet"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
        .str.replace(r"^>{2,3}", "", regex=True)
    )
    out = ensure_published_date_fast(out)
    out["Normalized Headline"] = (
        out["Headline"]
        .str.lower()
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
        .str.translate(PUNCT_TRANSLATOR)
    )
    out["Normalized Snippet"] = (
        out["Snippet"]
        .str.lower()
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
        .str.translate(PUNCT_TRANSLATOR)
    )
    return out


def split_batches_by_date(media_df: pd.DataFrame, max_batch_size: int) -> List[pd.DataFrame]:
    if media_df.empty:
        return []

    ordered = media_df.sort_values(
        ["Published Date", "Headline"],
        na_position="last",
        kind="mergesort",
    ).reset_index(drop=True)

    batches: List[pd.DataFrame] = []
    current_indices: List[int] = []
    current_size = 0

    def flush_current() -> None:
        nonlocal current_indices, current_size
        if current_indices:
            batches.append(ordered.iloc[current_indices].copy())
            current_indices = []
            current_size = 0

    date_keys = ordered["Published Date"].fillna("__UNKNOWN__")

    for _, group in ordered.groupby(date_keys, sort=False):
        group_indices = group.index.to_list()

        if len(group_indices) <= max_batch_size:
            if current_size + len(group_indices) > max_batch_size:
                flush_current()
            current_indices.extend(group_indices)
            current_size += len(group_indices)
            continue

        flush_current()
        for start in range(0, len(group_indices), max_batch_size):
            chunk = group_indices[start : start + max_batch_size]
            batches.append(ordered.iloc[chunk].copy())

    flush_current()
    return batches


def cluster_similar_stories(df: pd.DataFrame, similarity_threshold: float) -> pd.DataFrame:
    df = df.copy()

    texts = (df["Normalized Headline"] + " " + df["Normalized Snippet"]).fillna("")
    if not any(text.strip() for text in texts):
        df["Group ID"] = range(len(df))
        return df

    tfidf_matrix = TfidfVectorizer().fit_transform(texts)

    if tfidf_matrix.shape[0] == 1:
        df["Group ID"] = 0
        return df

    radius = max(1.0 - similarity_threshold, 0.0)
    nn = NearestNeighbors(metric="cosine", radius=radius, algorithm="brute", n_jobs=-1)
    nn.fit(tfidf_matrix)
    graph = nn.radius_neighbors_graph(tfidf_matrix, radius=radius, mode="connectivity")
    graph = graph + sparse.eye(graph.shape[0], format="csr")

    _, labels = csgraph.connected_components(graph, directed=False)
    df["Group ID"] = labels
    return df


def cluster_by_media_type_legacy(
    df: pd.DataFrame,
    similarity_threshold: float = 0.935,
    max_batch_size: int = 1800,
) -> pd.DataFrame:
    type_column = "Media Type" if "Media Type" in df.columns else "Type"
    clustered_frames: List[pd.DataFrame] = []
    group_id_offset = 0

    for media_type in df[type_column].dropna().unique():
        media_df = df[df[type_column] == media_type].copy()
        if media_df.empty:
            continue

        media_df["Headline"] = media_df["Headline"].fillna("").apply(remove_extra_spaces)
        media_df["Snippet"] = media_df["Snippet"].fillna("").apply(remove_extra_spaces).apply(clean_snippet)
        media_df = ensure_published_date(media_df)
        media_df["Normalized Headline"] = media_df["Headline"].apply(normalize_text)
        media_df["Normalized Snippet"] = media_df["Snippet"].apply(normalize_text)

        if media_df[["Headline", "Snippet"]].apply(lambda x: x.str.strip()).eq("").all(axis=None):
            continue

        batches = split_batches_by_date(media_df, max_batch_size=max_batch_size)

        for batch in batches:
            if len(batch) == 1:
                batch["Group ID"] = group_id_offset
                group_id_offset += 1
            else:
                clustered_batch = cluster_similar_stories(batch, similarity_threshold)
                clustered_batch["Group ID"] += group_id_offset
                group_id_offset += clustered_batch["Group ID"].max() + 1
                batch = clustered_batch

            batch = batch.drop(columns=["Normalized Headline", "Normalized Snippet"], errors="ignore")
            clustered_frames.append(batch)

    if not clustered_frames:
        out = df.copy()
        out["Group ID"] = range(len(out))
        return out

    return pd.concat(clustered_frames, ignore_index=True)


def _canonical_cluster_signature(df: pd.DataFrame) -> list[tuple[int, ...]]:
    if df.empty or "Group ID" not in df.columns or CLUSTER_SOURCE_ID_COL not in df.columns:
        return []
    grouped = (
        df.groupby("Group ID", dropna=False)[CLUSTER_SOURCE_ID_COL]
        .apply(lambda s: tuple(sorted(int(x) for x in s.tolist())))
        .tolist()
    )
    return sorted(grouped)


def cluster_by_media_type(
    df: pd.DataFrame,
    similarity_threshold: float = 0.935,
    max_batch_size: int = 1800,
) -> pd.DataFrame:
    type_column = "Media Type" if "Media Type" in df.columns else "Type"
    working_df = df.copy()
    if CLUSTER_SOURCE_ID_COL not in working_df.columns:
        working_df[CLUSTER_SOURCE_ID_COL] = range(len(working_df))

    clustered_frames: List[pd.DataFrame] = []
    group_id_offset = 0

    for media_type in working_df[type_column].dropna().unique():
        media_df = working_df[working_df[type_column] == media_type].copy()
        if media_df.empty:
            continue

        media_df = preprocess_media_df_fast(media_df)

        if media_df[["Headline", "Snippet"]].apply(lambda x: x.str.strip()).eq("").all(axis=None):
            continue

        batches = split_batches_by_date(media_df, max_batch_size=max_batch_size)

        for batch in batches:
            if len(batch) == 1:
                batch["Group ID"] = group_id_offset
                group_id_offset += 1
            else:
                clustered_batch = cluster_similar_stories(batch, similarity_threshold)
                batch_group_count = int(clustered_batch["Group ID"].nunique(dropna=False))
                clustered_batch["Group ID"] += group_id_offset
                group_id_offset += batch_group_count
                batch = clustered_batch

            batch = batch.drop(columns=["Normalized Headline", "Normalized Snippet"], errors="ignore")
            clustered_frames.append(batch)

    if not clustered_frames:
        out = working_df.copy()
        out["Group ID"] = range(len(out))
        return out.drop(columns=[CLUSTER_SOURCE_ID_COL], errors="ignore")

    out = pd.concat(clustered_frames, ignore_index=True)
    return out.drop(columns=[CLUSTER_SOURCE_ID_COL], errors="ignore")


def cluster_by_media_type_with_timings(
    df: pd.DataFrame,
    similarity_threshold: float = 0.935,
    max_batch_size: int = 1800,
    validate: bool = False,
) -> tuple[pd.DataFrame, list[dict[str, float | str]], dict[str, object]]:
    timings: list[dict[str, float | str]] = []
    validation: dict[str, object] = {"Enabled": validate}

    def record_timing(step_name: str, start_time: float) -> None:
        timings.append({"Step": step_name, "Seconds": round(time.perf_counter() - start_time, 4)})

    if df.empty:
        out = df.copy()
        out["Group ID"] = range(len(out))
        validation["Matches Legacy"] = True
        return out, timings, validation

    working = df.copy()
    working[CLUSTER_SOURCE_ID_COL] = range(len(working))

    start_time = time.perf_counter()
    optimized = cluster_by_media_type(
        working,
        similarity_threshold=similarity_threshold,
        max_batch_size=max_batch_size,
    )
    record_timing("Cluster optimized path", start_time)

    if validate:
        legacy_start = time.perf_counter()
        legacy = cluster_by_media_type_legacy(
            working,
            similarity_threshold=similarity_threshold,
            max_batch_size=max_batch_size,
        )
        validation["Legacy Seconds"] = round(time.perf_counter() - legacy_start, 4)

        optimized_with_ids = optimized.copy()
        legacy_with_ids = legacy.copy()
        if CLUSTER_SOURCE_ID_COL not in optimized_with_ids.columns:
            optimized_with_ids[CLUSTER_SOURCE_ID_COL] = working[CLUSTER_SOURCE_ID_COL].values
        if CLUSTER_SOURCE_ID_COL not in legacy_with_ids.columns:
            legacy_with_ids[CLUSTER_SOURCE_ID_COL] = working[CLUSTER_SOURCE_ID_COL].values

        validation["Matches Legacy"] = _canonical_cluster_signature(optimized_with_ids) == _canonical_cluster_signature(legacy_with_ids)
        if not validation["Matches Legacy"]:
            validation["Optimized Groups"] = int(optimized_with_ids["Group ID"].nunique(dropna=False))
            validation["Legacy Groups"] = int(legacy_with_ids["Group ID"].nunique(dropna=False))
    else:
        validation["Matches Legacy"] = None

    return optimized, timings, validation

def mark_prime_examples_legacy(grouped_df: pd.DataFrame) -> pd.DataFrame:
    """
    Mark exactly one row per Group ID as Prime Example = 1, all others 0.
    """
    if grouped_df.empty or "Group ID" not in grouped_df.columns:
        out = grouped_df.copy()
        if "Prime Example" not in out.columns:
            out["Prime Example"] = 0
        return out

    out = grouped_df.copy()
    out["Prime Example"] = 0

    working = out.copy()

    for col in ["Outlet", "Type", "Coverage Flags"]:
        if col not in working.columns:
            working[col] = ""
        working[col] = working[col].fillna("").astype(str)

    if "Impressions" not in working.columns:
        working["Impressions"] = 0
    working["Impressions"] = pd.to_numeric(working["Impressions"], errors="coerce").fillna(0)

    preferred_wire_pattern = r"Reuters|Associated Press|Canadian Press"
    working["_is_preferred_wire"] = working["Outlet"].str.contains(
        preferred_wire_pattern,
        case=False,
        na=False,
        regex=True,
    )

    flags = working["Coverage Flags"].str.strip()
    working["_quality_rank"] = 2
    working.loc[flags.eq("Advertorial"), "_quality_rank"] = 7
    working.loc[flags.eq("Market Report Spam"), "_quality_rank"] = 6
    working.loc[flags.eq("Press Release"), "_quality_rank"] = 5
    working.loc[flags.eq("Financial Outlet"), "_quality_rank"] = 4
    working.loc[flags.eq("Aggregator"), "_quality_rank"] = 3
    working.loc[flags.eq(""), "_quality_rank"] = 2
    working.loc[flags.eq("Good Outlet"), "_quality_rank"] = 1
    working.loc[working["_is_preferred_wire"], "_quality_rank"] = 0

    if "Date" in working.columns:
        working["_date_dt"] = pd.to_datetime(working["Date"], errors="coerce")
    else:
        working["_date_dt"] = pd.NaT

    for group_id, group in working.groupby("Group ID", dropna=False):
        best_index = (
            group.sort_values(
                by=["_quality_rank", "Impressions", "_date_dt"],
                ascending=[True, False, True],
                na_position="last",
            )
            .index[0]
        )
        out.loc[best_index, "Prime Example"] = 1

    return out


def mark_prime_examples(grouped_df: pd.DataFrame) -> pd.DataFrame:
    if grouped_df.empty or "Group ID" not in grouped_df.columns:
        out = grouped_df.copy()
        if "Prime Example" not in out.columns:
            out["Prime Example"] = 0
        return out

    working = grouped_df.copy()
    working["Prime Example"] = 0

    for col in ["Outlet", "Type", "Coverage Flags"]:
        if col not in working.columns:
            working[col] = ""
        working[col] = working[col].fillna("").astype(str)

    if "Impressions" not in working.columns:
        working["Impressions"] = 0
    working["Impressions"] = pd.to_numeric(working["Impressions"], errors="coerce").fillna(0)

    preferred_wire_pattern = r"Reuters|Associated Press|Canadian Press"
    working["_quality_rank"] = 2
    flags = working["Coverage Flags"].str.strip()
    working.loc[flags.eq("Advertorial"), "_quality_rank"] = 7
    working.loc[flags.eq("Market Report Spam"), "_quality_rank"] = 6
    working.loc[flags.eq("Press Release"), "_quality_rank"] = 5
    working.loc[flags.eq("Financial Outlet"), "_quality_rank"] = 4
    working.loc[flags.eq("Aggregator"), "_quality_rank"] = 3
    working.loc[flags.eq(""), "_quality_rank"] = 2
    working.loc[flags.eq("Good Outlet"), "_quality_rank"] = 1
    working.loc[
        working["Outlet"].str.contains(preferred_wire_pattern, case=False, na=False, regex=True),
        "_quality_rank",
    ] = 0

    if "Date" in working.columns:
        working["_date_dt"] = pd.to_datetime(working["Date"], errors="coerce")
    else:
        working["_date_dt"] = pd.NaT

    working["_impressions_sort"] = -working["Impressions"]
    ordered = working.sort_values(
        by=["Group ID", "_quality_rank", "_impressions_sort", "_date_dt"],
        ascending=[True, True, True, True],
        na_position="last",
        kind="mergesort",
    )
    best_indices = ordered.groupby("Group ID", sort=False).head(1).index
    working.loc[best_indices, "Prime Example"] = 1

    return working.drop(columns=["_quality_rank", "_date_dt", "_impressions_sort"], errors="ignore")


def mark_prime_examples_with_timings(
    grouped_df: pd.DataFrame,
    validate: bool = False,
) -> tuple[pd.DataFrame, list[dict[str, float | str]], dict[str, object]]:
    timings: list[dict[str, float | str]] = []
    validation: dict[str, object] = {"Enabled": validate}

    def record_timing(step_name: str, start_time: float) -> None:
        timings.append({
            "Step": step_name,
            "Seconds": round(time.perf_counter() - start_time, 4),
        })

    step_start = time.perf_counter()
    if grouped_df.empty or "Group ID" not in grouped_df.columns:
        out = grouped_df.copy()
        if "Prime Example" not in out.columns:
            out["Prime Example"] = 0
        record_timing("Handle empty/missing Group ID", step_start)
        validation["Matches Legacy"] = True
        return out, timings, validation

    step_start = time.perf_counter()
    out = grouped_df.copy()
    out["Prime Example"] = 0
    record_timing("Initialize prime example column", step_start)

    step_start = time.perf_counter()
    materialized_groups = out["Group ID"].nunique(dropna=False)
    record_timing("Materialize groups", step_start)
    validation["Group Count"] = int(materialized_groups)

    step_start = time.perf_counter()
    optimized = mark_prime_examples(grouped_df)
    record_timing("Select best row per group", step_start)

    if validate:
        legacy_start = time.perf_counter()
        legacy = mark_prime_examples_legacy(grouped_df)
        validation["Legacy Seconds"] = round(time.perf_counter() - legacy_start, 4)

        legacy_flags = legacy.get("Prime Example", pd.Series(index=legacy.index, dtype="int64")).fillna(0).astype(int)
        optimized_flags = optimized.get("Prime Example", pd.Series(index=optimized.index, dtype="int64")).fillna(0).astype(int)
        validation["Matches Legacy"] = bool(legacy_flags.equals(optimized_flags))
        if not validation["Matches Legacy"]:
            validation["Mismatch Count"] = int((legacy_flags != optimized_flags).sum())
    else:
        validation["Matches Legacy"] = None

    return optimized, timings, validation


def build_unique_story_table_from_prime(grouped_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build one-row-per-group table using the Prime Example row as the representative row,
    while summing group-level metrics like mentions and impressions.
    """
    if grouped_df.empty or "Group ID" not in grouped_df.columns:
        return pd.DataFrame()

    working = grouped_df.copy()

    if "Prime Example" not in working.columns:
        working = mark_prime_examples(working)

    group_counts = working.groupby("Group ID").size().reset_index(name="Group Count")

    metric_frames = [group_counts]

    if "Mentions" in working.columns:
        mentions = (
            pd.to_numeric(working["Mentions"], errors="coerce")
            .fillna(0)
            .groupby(working["Group ID"])
            .sum()
            .reset_index(name="Mentions")
        )
        metric_frames.append(mentions)

    if "Impressions" in working.columns:
        impressions = (
            pd.to_numeric(working["Impressions"], errors="coerce")
            .fillna(0)
            .groupby(working["Group ID"])
            .sum()
            .reset_index(name="Impressions")
        )
        metric_frames.append(impressions)

    if "Effective Reach" in working.columns:
        er = (
            pd.to_numeric(working["Effective Reach"], errors="coerce")
            .fillna(0)
            .groupby(working["Group ID"])
            .sum()
            .reset_index(name="Effective Reach")
        )
        metric_frames.append(er)

    metrics = metric_frames[0]
    for frame in metric_frames[1:]:
        metrics = metrics.merge(frame, on="Group ID", how="left")

    prime_rows = working.loc[working["Prime Example"] == 1].copy()

    # Keep one prime row per group just in case
    prime_rows = prime_rows.drop_duplicates(subset=["Group ID"], keep="first")

    # Drop metric columns from representative rows so the summed versions win
    prime_rows = prime_rows.drop(
        columns=["Mentions", "Impressions", "Effective Reach", "Group Count"],
        errors="ignore",
    )

    unique_stories = prime_rows.merge(metrics, on="Group ID", how="left")

    # Sort most important groups first
    sort_cols = [c for c in ["Group Count", "Impressions", "Effective Reach", "Mentions"] if c in unique_stories.columns]
    if sort_cols:
        unique_stories = unique_stories.sort_values(by=sort_cols, ascending=False)

    return unique_stories.reset_index(drop=True)


def build_unique_story_table(grouped_df: pd.DataFrame) -> pd.DataFrame:
    return build_unique_story_table_from_prime(grouped_df)
