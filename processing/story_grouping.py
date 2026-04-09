# story_grouping.py


from __future__ import annotations

import math
import re
import string
from typing import List

import pandas as pd
from scipy import sparse
from scipy.sparse import csgraph
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from processing.story_examples import pick_best_story_row


def normalize_text(text: str) -> str:
    text = str(text).lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
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
    nn = NearestNeighbors(metric="cosine", radius=radius)
    nn.fit(tfidf_matrix)
    graph = nn.radius_neighbors_graph(tfidf_matrix, radius=radius, mode="connectivity")
    graph = graph + sparse.eye(graph.shape[0], format="csr")

    _, labels = csgraph.connected_components(graph, directed=False)
    df["Group ID"] = labels
    return df


def cluster_by_media_type(
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

def mark_prime_examples(grouped_df: pd.DataFrame) -> pd.DataFrame:
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

    for group_id, group in out.groupby("Group ID", dropna=False):
        best_row = pick_best_story_row(group)
        if best_row is None:
            continue
        out.loc[best_row.name, "Prime Example"] = 1

    return out


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