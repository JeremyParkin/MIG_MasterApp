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


def build_unique_story_table(grouped_df: pd.DataFrame) -> pd.DataFrame:
    if grouped_df.empty or "Group ID" not in grouped_df.columns:
        return pd.DataFrame()

    group_counts = grouped_df.groupby("Group ID").size().reset_index(name="Group Count")
    unique_stories = grouped_df.groupby("Group ID").agg(lambda x: x.iloc[0]).reset_index()
    unique_stories = unique_stories.merge(group_counts, on="Group ID")
    unique_stories = unique_stories.sort_values(by="Group Count", ascending=False).reset_index(drop=True)
    return unique_stories