"""Frozen pre-optimization reference from c1c3a5614878d842151590f9fd5aadf5c02fdd9f.

Test/benchmark oracle only. Do not update this alongside the optimized function.
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher

import pandas as pd

MAX_UNDATED_FUZZY_DEDUPE_GROUP_SIZE = 250

def lengths_are_similar_enough(len_a: int, len_b: int, min_length_ratio: float = 0.70) -> bool:
    if len_a <= 0 or len_b <= 0:
        return False
    return min(len_a, len_b) / max(len_a, len_b) >= min_length_ratio


def normalize_snippet_for_compare(text: str) -> str:
    text = str(text or "")
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _leading_token_tuple(text: str, limit: int = 12) -> tuple[str, ...]:
    if not text:
        return ()
    return tuple(text.split()[:limit])


def _cheap_broadcast_similarity_gate(
    prefix_a: tuple[str, ...],
    prefix_b: tuple[str, ...],
    *,
    min_overlap_ratio: float = 0.65,
    min_shared_tokens: int = 5,
) -> bool:
    if not prefix_a or not prefix_b:
        return False

    if prefix_a == prefix_b:
        return True

    shared = len(set(prefix_a) & set(prefix_b))
    shortest = min(len(prefix_a), len(prefix_b))
    if shortest == 0:
        return False

    overlap_ratio = shared / shortest
    return shared >= min_shared_tokens and overlap_ratio >= min_overlap_ratio


def choose_best_row_index(group_df: pd.DataFrame) -> int:
    working = group_df.copy()

    if "Author" in working.columns:
        working["_author_present"] = working["Author"].notna() & working["Author"].astype(str).str.strip().ne("")
    else:
        working["_author_present"] = False

    if "Impressions" in working.columns:
        working["_impressions_num"] = pd.to_numeric(working["Impressions"], errors="coerce").fillna(0)
    else:
        working["_impressions_num"] = 0

    if "Date" in working.columns:
        working["_date_dt"] = pd.to_datetime(working["Date"], errors="coerce")
    else:
        working["_date_dt"] = pd.NaT

    working = working.sort_values(
        by=["_author_present", "_impressions_num", "_date_dt"],
        ascending=[False, False, True],
        na_position="last",
    )

    return int(working.index[0])


def dedupe_non_broadcast_by_fields(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Non-broadcast dedupe runs after media-type normalization:
    first by same normalized Type + URL, then by same normalized
    Type + Outlet + Headline.
    """
    df = df.copy()

    required_cols = ["Type", "Outlet", "Headline"]
    for col in required_cols:
        if col not in df.columns:
            return df, pd.DataFrame()

    blank_mask = (
        df["Headline"].fillna("").str.strip().eq("")
        | df["Outlet"].fillna("").str.strip().eq("")
        | df["Type"].fillna("").str.strip().eq("")
    )

    blank_set = df[blank_mask].copy()
    working = df[~blank_mask].copy()

    if working.empty:
        return df.copy(), pd.DataFrame()

    working["_dedupe_key"] = (
        working["Type"].astype("string").fillna("")
        + "||"
        + working["Outlet"].astype("string").fillna("")
        + "||"
        + working["Headline"].astype("string").fillna("")
    )

    if "Date" in working.columns:
        working["_date_dt"] = pd.to_datetime(working["Date"], errors="coerce")
    else:
        working["_date_dt"] = pd.NaT

    if "Snippet" not in working.columns:
        working["Snippet"] = ""
    working["_snippet_norm"] = working["Snippet"].map(normalize_snippet_for_compare)
    working["_snippet_prefix_tokens"] = working["_snippet_norm"].map(_leading_token_tuple)
    working["_snippet_len"] = working["_snippet_norm"].str.len()

    duplicate_indices = set()

    for _, group in working.groupby("_dedupe_key", dropna=False):
        if len(group) <= 1:
            continue

        group = group.sort_values(["_date_dt"], na_position="last", kind="mergesort").copy()

        indices = group.index.tolist()
        adjacency = {idx: set() for idx in indices}

        for _, snippet_group in group.groupby("_snippet_norm", dropna=False):
            if len(snippet_group) <= 1:
                continue
            snippet_norm = str(snippet_group["_snippet_norm"].iloc[0] or "")
            if not snippet_norm:
                continue
            snippet_indices = snippet_group.index.tolist()
            first_idx = snippet_indices[0]
            for idx in snippet_indices[1:]:
                adjacency[first_idx].add(idx)
                adjacency[idx].add(first_idx)

        for i, idx_i in enumerate(indices):
            row_i = group.loc[idx_i]
            date_i = row_i["_date_dt"]
            snippet_i = row_i["_snippet_norm"]
            prefix_i = row_i["_snippet_prefix_tokens"]

            for idx_j in indices[i + 1:]:
                row_j = group.loc[idx_j]
                date_j = row_j["_date_dt"]

                within_48h = False
                if pd.notna(date_i) and pd.notna(date_j):
                    hours_diff = abs((date_j - date_i).total_seconds()) / 3600
                    within_48h = hours_diff <= 48
                    if hours_diff > 48:
                        break

                if within_48h:
                    adjacency[idx_i].add(idx_j)
                    adjacency[idx_j].add(idx_i)
                    continue

                if pd.notna(date_i) and pd.notna(date_j):
                    continue

                if len(group) > MAX_UNDATED_FUZZY_DEDUPE_GROUP_SIZE:
                    continue

                snippet_j = row_j["_snippet_norm"]
                if not snippet_i or not snippet_j:
                    continue

                if not lengths_are_similar_enough(row_i["_snippet_len"], row_j["_snippet_len"]):
                    continue

                if not _cheap_broadcast_similarity_gate(
                    prefix_i,
                    row_j["_snippet_prefix_tokens"],
                    min_overlap_ratio=0.60,
                    min_shared_tokens=3,
                ):
                    continue

                if SequenceMatcher(None, snippet_i, snippet_j).ratio() >= 0.90:
                    adjacency[idx_i].add(idx_j)
                    adjacency[idx_j].add(idx_i)

        visited = set()
        for idx in indices:
            if idx in visited:
                continue

            stack = [idx]
            component = []

            while stack:
                node = stack.pop()
                if node in visited:
                    continue
                visited.add(node)
                component.append(node)
                stack.extend(adjacency[node] - visited)

            if len(component) <= 1:
                continue

            component_rows = group.loc[component].copy()
            keep_index = choose_best_row_index(component_rows)
            duplicate_indices.update(set(component) - {keep_index})

    dupe_cols = working.loc[list(duplicate_indices)].copy() if duplicate_indices else pd.DataFrame()
    deduped = working.drop(index=list(duplicate_indices)).copy() if duplicate_indices else working.copy()

    helper_cols = [
        "_dedupe_key",
        "_date_dt",
        "_snippet_norm",
        "_snippet_prefix_tokens",
        "_snippet_len",
    ]
    deduped.drop(columns=helper_cols, inplace=True, errors="ignore")
    dupe_cols.drop(columns=helper_cols, inplace=True, errors="ignore")

    deduped = pd.concat([deduped, blank_set], ignore_index=True)
    return deduped, dupe_cols
