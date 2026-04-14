# standard_cleaning.py

from __future__ import annotations

import re
import time
from difflib import SequenceMatcher

import numpy as np
import pandas as pd
from titlecase import titlecase
from processing.coverage_flags import add_coverage_flags

SOCIAL_TYPES = ["FACEBOOK", "TWITTER", "X", "INSTAGRAM", "REDDIT", "YOUTUBE", "TIKTOK", "LINKEDIN", "BLUESKY"]
BROADCAST_TYPES = ["RADIO", "TV"]
SOURCE_ROW_COL = "__source_row__"


def lengths_are_similar_enough(len_a: int, len_b: int, min_length_ratio: float = 0.70) -> bool:
    if len_a <= 0 or len_b <= 0:
        return False
    return min(len_a, len_b) / max(len_a, len_b) >= min_length_ratio


def normalize_snippet_for_compare(text: str) -> str:
    text = str(text or "")
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def snippet_similarity(a: str, b: str) -> float:
    a_norm = normalize_snippet_for_compare(a)
    b_norm = normalize_snippet_for_compare(b)

    if not a_norm and not b_norm:
        return 1.0
    if not a_norm or not b_norm:
        return 0.0

    if a_norm == b_norm:
        return 1.0

    if not lengths_are_similar_enough(len(a_norm), len(b_norm)):
        return 0.0

    return SequenceMatcher(None, a_norm, b_norm).ratio()


def prepare_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for column in ["Headline", "Snippet", "Outlet", "Author", "URL"]:
        if column in df.columns:
            df[column] = df[column].fillna("").astype(str)

    if "Author" in df.columns:
        df["Author"] = df["Author"].replace("", np.nan)

    return df


def standardize_media_types(df: pd.DataFrame, merge_online: bool = True) -> pd.DataFrame:
    df = df.copy()

    if "Type" not in df.columns:
        return df

    df["Type"] = df["Type"].fillna("").astype(str)

    df["Type"] = df["Type"].replace({
        "ONLINE_NEWS": "ONLINE NEWS",
        "PRESS_RELEASE": "PRESS RELEASE",
    })

    if "URL" in df.columns:
        url_series = df["URL"].fillna("").astype(str)

        df.loc[url_series.str.contains("www.facebook.com", na=False), "Type"] = "FACEBOOK"
        df.loc[url_series.str.contains("twitter.com", na=False), "Type"] = "X"
        df.loc[url_series.str.match(r"^https?://(www\.)?x\.com(/|$)", na=False), "Type"] = "X"
        df.loc[url_series.str.contains("tiktok.com", na=False), "Type"] = "TIKTOK"
        df.loc[url_series.str.contains("www.instagram.com", na=False), "Type"] = "INSTAGRAM"
        df.loc[url_series.str.contains("reddit.com", na=False), "Type"] = "REDDIT"
        df.loc[url_series.str.contains("youtube.com", na=False), "Type"] = "YOUTUBE"
        df.loc[url_series.str.contains("linkedin.com", na=False), "Type"] = "LINKEDIN"
        df.loc[url_series.str.contains("bsky.app", na=False), "Type"] = "BLUESKY"

    if "Outlet" in df.columns and "URL" in df.columns:
        df.loc[
            (df["Outlet"].str.lower() == "report on business")
            & (df["URL"].str.contains("globeandmail", na=False)),
            "Outlet",
        ] = "The Globe and Mail"

    if merge_online:
        df["Type"] = df["Type"].replace({
            "ONLINE NEWS": "ONLINE",
            "PRESS RELEASE": "ONLINE",
            "BLOGS": "ONLINE",
        })

    if "Original URL" in df.columns:
        df.loc[df["Original URL"].notnull(), "URL"] = df["Original URL"]
        df.drop(columns=["Original URL"], inplace=True, errors="ignore")

    return df


def reorder_key_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "Impressions" in df.columns:
        temp = df.pop("Impressions")
        insert_at = min(4, len(df.columns))
        df.insert(insert_at, "Impressions", temp)

    if "Mentions" in df.columns:
        temp = df.pop("Mentions")
        insert_at = min(4, len(df.columns))
        df.insert(insert_at, "Mentions", temp)

    return df


def clean_text_fields(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    strip_columns = ["Headline", "Outlet", "Author", "Snippet"]

    for column in strip_columns:
        if column not in df.columns:
            continue

        df[column] = (
            df[column]
            .fillna("")
            .astype(str)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
            .str.replace("& amp;", "&", regex=False)
        )

    if "Outlet" in df.columns:
        df["Outlet"] = df["Outlet"].str.replace(r" \(Online\)", "", regex=True)

    if "Headline" in df.columns:
        df["Headline"] = df["Headline"].str.replace("\u2018", "'", regex=False)
        df["Headline"] = df["Headline"].str.replace("\u2019", "'", regex=False)
        df["Headline"] = df["Headline"].fillna("").map(titlecase)

    return df


def split_social(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()

    if "Type" not in df.columns:
        return df.copy(), pd.DataFrame()

    df_social = df.loc[df["Type"].isin(SOCIAL_TYPES)].copy()
    df_traditional = df.loc[~df["Type"].isin(SOCIAL_TYPES)].copy()

    return df_traditional, df_social


def split_broadcast(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()

    if "Type" not in df.columns:
        return df.copy(), pd.DataFrame()

    broadcast_df = df.loc[df["Type"].isin(BROADCAST_TYPES)].copy()
    non_broadcast_df = df.loc[~df["Type"].isin(BROADCAST_TYPES)].copy()

    return non_broadcast_df, broadcast_df


def dedupe_by_url(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()

    if "URL" not in df.columns:
        return df, pd.DataFrame()

    url_series = df["URL"].replace("", np.nan)
    blank_urls = df[url_series.isna()].copy()
    working = df[~url_series.isna()].copy()

    if working.empty:
        return df.copy(), pd.DataFrame()

    working["URL_Helper"] = (
        working["URL"]
        .fillna("")
        .astype(str)
        .str.lower()
        .str.replace("http:", "https:", regex=False)
    )

    sort_cols = [c for c in ["URL_Helper", "Author", "Impressions", "Date"] if c in working.columns]
    if sort_cols:
        ascending = []
        for col in sort_cols:
            if col == "Impressions":
                ascending.append(False)
            else:
                ascending.append(True)
        working = working.sort_values(sort_cols, ascending=ascending)

    dupe_urls = working[working["URL_Helper"].duplicated(keep="first")].copy()
    deduped = working[~working["URL_Helper"].duplicated(keep="first")].copy()

    deduped.drop(columns=["URL_Helper"], inplace=True, errors="ignore")
    dupe_urls.drop(columns=["URL_Helper"], inplace=True, errors="ignore")

    deduped = pd.concat([deduped, blank_urls], ignore_index=True)
    return deduped, dupe_urls


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
    Candidate duplicates are rows with same Type + Outlet + Headline.
    Confirm duplication if either:
    - dates are within 48 hours, or
    - snippets are >= 90% similar.
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

    duplicate_indices = set()

    for _, group in working.groupby("_dedupe_key", dropna=False):
        if len(group) <= 1:
            continue

        group = group.copy()

        indices = group.index.tolist()
        adjacency = {idx: set() for idx in indices}

        for i, idx_i in enumerate(indices):
            row_i = group.loc[idx_i]
            date_i = row_i["_date_dt"]
            snippet_i = row_i.get("Snippet", "")

            for idx_j in indices[i + 1:]:
                row_j = group.loc[idx_j]
                date_j = row_j["_date_dt"]
                snippet_j = row_j.get("Snippet", "")

                within_48h = False
                if pd.notna(date_i) and pd.notna(date_j):
                    hours_diff = abs((date_j - date_i).total_seconds()) / 3600
                    within_48h = hours_diff <= 48

                sim = snippet_similarity(snippet_i, snippet_j)
                snippet_match = sim >= 0.90

                if within_48h or snippet_match:
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

    deduped.drop(columns=["_dedupe_key", "_date_dt"], inplace=True, errors="ignore")
    dupe_cols.drop(columns=["_dedupe_key", "_date_dt"], inplace=True, errors="ignore")

    deduped = pd.concat([deduped, blank_set], ignore_index=True)
    return deduped, dupe_cols


def dedupe_broadcast_legacy(broadcast_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    broadcast_df = broadcast_df.copy()

    if broadcast_df.empty:
        return broadcast_df.copy(), pd.DataFrame()

    working = broadcast_df.reset_index(drop=True).copy()
    working["_original_order"] = np.arange(len(working))
    working["_date_time"] = pd.to_datetime(working["Date"], errors="coerce") if "Date" in working.columns else pd.NaT
    working["_date_only"] = working["_date_time"].dt.date

    snippet_col = "Snippet" if "Snippet" in working.columns else None
    if snippet_col is None:
        working["_snippet_text"] = ""
    else:
        working["_snippet_text"] = working[snippet_col].fillna("").astype(str)

    working["_snippet_norm"] = (
        working["_snippet_text"]
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    working["_snippet_len"] = working["_snippet_text"].str.len()

    duplicate_indexes = set()

    group_cols = [c for c in ["Outlet", "Type", "_date_only"] if c in working.columns]
    if not group_cols:
        return broadcast_df.copy(), pd.DataFrame()

    for _, group in working.groupby(group_cols, dropna=False):
        valid_group = group[group["_date_time"].notna()].sort_values(["_date_time", "_original_order"])
        if len(valid_group) < 2:
            continue

        group_indices = valid_group.index.tolist()
        adjacency = {idx: set() for idx in group_indices}

        for i, idx_i in enumerate(group_indices):
            row_i = valid_group.loc[idx_i]
            snippet_i = row_i["_snippet_norm"]
            time_i = row_i["_date_time"]

            for idx_j in group_indices[i + 1:]:
                row_j = valid_group.loc[idx_j]
                time_j = row_j["_date_time"]
                seconds_diff = abs((time_j - time_i).total_seconds())

                if seconds_diff > 60:
                    break

                snippet_j = row_j["_snippet_norm"]
                snippets_match = snippet_i == snippet_j

                if not snippets_match and lengths_are_similar_enough(
                    row_i["_snippet_len"], row_j["_snippet_len"]
                ):
                    snippets_match = SequenceMatcher(None, snippet_i, snippet_j).ratio() >= 0.90

                if snippets_match:
                    adjacency[idx_i].add(idx_j)
                    adjacency[idx_j].add(idx_i)

        visited = set()
        for idx in group_indices:
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

            component_rows = working.loc[component].copy()
            component_rows = component_rows.sort_values(
                ["_snippet_len", "_date_time", "_original_order"],
                ascending=[False, True, True],
            )
            keep_index = component_rows.index[0]
            duplicate_indexes.update(set(component) - {keep_index})

    if duplicate_indexes:
        broadcast_dupes = working.loc[list(duplicate_indexes)].copy()
        cleaned_broadcast = working.drop(index=list(duplicate_indexes)).copy()
    else:
        broadcast_dupes = pd.DataFrame()
        cleaned_broadcast = working.copy()

    helper_cols = [
        "_original_order",
        "_date_time",
        "_date_only",
        "_snippet_text",
        "_snippet_norm",
        "_snippet_len",
    ]
    cleaned_broadcast.drop(columns=helper_cols, inplace=True, errors="ignore")
    broadcast_dupes.drop(columns=helper_cols, inplace=True, errors="ignore")

    return cleaned_broadcast, broadcast_dupes


def _cluster_time_proximate_indices(sorted_group: pd.DataFrame, time_col: str) -> list[list[int]]:
    clusters: list[list[int]] = []
    current_cluster: list[int] = []
    last_time = None

    for idx, row in sorted_group.iterrows():
        current_time = row[time_col]
        if pd.isna(current_time):
            if current_cluster:
                clusters.append(current_cluster)
                current_cluster = []
            clusters.append([idx])
            last_time = None
            continue

        if not current_cluster:
            current_cluster = [idx]
            last_time = current_time
            continue

        gap_seconds = abs((current_time - last_time).total_seconds()) if last_time is not None else 0
        if gap_seconds <= 60:
            current_cluster.append(idx)
        else:
            clusters.append(current_cluster)
            current_cluster = [idx]
        last_time = current_time

    if current_cluster:
        clusters.append(current_cluster)

    return clusters


def dedupe_broadcast(
    broadcast_df: pd.DataFrame,
    return_mapping: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame] | tuple[pd.DataFrame, pd.DataFrame, list[tuple[int, int]]]:
    broadcast_df = broadcast_df.copy()

    if broadcast_df.empty:
        empty_result: tuple[pd.DataFrame, pd.DataFrame] = (broadcast_df.copy(), pd.DataFrame())
        if return_mapping:
            return empty_result[0], empty_result[1], []
        return empty_result

    working = broadcast_df.reset_index(drop=True).copy()
    working["_original_order"] = np.arange(len(working))
    working["_date_time"] = pd.to_datetime(working["Date"], errors="coerce") if "Date" in working.columns else pd.NaT
    working["_date_only"] = working["_date_time"].dt.date

    snippet_col = "Snippet" if "Snippet" in working.columns else None
    if snippet_col is None:
        working["_snippet_text"] = ""
    else:
        working["_snippet_text"] = working[snippet_col].fillna("").astype(str)

    working["_snippet_norm"] = (
        working["_snippet_text"]
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    working["_snippet_len"] = working["_snippet_text"].str.len()

    group_cols = [c for c in ["Outlet", "Type", "_date_only"] if c in working.columns]
    if not group_cols:
        empty_result = (broadcast_df.copy(), pd.DataFrame())
        if return_mapping:
            return empty_result[0], empty_result[1], []
        return empty_result

    duplicate_indexes = set()
    duplicate_to_keep_pairs: list[tuple[int, int]] = []

    for _, group in working.groupby(group_cols, dropna=False):
        valid_group = group[group["_date_time"].notna()].sort_values(["_date_time", "_original_order"])
        if len(valid_group) < 2:
            continue

        remaining_group = valid_group.copy()

        exact_duplicate_indexes = set()
        for _, snippet_group in remaining_group.groupby("_snippet_norm", dropna=False):
            if len(snippet_group) < 2:
                continue
            clusters = _cluster_time_proximate_indices(snippet_group.sort_values(["_date_time", "_original_order"]), "_date_time")
            for cluster in clusters:
                if len(cluster) <= 1:
                    continue
                component_rows = remaining_group.loc[cluster].copy().sort_values(
                    ["_snippet_len", "_date_time", "_original_order"],
                    ascending=[False, True, True],
                )
                keep_index = component_rows.index[0]
                dropped = set(cluster) - {keep_index}
                exact_duplicate_indexes.update(dropped)
                if return_mapping and SOURCE_ROW_COL in remaining_group.columns:
                    keep_source = int(remaining_group.loc[keep_index, SOURCE_ROW_COL])
                    duplicate_to_keep_pairs.extend(
                        (int(remaining_group.loc[idx, SOURCE_ROW_COL]), keep_source) for idx in dropped
                    )

        if exact_duplicate_indexes:
            duplicate_indexes.update(exact_duplicate_indexes)
            remaining_group = remaining_group.drop(index=list(exact_duplicate_indexes))

        if len(remaining_group) < 2:
            continue

        group_indices = remaining_group.index.tolist()
        adjacency = {idx: set() for idx in group_indices}

        for i, idx_i in enumerate(group_indices):
            row_i = remaining_group.loc[idx_i]
            snippet_i = row_i["_snippet_norm"]
            time_i = row_i["_date_time"]

            for idx_j in group_indices[i + 1:]:
                row_j = remaining_group.loc[idx_j]
                time_j = row_j["_date_time"]
                seconds_diff = abs((time_j - time_i).total_seconds())

                if seconds_diff > 60:
                    break

                snippet_j = row_j["_snippet_norm"]
                if snippet_i == snippet_j:
                    continue

                if not lengths_are_similar_enough(row_i["_snippet_len"], row_j["_snippet_len"]):
                    continue

                if SequenceMatcher(None, snippet_i, snippet_j).ratio() >= 0.90:
                    adjacency[idx_i].add(idx_j)
                    adjacency[idx_j].add(idx_i)

        visited = set()
        for idx in group_indices:
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

            component_rows = remaining_group.loc[component].copy().sort_values(
                ["_snippet_len", "_date_time", "_original_order"],
                ascending=[False, True, True],
            )
            keep_index = component_rows.index[0]
            dropped = set(component) - {keep_index}
            duplicate_indexes.update(dropped)
            if return_mapping and SOURCE_ROW_COL in remaining_group.columns:
                keep_source = int(remaining_group.loc[keep_index, SOURCE_ROW_COL])
                duplicate_to_keep_pairs.extend(
                    (int(remaining_group.loc[idx, SOURCE_ROW_COL]), keep_source) for idx in dropped
                )

    if duplicate_indexes:
        broadcast_dupes = working.loc[list(duplicate_indexes)].copy()
        cleaned_broadcast = working.drop(index=list(duplicate_indexes)).copy()
    else:
        broadcast_dupes = pd.DataFrame()
        cleaned_broadcast = working.copy()

    helper_cols = [
        "_original_order",
        "_date_time",
        "_date_only",
        "_snippet_text",
        "_snippet_norm",
        "_snippet_len",
    ]
    cleaned_broadcast.drop(columns=helper_cols, inplace=True, errors="ignore")
    broadcast_dupes.drop(columns=helper_cols, inplace=True, errors="ignore")

    if return_mapping:
        return cleaned_broadcast, broadcast_dupes, duplicate_to_keep_pairs
    return cleaned_broadcast, broadcast_dupes


def dedupe_traditional(df: pd.DataFrame, validate: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, float | str]], dict[str, object]]:
    validation: dict[str, object] = {"Enabled": validate}
    df = df.copy()
    timings: list[dict[str, float | str]] = []

    def record_timing(step_name: str, start_time: float) -> None:
        timings.append({
            "Step": step_name,
            "Seconds": round(time.perf_counter() - start_time, 4),
        })

    step_start = time.perf_counter()
    non_broadcast_df, broadcast_df = split_broadcast(df)
    record_timing("Split broadcast", step_start)

    step_start = time.perf_counter()
    deduped_non_broadcast, dupe_urls = dedupe_by_url(non_broadcast_df)
    record_timing("URL dedupe", step_start)

    step_start = time.perf_counter()
    deduped_non_broadcast, dupe_cols = dedupe_non_broadcast_by_fields(deduped_non_broadcast)
    record_timing("Field/snippet dedupe", step_start)

    step_start = time.perf_counter()
    step_start = time.perf_counter()
    if validate:
        cleaned_broadcast, broadcast_dupes, broadcast_duplicate_pairs = dedupe_broadcast(
            broadcast_df,
            return_mapping=True,
        )
    else:
        cleaned_broadcast, broadcast_dupes = dedupe_broadcast(
            broadcast_df,
            return_mapping=False,
        )
        broadcast_duplicate_pairs = []
    record_timing("Broadcast dedupe", step_start)

    if validate:
        legacy_start = time.perf_counter()
        legacy_cleaned_broadcast, legacy_broadcast_dupes = dedupe_broadcast_legacy(broadcast_df)
        validation["Broadcast Legacy Seconds"] = round(time.perf_counter() - legacy_start, 4)

        def _frame_signature(df_in: pd.DataFrame) -> set[tuple]:
            if df_in is None or df_in.empty:
                return set()
            cols = [c for c in ["Outlet", "Type", "Date", "Headline", "Snippet", "URL"] if c in df_in.columns]
            return set(map(tuple, df_in[cols].fillna("").astype(str).itertuples(index=False, name=None)))

        validation["Broadcast Matches Legacy"] = (
            _frame_signature(cleaned_broadcast) == _frame_signature(legacy_cleaned_broadcast)
            and _frame_signature(broadcast_dupes) == _frame_signature(legacy_broadcast_dupes)
        )
        validation["Broadcast Duplicate Pairs"] = broadcast_duplicate_pairs
        validation["Pre-Dedupe Broadcast Rows"] = broadcast_df.copy()
    else:
        validation["Broadcast Matches Legacy"] = None

    step_start = time.perf_counter()
    cleaned_df = pd.concat([deduped_non_broadcast, cleaned_broadcast], ignore_index=True).reset_index(drop=True)
    dupes_df = pd.concat([dupe_urls, dupe_cols, broadcast_dupes], ignore_index=True)
    record_timing("Recombine deduped sets", step_start)

    return cleaned_df, dupes_df, timings, validation


def extract_relevant_text(snippet: str) -> str:
    words = str(snippet or "").split()
    if len(words) > 250:
        return " ".join(words[:125] + words[-125:])
    return str(snippet or "")





def run_standard_cleaning(
    df: pd.DataFrame,
    merge_online: bool = True,
    drop_dupes: bool = True,
    add_flags: bool = True,
    validate_optimizations: bool = False,
) -> dict:
    timings: list[dict[str, float | str]] = []

    def record_timing(step_name: str, start_time: float) -> None:
        timings.append({
            "Step": step_name,
            "Seconds": round(time.perf_counter() - start_time, 4),
        })

    df = df.copy()
    if SOURCE_ROW_COL not in df.columns:
        df[SOURCE_ROW_COL] = range(len(df))

    step_start = time.perf_counter()
    df = prepare_text_columns(df)
    record_timing("Prepare text columns", step_start)

    step_start = time.perf_counter()
    df = standardize_media_types(df, merge_online=merge_online)
    record_timing("Standardize media types", step_start)

    step_start = time.perf_counter()
    df = reorder_key_columns(df)
    record_timing("Reorder key columns", step_start)

    step_start = time.perf_counter()
    df = clean_text_fields(df)
    record_timing("Clean text fields", step_start)

    step_start = time.perf_counter()
    df_traditional, df_social = split_social(df)
    record_timing("Split social", step_start)

    if drop_dupes:
        step_start = time.perf_counter()
        df_traditional, df_dupes, dedupe_timings, dedupe_validation = dedupe_traditional(
            df_traditional,
            validate=validate_optimizations,
        )
        record_timing("Deduplicate traditional", step_start)
        for row in dedupe_timings:
            timings.append({
                "Step": f"  - {row.get('Step', '')}",
                "Seconds": row.get("Seconds", 0.0),
            })
    else:
        df_dupes = pd.DataFrame()
        dedupe_validation = {"Enabled": validate_optimizations}

    step_start = time.perf_counter()
    df_traditional = df_traditional.reset_index(drop=True)
    df_social = df_social.reset_index(drop=True)
    df_dupes = df_dupes.reset_index(drop=True)
    record_timing("Reset indexes", step_start)

    if add_flags:
        step_start = time.perf_counter()
        df_traditional = add_coverage_flags(df_traditional)
        record_timing("Add coverage flags", step_start)

    return {
        "df_traditional": df_traditional,
        "df_social": df_social,
        "df_dupes": df_dupes,
        "timings": timings,
        "validation": {
            "dedupe": dedupe_validation,
        },
    }
