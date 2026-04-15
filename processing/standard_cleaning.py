# standard_cleaning.py

from __future__ import annotations

import re
from difflib import SequenceMatcher

import numpy as np
import pandas as pd
from titlecase import titlecase
from processing.coverage_flags import add_coverage_flags

SOCIAL_TYPES = ["FACEBOOK", "TWITTER", "X", "INSTAGRAM", "REDDIT", "YOUTUBE", "TIKTOK", "LINKEDIN", "BLUESKY"]
BROADCAST_TYPES = ["RADIO", "TV"]


def lengths_are_similar_enough(len_a: int, len_b: int, min_length_ratio: float = 0.70) -> bool:
    if len_a <= 0 or len_b <= 0:
        return False
    return min(len_a, len_b) / max(len_a, len_b) >= min_length_ratio


def normalize_snippet_for_compare(text: str) -> str:
    text = str(text or "")
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def has_nonblank_value(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().ne("")


def snippet_similarity(a: str, b: str) -> float:
    a_norm = normalize_snippet_for_compare(a)
    b_norm = normalize_snippet_for_compare(b)

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
    """URL dedupe stays within the same normalized media type and skips blank keys."""
    df = df.copy()

    if "URL" not in df.columns or "Type" not in df.columns:
        return df, pd.DataFrame()

    valid_mask = has_nonblank_value(df["URL"]) & has_nonblank_value(df["Type"])
    excluded_rows = df[~valid_mask].copy()
    working = df[valid_mask].copy()

    if working.empty:
        return df.copy(), pd.DataFrame()

    working["URL_Helper"] = (
        working["URL"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace("http:", "https:", regex=False)
    )
    working["_url_dedupe_key"] = working["Type"].astype(str).str.strip() + "||" + working["URL_Helper"]

    sort_cols = [c for c in ["_url_dedupe_key", "Author", "Impressions", "Date"] if c in working.columns]
    if sort_cols:
        ascending = []
        for col in sort_cols:
            if col == "Impressions":
                ascending.append(False)
            else:
                ascending.append(True)
        working = working.sort_values(sort_cols, ascending=ascending)

    dupe_urls = working[working["_url_dedupe_key"].duplicated(keep="first")].copy()
    deduped = working[~working["_url_dedupe_key"].duplicated(keep="first")].copy()

    deduped.drop(columns=["URL_Helper", "_url_dedupe_key"], inplace=True, errors="ignore")
    dupe_urls.drop(columns=["URL_Helper", "_url_dedupe_key"], inplace=True, errors="ignore")

    deduped = pd.concat([deduped, excluded_rows], ignore_index=True)
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

    required_mask = (
        has_nonblank_value(working["Outlet"])
        & has_nonblank_value(working["Type"])
        & has_nonblank_value(working["_snippet_text"])
        & working["_date_time"].notna()
    )
    excluded_rows = working[~required_mask].copy()
    working = working[required_mask].copy()

    group_cols = [c for c in ["Outlet", "Type", "_date_only"] if c in working.columns]
    if not group_cols or working.empty:
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
    cleaned_broadcast = pd.concat([cleaned_broadcast, excluded_rows], ignore_index=True)
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


def dedupe_broadcast(broadcast_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Broadcast dedupe uses outlet, normalized media type, date, time proximity, and snippet similarity."""
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

    required_mask = (
        has_nonblank_value(working["Outlet"])
        & has_nonblank_value(working["Type"])
        & has_nonblank_value(working["_snippet_text"])
        & working["_date_time"].notna()
    )
    excluded_rows = working[~required_mask].copy()
    working = working[required_mask].copy()

    group_cols = [c for c in ["Outlet", "Type", "_date_only"] if c in working.columns]
    if not group_cols or working.empty:
        return broadcast_df.copy(), pd.DataFrame()

    duplicate_indexes = set()

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
    cleaned_broadcast = pd.concat([cleaned_broadcast, excluded_rows], ignore_index=True)
    cleaned_broadcast.drop(columns=helper_cols, inplace=True, errors="ignore")
    broadcast_dupes.drop(columns=helper_cols, inplace=True, errors="ignore")

    return cleaned_broadcast, broadcast_dupes


def dedupe_traditional(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    non_broadcast_df, broadcast_df = split_broadcast(df)

    deduped_non_broadcast, dupe_urls = dedupe_by_url(non_broadcast_df)

    deduped_non_broadcast, dupe_cols = dedupe_non_broadcast_by_fields(deduped_non_broadcast)

    cleaned_broadcast, broadcast_dupes = dedupe_broadcast(broadcast_df)

    cleaned_df = pd.concat([deduped_non_broadcast, cleaned_broadcast], ignore_index=True).reset_index(drop=True)
    dupes_df = pd.concat([dupe_urls, dupe_cols, broadcast_dupes], ignore_index=True)
    return cleaned_df, dupes_df


def dedupe_social(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Social dedupe is intentionally conservative and only removes obvious accidental duplicates:
    1. same normalized Type + URL
    2. same normalized Type + exact normalized text + exact timestamp
    3. same normalized Type + exact normalized text + within 60 seconds
    """
    df = df.copy()

    if df.empty or "Type" not in df.columns:
        return df.copy(), pd.DataFrame()

    working = df.reset_index(drop=True).copy()
    working["_original_order"] = np.arange(len(working))
    working["_date_time"] = pd.to_datetime(working["Date"], errors="coerce") if "Date" in working.columns else pd.NaT
    working["_text_norm"] = (
        working["Snippet"].apply(normalize_snippet_for_compare) if "Snippet" in working.columns else ""
    )

    duplicate_indexes = set()

    url_valid_mask = has_nonblank_value(working["Type"])
    if "URL" in working.columns:
        url_valid_mask = url_valid_mask & has_nonblank_value(working["URL"])
        url_working = working[url_valid_mask].copy()
        if not url_working.empty:
            url_working["_url_norm"] = (
                url_working["URL"]
                .fillna("")
                .astype(str)
                .str.strip()
                .str.lower()
                .str.replace("http:", "https:", regex=False)
            )
            url_working["_social_url_key"] = url_working["Type"].astype(str).str.strip() + "||" + url_working["_url_norm"]
            sort_cols = [c for c in ["_social_url_key", "Impressions", "_date_time", "_original_order"] if c in url_working.columns]
            ascending = [True if col != "Impressions" else False for col in sort_cols]
            url_working = url_working.sort_values(sort_cols, ascending=ascending)
            duplicate_indexes.update(url_working[url_working["_social_url_key"].duplicated(keep="first")].index.tolist())

    remaining = working.drop(index=list(duplicate_indexes)).copy() if duplicate_indexes else working.copy()
    exact_required_mask = (
        has_nonblank_value(remaining["Type"])
        & has_nonblank_value(remaining["_text_norm"])
        & remaining["_date_time"].notna()
    )
    exact_working = remaining[exact_required_mask].copy()

    if not exact_working.empty:
        exact_working["_exact_social_key"] = (
            exact_working["Type"].astype(str).str.strip()
            + "||"
            + exact_working["_text_norm"].astype(str)
            + "||"
            + exact_working["_date_time"].astype(str)
        )
        sort_cols = [c for c in ["_exact_social_key", "Impressions", "_original_order"] if c in exact_working.columns]
        ascending = [True if col != "Impressions" else False for col in sort_cols]
        exact_working = exact_working.sort_values(sort_cols, ascending=ascending)
        duplicate_indexes.update(exact_working[exact_working["_exact_social_key"].duplicated(keep="first")].index.tolist())

    remaining = working.drop(index=list(duplicate_indexes)).copy() if duplicate_indexes else working.copy()
    window_required_mask = (
        has_nonblank_value(remaining["Type"])
        & has_nonblank_value(remaining["_text_norm"])
        & remaining["_date_time"].notna()
    )
    window_working = remaining[window_required_mask].copy()

    for _, group in window_working.groupby(["Type", "_text_norm"], dropna=False):
        if len(group) < 2:
            continue
        group = group.sort_values(["_date_time", "_original_order"])
        clusters = _cluster_time_proximate_indices(group, "_date_time")
        for cluster in clusters:
            if len(cluster) <= 1:
                continue
            component_rows = window_working.loc[cluster].copy()
            keep_index = choose_best_row_index(component_rows)
            duplicate_indexes.update(set(cluster) - {keep_index})

    if duplicate_indexes:
        social_dupes = working.loc[list(duplicate_indexes)].copy()
        cleaned_social = working.drop(index=list(duplicate_indexes)).copy()
    else:
        social_dupes = pd.DataFrame()
        cleaned_social = working.copy()

    helper_cols = [
        "_original_order",
        "_date_time",
        "_text_norm",
        "_url_norm",
        "_social_url_key",
        "_exact_social_key",
    ]
    cleaned_social.drop(columns=helper_cols, inplace=True, errors="ignore")
    social_dupes.drop(columns=helper_cols, inplace=True, errors="ignore")

    return cleaned_social, social_dupes


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
) -> dict:
    df = df.copy()
    df = prepare_text_columns(df)
    df = standardize_media_types(df, merge_online=merge_online)
    df = reorder_key_columns(df)
    df = clean_text_fields(df)
    df_traditional, df_social = split_social(df)

    if drop_dupes:
        df_traditional, df_dupes = dedupe_traditional(df_traditional)
        df_social, social_dupes = dedupe_social(df_social)
        df_dupes = pd.concat([df_dupes, social_dupes], ignore_index=True).reset_index(drop=True)
    else:
        df_dupes = pd.DataFrame()

    df_traditional = df_traditional.reset_index(drop=True)
    df_social = df_social.reset_index(drop=True)
    df_dupes = df_dupes.reset_index(drop=True)

    if add_flags:
        df_traditional = add_coverage_flags(df_traditional)

    return {
        "df_traditional": df_traditional,
        "df_social": df_social,
        "df_dupes": df_dupes,
    }
