# top_stories.py

from __future__ import annotations

import re
import unicodedata
from typing import Any

import numpy as np
import pandas as pd

from processing.prominence import get_prominence_weight_series

TOP_STORY_DEFAULT_COLUMNS = {
    "Group ID": pd.NA,
    "Headline": "",
    "Date": pd.NaT,
    "Mentions": 1,
    "Impressions": 0,
    "Effective Reach": 0,
    "Type": "",
    "Outlet": "",
    "URL": "",
    "Snippet": "",
    "Tags": "",
    "Coverage Flags": "",
}

TIER_ONE_OUTLETS = {
    "new york times",
    "the new york times",
    "nyt",
    "washington post",
    "the washington post",
    "wapo",
    "usa today",
    "los angeles times",
    "la times",
    "financial times",
    "the financial times",
    "bloomberg",
    "bloomberg news",
    "politico",
    "the guardian",
    "guardian",
    "the times",
    "the sunday times",
    "telegraph",
    "the telegraph",
    "associated press",
    "ap",
    "cnn",
    "abc news",
    "nbc news",
    "cbs news",
    "npr",
    "fox news",
    "bbc",
    "bbc news",
    "sky news",
    "al jazeera",
    "al jazeera english",
    "france 24",
    "deutsche welle",
    "dw",
    "euronews",
    "globe and mail",
    "the globe and mail",
    "toronto star",
    "the toronto star",
    "national post",
    "the national post",
    "montreal gazette",
    "vancouver sun",
    "calgary herald",
    "edmonton journal",
    "wall street journal",
    "the wall street journal",
    "wsj",
    "reuters",
    "associated press",
    "ap news",
    "canadian press",
    "the canadian press",
    "cbc",
    "cbc news",
    "global news",
    "ctv news",
    "cp24",
    "bnn bloomberg",
    "bnn",
}


def normalize_top_stories_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize dataframe enough for Top Stories page to work safely."""
    df = df.copy()

    for col, default in TOP_STORY_DEFAULT_COLUMNS.items():
        if col not in df.columns:
            df[col] = default

    text_columns = ["Headline", "Type", "Outlet", "URL", "Snippet", "Tags", "Coverage Flags"]
    for col in text_columns:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)

    if "Mentions" in df.columns:
        df["Mentions"] = pd.to_numeric(df["Mentions"], errors="coerce").fillna(1).astype(int)

    if "Impressions" in df.columns:
        df["Impressions"] = pd.to_numeric(df["Impressions"], errors="coerce").fillna(0)
    if "Effective Reach" in df.columns:
        df["Effective Reach"] = pd.to_numeric(df["Effective Reach"], errors="coerce").fillna(0)

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.date

    return df


def _normalize_recommendation_text(text: str) -> str:
    text = str(text or "")
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.casefold()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _clean_recommendation_terms(terms: list[str] | None) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for term in terms or []:
        normalized = _normalize_recommendation_text(term)
        if not normalized or len(normalized) < 3:
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        cleaned.append(normalized)
    return cleaned


def _term_presence_score(text: str, terms: list[str]) -> tuple[int, int]:
    normalized_text = f" {_normalize_recommendation_text(text)} "
    if not normalized_text.strip() or not terms:
        return 0, 0

    matched = 0
    for term in terms:
        if f" {term} " in normalized_text:
            matched += 1
    return matched, len(terms)


def _headline_position_score(text: str, terms: list[str]) -> float:
    normalized_text = _normalize_recommendation_text(text)
    if not normalized_text or not terms:
        return 0.0

    best_score = 0.0
    for term in terms:
        idx = normalized_text.find(term)
        if idx < 0:
            continue
        if idx == 0:
            best_score = max(best_score, 1.0)
        elif idx <= 20:
            best_score = max(best_score, 0.85)
        elif idx <= 50:
            best_score = max(best_score, 0.65)
        else:
            best_score = max(best_score, 0.45)
    return best_score


def _snippet_lead_position_score(text: str, terms: list[str]) -> float:
    normalized_text = _normalize_recommendation_text(" ".join(str(text or "").split()[:125]))
    if not normalized_text or not terms:
        return 0.0

    best_score = 0.0
    for term in terms:
        idx = normalized_text.find(term)
        if idx < 0:
            continue
        if idx <= 30:
            best_score = max(best_score, 0.8)
        elif idx <= 80:
            best_score = max(best_score, 0.55)
        else:
            best_score = max(best_score, 0.25)
    return best_score


def _is_tier_one_outlet(outlet: str) -> bool:
    normalized = _normalize_recommendation_text(outlet)
    if not normalized:
        return False
    return normalized in TIER_ONE_OUTLETS


def recommend_top_story_group_ids(
    candidate_df: pd.DataFrame,
    *,
    entity_terms: list[str] | None = None,
    selected_prominence_column: str = "",
    count: int = 10,
) -> list[str]:
    working = normalize_top_stories_df(candidate_df.copy())
    if working.empty or "Group ID" not in working.columns:
        return []

    candidate_pool_frames: list[pd.DataFrame] = []
    for metric in ["Mentions", "Impressions", "Effective Reach"]:
        if metric in working.columns:
            metric_top = working.sort_values(
                by=[metric, "Mentions", "Impressions", "Effective Reach"],
                ascending=[False, False, False, False],
                na_position="last",
            ).head(20)
            candidate_pool_frames.append(metric_top)

    if not candidate_pool_frames:
        return []

    pool = pd.concat(candidate_pool_frames, ignore_index=True)
    pool["_group_key"] = pool["Group ID"].astype(str).str.strip()
    pool = pool.drop_duplicates(subset=["_group_key"], keep="first").copy()
    if pool.empty:
        return []

    entity_terms = _clean_recommendation_terms(entity_terms)

    for metric in ["Mentions", "Impressions", "Effective Reach"]:
        if metric not in pool.columns:
            pool[metric] = 0
        pool[metric] = pd.to_numeric(pool[metric], errors="coerce").fillna(0)

        rank_series = (
            pool[metric]
            .rank(method="dense", ascending=False)
            .astype(float)
        )
        max_rank = float(rank_series.max()) if not rank_series.empty else 1.0
        if max_rank <= 1:
            pool[f"_{metric}_rank_score"] = 1.0
        else:
            pool[f"_{metric}_rank_score"] = 1.0 - ((rank_series - 1.0) / (max_rank - 1.0))

    pool["_headline_term_hits"] = pool["Headline"].apply(lambda value: _term_presence_score(value, entity_terms)[0])
    pool["_snippet_term_hits"] = pool["Example Snippet"].apply(
        lambda value: _term_presence_score(" ".join(str(value or "").split()[:125]), entity_terms)[0]
    )
    pool["_headline_position_score"] = pool["Headline"].apply(lambda value: _headline_position_score(value, entity_terms))
    pool["_snippet_position_score"] = pool["Example Snippet"].apply(
        lambda value: _snippet_lead_position_score(value, entity_terms)
    )
    pool["_multi_term_bonus"] = (
        (pool["_headline_term_hits"] >= 2).astype(float) * 0.75
        + (pool["_snippet_term_hits"] >= 2).astype(float) * 0.35
    )
    pool["_passing_mention_penalty"] = (
        (pool["_headline_term_hits"] == 0).astype(float)
        * (pool["_snippet_term_hits"] == 1).astype(float)
        * 0.75
    )

    pool["_coverage_flag_text"] = pool.get("Coverage Flags", "").fillna("").astype(str).str.casefold()
    pool["_flag_penalty"] = 0.0
    pool.loc[pool["_coverage_flag_text"].str.contains("press release", regex=False), "_flag_penalty"] += 2.75
    pool.loc[pool["_coverage_flag_text"].str.contains("advertorial", regex=False), "_flag_penalty"] += 3.0
    pool.loc[pool["_coverage_flag_text"].str.contains("financial outlet", regex=False), "_flag_penalty"] += 1.4
    pool["_tier_one_bonus"] = pool["Example Outlet"].apply(lambda value: 0.75 if _is_tier_one_outlet(value) else 0.0)
    pool["_selected_prominence_weight"] = get_prominence_weight_series(pool, selected_prominence_column)

    pool["_recommendation_score"] = (
        pool["_headline_term_hits"] * 3.2
        + pool["_snippet_term_hits"] * 1.4
        + pool["_headline_position_score"] * 2.5
        + pool["_snippet_position_score"] * 1.25
        + pool["_multi_term_bonus"]
        + pool["_Mentions_rank_score"] * 3.0
        + pool["_Impressions_rank_score"] * 2.0
        + pool["_Effective Reach_rank_score"] * 2.0
        + pool["_tier_one_bonus"]
        + pool["_selected_prominence_weight"] * 1.4
        - pool["_flag_penalty"]
        - pool["_passing_mention_penalty"]
    )

    ranked = pool.sort_values(
        by=[
            "_recommendation_score",
            "_headline_term_hits",
            "_snippet_term_hits",
            "Mentions",
            "Impressions",
            "Effective Reach",
        ],
        ascending=[False, False, False, False, False, False],
        na_position="last",
    )

    return ranked["_group_key"].head(count).tolist()


def _empty_top_story_candidate_df() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "Group ID",
        "Headline",
        "Date",
        "Mentions",
        "Impressions",
        "Effective Reach",
        "Example Outlet",
        "Example URL",
        "Example Type",
        "Example Snippet",
        "Source Group IDs",
    ])


def _normalize_top_story_headline(text: str) -> str:
    text = str(text or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()


def _normalize_top_story_snippet(text: str) -> str:
    text = str(text or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()


def strip_html_tags(text: str) -> str:
    raw = str(text or "")
    no_tags = re.sub(r"<[^>]+>", " ", raw)
    return re.sub(r"\s+", " ", no_tags).strip()


def _top_story_snippet_fingerprint(text: str, max_chars: int = 160) -> str:
    normalized = _normalize_top_story_snippet(text)
    if not normalized:
        return ""
    return normalized[:max_chars].strip()


def _headline_similarity(a: str, b: str) -> float:
    a_norm = _normalize_top_story_headline(a)
    b_norm = _normalize_top_story_headline(b)
    if not a_norm or not b_norm:
        return 0.0

    a_tokens = set(a_norm.split())
    b_tokens = set(b_norm.split())
    if not a_tokens or not b_tokens:
        return 0.0

    overlap = len(a_tokens & b_tokens)
    return overlap / max(min(len(a_tokens), len(b_tokens)), 1)


def _pick_consolidated_top_story_row(group: pd.DataFrame) -> pd.Series:
    working = group.copy()
    working["_snippet_len"] = working["Example Snippet"].fillna("").astype(str).str.len()
    working["_headline_len"] = working["Headline"].fillna("").astype(str).str.len()
    working["_has_url"] = working["Example URL"].fillna("").astype(str).str.strip().ne("")
    ordered = working.sort_values(
        by=["_snippet_len", "Mentions", "Impressions", "_has_url", "_headline_len"],
        ascending=[False, False, False, False, False],
        na_position="last",
    )
    return ordered.iloc[0]


def _top_story_merge_type(example_type: str) -> str:
    example_type = str(example_type or "").strip().upper()
    if example_type in {"ONLINE", "PRINT"}:
        return "TEXT"
    return example_type


def _assign_text_story_merge_ids(mergeable: pd.DataFrame) -> pd.Series:
    merge_ids = pd.Series(index=mergeable.index, dtype="object")
    next_id = 0
    ordered = mergeable.sort_values(["Date", "Headline"], na_position="last", kind="mergesort")
    headline_groups: dict[str, tuple[str, object]] = {}
    snippet_groups: list[tuple[str, str, str, object]] = []

    for idx, row in ordered.iterrows():
        date_value = row.get("Date")
        headline_key = row.get("_normalized_headline", "")
        snippet_key = row.get("_snippet_fingerprint", "")
        headline = row.get("Headline", "")

        chosen_group: str | None = None

        if headline_key and headline_key in headline_groups:
            existing_group, existing_date = headline_groups[headline_key]
            if (
                pd.notna(date_value)
                and pd.notna(existing_date)
                and abs((pd.Timestamp(date_value) - pd.Timestamp(existing_date)).days) <= 1
            ):
                chosen_group = existing_group

        if chosen_group is None and snippet_key and len(snippet_key) >= 80:
            for existing_snippet_key, existing_headline, existing_group, existing_date in reversed(snippet_groups):
                if pd.notna(date_value) and pd.notna(existing_date):
                    date_gap = abs((pd.Timestamp(date_value) - pd.Timestamp(existing_date)).days)
                else:
                    date_gap = 999
                if date_gap > 1:
                    continue
                if snippet_key == existing_snippet_key and _headline_similarity(headline, existing_headline) >= 0.5:
                    chosen_group = existing_group
                    break

        if chosen_group is None:
            chosen_group = f"TEXT::{date_value}::{next_id}"
            next_id += 1

        merge_ids.loc[idx] = chosen_group

        if headline_key:
            headline_groups[headline_key] = (chosen_group, date_value)
        if snippet_key and len(snippet_key) >= 80:
            snippet_groups.append((snippet_key, headline, chosen_group, date_value))

    return merge_ids


def build_prime_grouped_story_candidates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build one representative candidate per original Group ID, before any
    Top Stories-specific consolidation.
    """
    df = normalize_top_stories_df(df)

    if df.empty or "Group ID" not in df.columns:
        return _empty_top_story_candidate_df()

    if "Prime Example" not in df.columns:
        raise ValueError(
            "Prime Example column is missing. Please rerun Basic Cleaning so grouped stories have a canonical exemplar."
        )

    rows: list[dict[str, Any]] = []

    for group_id, group in df.groupby("Group ID", dropna=False):
        mentions = pd.to_numeric(group["Mentions"], errors="coerce").fillna(0).sum()
        impressions = pd.to_numeric(group["Impressions"], errors="coerce").fillna(0).sum()
        effective_reach = pd.to_numeric(group["Effective Reach"], errors="coerce").fillna(0).sum()

        prime_group = group[group["Prime Example"] == 1].copy()
        if prime_group.empty:
            continue

        best_row = prime_group.iloc[0]
        group_id_str = "" if pd.isna(group_id) else str(group_id).strip()

        rows.append({
            "Group ID": group_id,
            "Headline": best_row.get("Headline", ""),
            "Date": best_row.get("Date", pd.NaT),
            "Mentions": int(mentions),
            "Impressions": impressions,
            "Effective Reach": effective_reach,
            "Example Outlet": best_row.get("Outlet", ""),
            "Example URL": best_row.get("URL", ""),
            "Example Type": best_row.get("Type", ""),
            "Example Snippet": best_row.get("Snippet", ""),
            "Source Group IDs": group_id_str,
        })

    if not rows:
        return _empty_top_story_candidate_df()

    result = pd.DataFrame(rows)
    result["Date"] = pd.to_datetime(result["Date"], errors="coerce").dt.date
    result["Impressions"] = pd.to_numeric(result["Impressions"], errors="coerce").fillna(0)
    result["Effective Reach"] = pd.to_numeric(result["Effective Reach"], errors="coerce").fillna(0).astype(int)
    result["Mentions"] = pd.to_numeric(result["Mentions"], errors="coerce").fillna(0).astype(int)
    return result[[
        "Group ID",
        "Headline",
        "Date",
        "Mentions",
        "Impressions",
        "Effective Reach",
        "Example Outlet",
        "Example URL",
        "Example Type",
        "Example Snippet",
        "Source Group IDs",
    ]]


def consolidate_top_story_candidates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Consolidate likely duplicate Top Story candidates for ONLINE/PRINT together
    using normalized headline + date, while keeping broadcast types separate.
    """
    working = normalize_top_stories_df(df)
    if working.empty:
        return working

    if "Example Type" not in working.columns:
        working["Example Type"] = ""

    working["Date"] = pd.to_datetime(working["Date"], errors="coerce").dt.date
    working["Example Type"] = working["Example Type"].fillna("").astype(str).str.strip().str.upper()
    working["_merge_type"] = working["Example Type"].map(_top_story_merge_type)
    working["_normalized_headline"] = working["Headline"].map(_normalize_top_story_headline)
    working["_snippet_fingerprint"] = working["Example Snippet"].map(_top_story_snippet_fingerprint)
    working["Source Group IDs"] = (
        working["Group ID"]
        .apply(lambda x: "" if pd.isna(x) else str(x).strip())
        .replace("None", "")
    )

    mergeable_mask = (
        working["_merge_type"].eq("TEXT")
        & working["Date"].notna()
        & (
            working["_normalized_headline"].ne("")
            | working["_snippet_fingerprint"].ne("")
        )
    )

    passthrough = working.loc[~mergeable_mask].copy()
    mergeable = working.loc[mergeable_mask].copy()

    if mergeable.empty:
        return working.drop(columns=["_normalized_headline", "_snippet_fingerprint", "_merge_type"], errors="ignore")

    rows: list[dict[str, Any]] = []
    mergeable["_text_story_merge_id"] = _assign_text_story_merge_ids(mergeable)

    for group_key, group in mergeable.groupby("_text_story_merge_id", dropna=False):
        if len(group) == 1:
            rows.append(group.drop(columns=["_normalized_headline", "_snippet_fingerprint", "_merge_type", "_text_story_merge_id"], errors="ignore").iloc[0].to_dict())
            continue

        best_row = _pick_consolidated_top_story_row(group)
        source_ids = [str(x).strip() for x in group["Group ID"].dropna().astype(str).tolist() if str(x).strip()]
        source_ids = list(dict.fromkeys(source_ids))

        merged_row = best_row.drop(labels=["_normalized_headline", "_snippet_fingerprint", "_merge_type", "_text_story_merge_id"], errors="ignore").to_dict()
        merged_row["Group ID"] = f"TOPMERGE::{group_key}"
        merged_row["Mentions"] = int(pd.to_numeric(group["Mentions"], errors="coerce").fillna(0).sum())
        merged_row["Impressions"] = int(pd.to_numeric(group["Impressions"], errors="coerce").fillna(0).sum())
        merged_row["Effective Reach"] = int(pd.to_numeric(group["Effective Reach"], errors="coerce").fillna(0).sum())
        merged_types = [str(x).strip().upper() for x in group["Example Type"].dropna().astype(str).tolist() if str(x).strip()]
        merged_types = list(dict.fromkeys(merged_types))
        merged_row["Example Type"] = ", ".join(merged_types)
        merged_row["Source Group IDs"] = " | ".join(source_ids)
        rows.append(merged_row)

    merged_df = pd.DataFrame(rows)
    passthrough = passthrough.drop(columns=["_normalized_headline", "_snippet_fingerprint", "_merge_type"], errors="ignore")
    passthrough["Source Group IDs"] = (
        passthrough["Source Group IDs"].fillna("").astype(str).replace("None", "")
    )
    if "Source Group IDs" not in merged_df.columns:
        merged_df["Source Group IDs"] = merged_df["Group ID"].fillna("").astype(str)

    out = pd.concat([merged_df, passthrough], ignore_index=True, sort=False)
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce").dt.date
    out["Mentions"] = pd.to_numeric(out["Mentions"], errors="coerce").fillna(0).astype(int)
    out["Impressions"] = pd.to_numeric(out["Impressions"], errors="coerce").fillna(0).astype(int)
    out["Effective Reach"] = pd.to_numeric(out["Effective Reach"], errors="coerce").fillna(0).astype(int)
    out["Source Group IDs"] = out["Source Group IDs"].fillna("").astype(str).replace("None", "")
    desired_cols = [
        "Group ID",
        "Headline",
        "Date",
        "Mentions",
        "Impressions",
        "Effective Reach",
        "Example Outlet",
        "Example URL",
        "Example Type",
        "Example Snippet",
        "Source Group IDs",
    ]
    existing = [c for c in desired_cols if c in out.columns]
    return out[existing].copy()



def build_grouped_story_candidates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group filtered row-level stories by Group ID and use the canonical Prime Example row
    as the representative row.
    """
    result = build_prime_grouped_story_candidates(df)
    return consolidate_top_story_candidates(result)


def parse_source_group_ids(value: Any, fallback_group_id: Any = None) -> list[str]:
    parts: list[str] = []
    raw = str(value or "").strip()
    if raw:
        parts.extend([p.strip() for p in raw.split("|") if p.strip()])

    if not parts and fallback_group_id is not None and not pd.isna(fallback_group_id):
        fallback = str(fallback_group_id).strip()
        if fallback:
            parts.append(fallback)

    return list(dict.fromkeys(parts))


def build_story_identity_key(source_group_ids: Any, fallback_group_id: Any = None) -> str:
    source_ids = parse_source_group_ids(source_group_ids, fallback_group_id=fallback_group_id)
    if source_ids:
        return "SRC::" + "|".join(sorted(source_ids))

    if fallback_group_id is not None and not pd.isna(fallback_group_id):
        fallback = str(fallback_group_id).strip()
        if fallback:
            return f"GRP::{fallback}"
    return ""


def dedupe_saved_top_stories(saved_df: pd.DataFrame) -> pd.DataFrame:
    if saved_df.empty or "Group ID" not in saved_df.columns:
        return saved_df

    working = saved_df.copy()
    source_group_ids = (
        working["Source Group IDs"]
        if "Source Group IDs" in working.columns
        else pd.Series(index=working.index, data="")
    )
    working["_story_identity_key"] = [
        build_story_identity_key(source_group_ids.iloc[idx], working.iloc[idx].get("Group ID"))
        for idx in range(len(working))
    ]
    working = working.drop_duplicates(subset=["_story_identity_key"], keep="last").copy()
    return working.drop(columns=["_story_identity_key"], errors="ignore").reset_index(drop=True)


def build_source_candidate_table(
    df: pd.DataFrame,
    source_group_ids: Any,
    fallback_group_id: Any = None,
) -> pd.DataFrame:
    candidates = build_prime_grouped_story_candidates(df)
    return build_source_candidate_table_from_candidates(
        candidates=candidates,
        source_group_ids=source_group_ids,
        fallback_group_id=fallback_group_id,
    )


def build_source_candidate_table_from_candidates(
    candidates: pd.DataFrame,
    source_group_ids: Any,
    fallback_group_id: Any = None,
    require_url_if_available: bool = False,
) -> pd.DataFrame:
    if candidates.empty:
        return candidates

    source_ids = parse_source_group_ids(source_group_ids, fallback_group_id=fallback_group_id)
    if not source_ids:
        return _empty_top_story_candidate_df()

    candidate_working = candidates.copy()
    candidate_working["_group_id_key"] = candidate_working["Group ID"].astype(str)
    candidate_working = candidate_working[candidate_working["_group_id_key"].isin(source_ids)].copy()
    if candidate_working.empty:
        return _empty_top_story_candidate_df()

    fallback_key = "" if pd.isna(fallback_group_id) else str(fallback_group_id).strip()
    candidate_working["_same_original_group"] = candidate_working["_group_id_key"].eq(fallback_key)
    candidate_working["_snippet_len"] = candidate_working["Example Snippet"].fillna("").astype(str).str.len()
    candidate_working["_has_url"] = candidate_working["Example URL"].fillna("").astype(str).str.strip().ne("")
    if require_url_if_available and candidate_working["_has_url"].any():
        candidate_working = candidate_working[candidate_working["_has_url"]].copy()
        if candidate_working.empty:
            return _empty_top_story_candidate_df()
    candidate_working["_headline_len"] = candidate_working["Headline"].fillna("").astype(str).str.len()
    candidate_working = candidate_working.sort_values(
        by=["_same_original_group", "_snippet_len", "Mentions", "Impressions", "_has_url", "_headline_len"],
        ascending=[False, False, False, False, False, False],
        na_position="last",
    ).reset_index(drop=True)
    candidate_working["Source Rank"] = range(1, len(candidate_working) + 1)

    display_cols = [
        "Source Rank",
        "Group ID",
        "Headline",
        "Date",
        "Mentions",
        "Impressions",
        "Effective Reach",
        "Example Outlet",
        "Example URL",
        "Example Type",
        "Example Snippet",
        "Source Group IDs",
    ]
    existing = [c for c in display_cols if c in candidate_working.columns]
    return candidate_working[existing].copy()


def rotate_saved_story_source(
    saved_df: pd.DataFrame,
    source_df: pd.DataFrame,
    story_group_id: Any,
    step: int = 1,
) -> pd.DataFrame:
    candidates = build_prime_grouped_story_candidates(source_df)
    return rotate_saved_story_source_from_candidates(
        saved_df=saved_df,
        candidates=candidates,
        story_group_id=story_group_id,
        step=step,
    )


def rotate_saved_story_source_from_candidates(
    saved_df: pd.DataFrame,
    candidates: pd.DataFrame,
    story_group_id: Any,
    step: int = 1,
) -> pd.DataFrame:
    if saved_df.empty or "Group ID" not in saved_df.columns:
        return saved_df

    story_key = "" if pd.isna(story_group_id) else str(story_group_id).strip()
    if not story_key:
        return saved_df

    working = saved_df.copy()
    working["_group_id_key"] = working["Group ID"].astype(str)
    match_idx = working.index[working["_group_id_key"] == story_key].tolist()
    if not match_idx:
        return saved_df

    row_idx = match_idx[0]
    current_row = working.loc[row_idx]
    source_candidates = build_source_candidate_table_from_candidates(
        candidates=candidates,
        source_group_ids=current_row.get("Source Group IDs", ""),
        fallback_group_id=current_row.get("Group ID"),
        require_url_if_available=bool(str(current_row.get("Example URL", "") or "").strip()),
    )
    if len(source_candidates) <= 1:
        return saved_df

    current_url = str(current_row.get("Example URL", "") or "").strip()
    current_outlet = str(current_row.get("Example Outlet", "") or "").strip()
    current_type = str(current_row.get("Example Type", "") or "").strip()

    current_pos = 0
    for pos, (_, candidate) in enumerate(source_candidates.iterrows()):
        if (
            str(candidate.get("Example URL", "") or "").strip() == current_url
            and str(candidate.get("Example Outlet", "") or "").strip() == current_outlet
            and str(candidate.get("Example Type", "") or "").strip() == current_type
        ):
            current_pos = pos
            break

    next_pos = (current_pos + step) % len(source_candidates)
    next_candidate = source_candidates.iloc[next_pos]

    for col in ["Example Outlet", "Example URL", "Example Type", "Example Snippet", "Date"]:
        if col in working.columns and col in next_candidate.index:
            working.at[row_idx, col] = next_candidate.get(col, working.at[row_idx, col])

    for generated_col in ["Chart Callout", "Top Story Summary", "Entity Sentiment Label", "Entity Sentiment Rationale", "Entity Sentiment"]:
        if generated_col in working.columns:
            working.at[row_idx, generated_col] = ""

    working = working.drop(columns=["_group_id_key"], errors="ignore")
    return working


def tokenize_boolean_query(query: str):
    token_pattern = r'"[^"]+"|\(|\)|\bAND\b|\bOR\b|\bNOT\b|[^\s()]+'
    tokens = re.findall(token_pattern, query, flags=re.IGNORECASE)
    return [token.strip() for token in tokens if token.strip()]


def normalize_tokens_for_implicit_and(tokens):
    normalized = []

    def is_operand(tok):
        upper = tok.upper()
        return tok not in ("(", ")") and upper not in ("AND", "OR", "NOT")

    def ends_expr(tok):
        return tok == ")" or is_operand(tok)

    def starts_expr(tok):
        upper = tok.upper()
        return tok == "(" or is_operand(tok) or upper == "NOT"

    for i, token in enumerate(tokens):
        if i > 0:
            prev = tokens[i - 1]
            if ends_expr(prev) and starts_expr(token):
                normalized.append("AND")
        normalized.append(token)

    return normalized


def boolean_query_to_postfix(tokens):
    precedence = {"NOT": 3, "AND": 2, "OR": 1}
    right_associative = {"NOT"}
    output = []
    operators = []

    for token in tokens:
        upper = token.upper()

        if upper in precedence:
            while (
                operators
                and operators[-1] != "("
                and (
                    precedence.get(operators[-1], 0) > precedence[upper]
                    or (
                        precedence.get(operators[-1], 0) == precedence[upper]
                        and upper not in right_associative
                    )
                )
            ):
                output.append(operators.pop())
            operators.append(upper)

        elif token == "(":
            operators.append(token)

        elif token == ")":
            while operators and operators[-1] != "(":
                output.append(operators.pop())
            if operators and operators[-1] == "(":
                operators.pop()

        else:
            output.append(token)

    while operators:
        output.append(operators.pop())

    return output


def evaluate_boolean_query(series: pd.Series, query: str) -> pd.Series:
    series = series.fillna("").astype(str)

    query = str(query).strip()
    if not query:
        return pd.Series(True, index=series.index)

    tokens = tokenize_boolean_query(query)
    if not tokens:
        return pd.Series(True, index=series.index)

    tokens = normalize_tokens_for_implicit_and(tokens)
    postfix = boolean_query_to_postfix(tokens)
    stack = []

    for token in postfix:
        upper = token.upper()

        if upper == "NOT":
            if len(stack) < 1:
                return pd.Series(False, index=series.index)
            operand = stack.pop()
            stack.append(~operand)

        elif upper == "AND":
            if len(stack) < 2:
                return pd.Series(False, index=series.index)
            right = stack.pop()
            left = stack.pop()
            stack.append(left & right)

        elif upper == "OR":
            if len(stack) < 2:
                return pd.Series(False, index=series.index)
            right = stack.pop()
            left = stack.pop()
            stack.append(left | right)

        else:
            term = token[1:-1] if len(token) >= 2 and token.startswith('"') and token.endswith('"') else token
            mask = series.str.contains(term, case=False, na=False, regex=False)
            stack.append(mask)

    if len(stack) != 1:
        return pd.Series(False, index=series.index)

    return stack[0]


def apply_filters(
    df: pd.DataFrame,
    start_date,
    end_date,
    exclude_types,
    exclude_coverage_flags,
    advanced_filters,
):
    working = normalize_top_stories_df(df.copy())

    if start_date is not None:
        working = working[working["Date"].notna()]
        working = working[working["Date"] >= start_date]

    if end_date is not None:
        working = working[working["Date"].notna()]
        working = working[working["Date"] <= end_date]

    if exclude_types:
        working = working[~working["Type"].isin(exclude_types)]

    if exclude_coverage_flags and "Coverage Flags" in working.columns:
        working = working[~working["Coverage Flags"].isin(exclude_coverage_flags)]

    for condition in advanced_filters:
        column = condition.get("column")
        value = condition.get("value")

        if not column or not value or column not in working.columns:
            continue

        series = working[column].fillna("").astype(str)
        mask = evaluate_boolean_query(series, value)
        working = working[mask]

    return working


def save_selected_rows(
    updated_data: pd.DataFrame,
    full_candidate_df: pd.DataFrame,
    added_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Return updated saved top stories dataset, preserving hidden/internal columns
    by selecting from the full candidate dataframe using Group ID.
    """
    if "Top Story" not in updated_data.columns or "Group ID" not in updated_data.columns:
        return added_df

    selected_ids = (
        updated_data.loc[updated_data["Top Story"] == True, "Group ID"]
        .dropna()
        .astype(str)
        .tolist()
    )

    if not selected_ids:
        return added_df

    full_candidate_df = full_candidate_df.copy()
    full_candidate_df["_group_id_key"] = full_candidate_df["Group ID"].astype(str)

    selected_rows = full_candidate_df.loc[
        full_candidate_df["_group_id_key"].isin(selected_ids)
    ].copy()
    selected_rows = selected_rows.drop(columns=["_group_id_key"], errors="ignore")

    new_added_df = pd.concat([added_df, selected_rows], ignore_index=True)
    return dedupe_saved_top_stories(new_added_df)


def remove_saved_candidates_from_display(
    display_df: pd.DataFrame,
    added_df: pd.DataFrame,
) -> pd.DataFrame:
    if display_df.empty or added_df.empty or "Group ID" not in display_df.columns or "Group ID" not in added_df.columns:
        return display_df

    display_working = display_df.copy()
    added_working = added_df.copy()

    display_source_ids = (
        display_working["Source Group IDs"]
        if "Source Group IDs" in display_working.columns
        else pd.Series(index=display_working.index, data="")
    )
    added_source_ids = (
        added_working["Source Group IDs"]
        if "Source Group IDs" in added_working.columns
        else pd.Series(index=added_working.index, data="")
    )

    display_working["_story_identity_key"] = [
        build_story_identity_key(display_source_ids.iloc[idx], display_working.iloc[idx].get("Group ID"))
        for idx in range(len(display_working))
    ]
    added_working["_story_identity_key"] = [
        build_story_identity_key(added_source_ids.iloc[idx], added_working.iloc[idx].get("Group ID"))
        for idx in range(len(added_working))
    ]

    existing_keys = set(
        added_working["_story_identity_key"].fillna("").astype(str).str.strip().tolist()
    )
    display_working = display_working[
        ~display_working["_story_identity_key"].fillna("").astype(str).str.strip().isin(existing_keys)
    ].copy()
    return display_working.drop(columns=["_story_identity_key"], errors="ignore")


def refresh_saved_story_metrics(
    saved_df: pd.DataFrame,
    source_df: pd.DataFrame,
) -> pd.DataFrame:
    if saved_df.empty or source_df.empty:
        return saved_df

    current_candidates = build_grouped_story_candidates(source_df)
    if current_candidates.empty:
        return saved_df

    saved_working = normalize_top_stories_df(saved_df.copy())
    candidate_working = normalize_top_stories_df(current_candidates.copy())

    saved_working["Source Group IDs"] = saved_working.get("Source Group IDs", "").fillna("").astype(str)
    candidate_working["Source Group IDs"] = candidate_working.get("Source Group IDs", "").fillna("").astype(str)

    saved_working["_source_key"] = saved_working["Source Group IDs"].str.strip()
    candidate_working["_source_key"] = candidate_working["Source Group IDs"].str.strip()
    saved_working["_group_key"] = saved_working["Group ID"].fillna("").astype(str).str.strip()
    candidate_working["_group_key"] = candidate_working["Group ID"].fillna("").astype(str).str.strip()

    candidate_by_source = (
        candidate_working[candidate_working["_source_key"] != ""]
        .drop_duplicates(subset=["_source_key"], keep="first")
        .set_index("_source_key")
    )
    candidate_by_group = (
        candidate_working[candidate_working["_group_key"] != ""]
        .drop_duplicates(subset=["_group_key"], keep="first")
        .set_index("_group_key")
    )

    metric_columns = ["Mentions", "Impressions", "Effective Reach"]

    for idx, row in saved_working.iterrows():
        match_row = None
        source_key = row.get("_source_key", "")
        group_key = row.get("_group_key", "")

        if source_key and source_key in candidate_by_source.index:
            match_row = candidate_by_source.loc[source_key]
        elif group_key and group_key in candidate_by_group.index:
            match_row = candidate_by_group.loc[group_key]

        if match_row is None:
            continue

        for col in metric_columns:
            if col in match_row.index:
                saved_working.at[idx, col] = match_row.get(col, saved_working.at[idx, col])

    return saved_working.drop(columns=["_source_key", "_group_key"], errors="ignore")


def refresh_generated_story_metrics(
    generated_df: pd.DataFrame,
    source_df: pd.DataFrame,
) -> pd.DataFrame:
    if generated_df.empty or source_df.empty:
        return generated_df

    current_candidates = build_grouped_story_candidates(source_df)
    if current_candidates.empty:
        return generated_df

    generated_working = normalize_top_stories_df(generated_df.copy())
    candidate_working = normalize_top_stories_df(current_candidates.copy())

    generated_working["Source Group IDs"] = generated_working.get("Source Group IDs", "").fillna("").astype(str)
    candidate_working["Source Group IDs"] = candidate_working.get("Source Group IDs", "").fillna("").astype(str)

    generated_working["_source_key"] = generated_working["Source Group IDs"].str.strip()
    candidate_working["_source_key"] = candidate_working["Source Group IDs"].str.strip()
    generated_working["_group_key"] = generated_working["Group ID"].fillna("").astype(str).str.strip()
    candidate_working["_group_key"] = candidate_working["Group ID"].fillna("").astype(str).str.strip()

    candidate_by_source = (
        candidate_working[candidate_working["_source_key"] != ""]
        .drop_duplicates(subset=["_source_key"], keep="first")
        .set_index("_source_key")
    )
    candidate_by_group = (
        candidate_working[candidate_working["_group_key"] != ""]
        .drop_duplicates(subset=["_group_key"], keep="first")
        .set_index("_group_key")
    )

    metric_columns = ["Mentions", "Impressions", "Effective Reach"]

    for idx, row in generated_working.iterrows():
        match_row = None
        source_key = row.get("_source_key", "")
        group_key = row.get("_group_key", "")

        if source_key and source_key in candidate_by_source.index:
            match_row = candidate_by_source.loc[source_key]
        elif group_key and group_key in candidate_by_group.index:
            match_row = candidate_by_group.loc[group_key]

        if match_row is None:
            continue

        for col in metric_columns:
            if col in match_row.index:
                generated_working.at[idx, col] = match_row.get(col, generated_working.at[idx, col])

    return generated_working.drop(columns=["_source_key", "_group_key"], errors="ignore")


def reset_generated_candidates(session_state) -> None:
    session_state.df_grouped = pd.DataFrame()
    session_state.filtered_df = pd.DataFrame()
    session_state.top_stories_generated = False
    session_state.top_stories_checked_group_ids = []
    session_state.top_stories_editor_version = int(session_state.get("top_stories_editor_version", 0) or 0) + 1
