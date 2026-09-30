from __future__ import annotations

import hashlib
import json
import re
from typing import Any

import pandas as pd
from openai import OpenAI

from processing.ai_sentiment import (
    DEFAULT_SENTIMENT_OBSERVATION_MODEL,
    build_sentiment_observation_prompt,
    build_sentiment_observation_schema,
)
from processing.ai_tagging import (
    DEFAULT_TAGGING_OBSERVATION_MODEL,
    RESERVED_OTHER_TAG,
    build_tag_observation_prompt,
    build_tag_observation_schema,
    normalize_tag_list,
)
from processing.prominence import get_prominence_weight_series
from processing.rapid_resolution import (
    RAPID_TAG_REVIEW_MODE_APPLICABLE,
    RAPID_TAG_REVIEW_MODE_BEST,
    build_effective_rapid_sentiment_frame,
    build_effective_rapid_tag_frame,
    build_final_rapid_sentiment_frame,
    build_final_rapid_tag_frame,
)
from utils.api_meter import add_api_usage, extract_usage_tokens


RAPID_SENTIMENT_SCALE_3_WAY = "3-way"
RAPID_SENTIMENT_SCALE_5_WAY = "5-way"
RAPID_INSIGHT_FIELD_OPTIONS = ["Outlet", "Date", "Media type", "Mentions", "Impressions", "Effective reach", "Examples"]
RAPID_SENTIMENT_ORDER_3_WAY = ["POSITIVE", "NEUTRAL", "NEGATIVE", "NOT RELEVANT"]
RAPID_SENTIMENT_ORDER_5_WAY = [
    "VERY POSITIVE",
    "SOMEWHAT POSITIVE",
    "NEUTRAL",
    "SOMEWHAT NEGATIVE",
    "VERY NEGATIVE",
    "NOT RELEVANT",
]
RAPID_PRIMARY_EXAMPLE_LIMIT = 10
RAPID_ALIGNED_EVIDENCE_LIMIT = 40


def _text_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series("", index=df.index, dtype="object")
    values = df[column]
    if isinstance(values, pd.DataFrame):
        values = values.iloc[:, 0]
    return values.fillna("").astype(str).str.strip()


def _truncate_text(text: Any, limit: int = 420) -> str:
    cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3].rstrip() + "..."


def _extract_json_payload(text: str) -> dict[str, Any] | None:
    raw = str(text or "").strip()
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        pass
    match = re.search(r"\{.*\}", raw, flags=re.S)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


def _normalized_rank_score(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").fillna(0)
    if numeric.empty:
        return pd.Series(dtype="float64")
    ranks = numeric.rank(method="dense", ascending=False).astype(float)
    max_rank = float(ranks.max()) if not ranks.empty else 1.0
    if max_rank <= 1:
        return pd.Series(1.0, index=numeric.index, dtype="float64")
    return 1.0 - ((ranks - 1.0) / (max_rank - 1.0))


def _add_observation_scores(working: pd.DataFrame, selected_prominence_column: str = "") -> pd.DataFrame:
    out = working.copy()
    out["_Group_Count_rank_score"] = _normalized_rank_score(out.get("Group Count", pd.Series(index=out.index, dtype="float64")))
    out["_Mentions_rank_score"] = _normalized_rank_score(out.get("Mentions", pd.Series(index=out.index, dtype="float64")))
    out["_Impressions_rank_score"] = _normalized_rank_score(out.get("Impressions", pd.Series(index=out.index, dtype="float64")))
    out["_Effective_Reach_rank_score"] = _normalized_rank_score(out.get("Effective Reach", pd.Series(index=out.index, dtype="float64")))
    out["_prominence_bonus"] = get_prominence_weight_series(out, selected_prominence_column)
    out["_obs_story_score"] = (
        out["_Group_Count_rank_score"] * 3.0
        + out["_Mentions_rank_score"] * 2.75
        + out["_Impressions_rank_score"] * 1.9
        + out["_Effective_Reach_rank_score"] * 1.9
        + out["_prominence_bonus"]
        + out["_has_url"].astype(float) * 0.2
        + out["_is_online_example"].astype(float) * 0.35
    )
    return out


def _prepare_story_columns(df: pd.DataFrame) -> pd.DataFrame:
    working = df.copy()
    for column in ["Mentions", "Impressions", "Effective Reach", "Group Count"]:
        if column not in working.columns:
            working[column] = 0
        working[column] = pd.to_numeric(working[column], errors="coerce").fillna(0)
    if "Group Count" in working.columns:
        working["Group Count"] = working["Group Count"].where(working["Group Count"] > 0, 1)

    text_columns = ["Headline", "Snippet", "Example Snippet", "URL", "Example URL", "Outlet", "Example Outlet", "Type", "Example Type", "Date"]
    for column in text_columns:
        if column not in working.columns:
            working[column] = ""
        working[column] = _text_series(working, column)

    working["Display URL"] = working["Example URL"].where(working["Example URL"] != "", working["URL"])
    working["Display Outlet"] = working["Example Outlet"].where(working["Example Outlet"] != "", working["Outlet"])
    working["Display Type"] = working["Example Type"].where(working["Example Type"] != "", working["Type"])
    working["Display Snippet"] = working["Example Snippet"].where(working["Example Snippet"] != "", working["Snippet"])
    working["_has_url"] = working["Display URL"].ne("")
    working["_is_online_example"] = working["Display Type"].str.upper().isin({"ONLINE", "ONLINE NEWS", "PRESS RELEASE", "BLOGS"})
    return working


def sentiment_order_for_scale(scale: str) -> list[str]:
    return RAPID_SENTIMENT_ORDER_5_WAY if scale == RAPID_SENTIMENT_SCALE_5_WAY else RAPID_SENTIMENT_ORDER_3_WAY


def sentiment_column_for_scale(scale: str) -> str:
    return "Final Rapid Sentiment 5-Way" if scale == RAPID_SENTIMENT_SCALE_5_WAY else "Final Rapid Sentiment 3-Way"


def effective_sentiment_column_for_scale(scale: str) -> str:
    return "Effective Rapid Sentiment 5-Way" if scale == RAPID_SENTIMENT_SCALE_5_WAY else "Effective Rapid Sentiment 3-Way"


def build_rapid_sentiment_insight_frame(df_unique: pd.DataFrame, *, scale: str) -> pd.DataFrame:
    if df_unique is None or df_unique.empty:
        return pd.DataFrame()
    final = build_final_rapid_sentiment_frame(df_unique)
    effective = build_effective_rapid_sentiment_frame(df_unique)
    working = pd.concat([df_unique.copy(), final, effective], axis=1)
    final_col = sentiment_column_for_scale(scale)
    effective_col = effective_sentiment_column_for_scale(scale)
    working["Rapid Insight Sentiment"] = _text_series(working, final_col).str.upper()
    working["Rapid Insight Machine Sentiment"] = _text_series(working, effective_col).str.upper()
    working = working[working["Rapid Insight Sentiment"].ne("")].copy()
    return working


def build_rapid_sentiment_distribution(df_unique: pd.DataFrame, *, scale: str) -> pd.DataFrame:
    working = build_rapid_sentiment_insight_frame(df_unique, scale=scale)
    order = sentiment_order_for_scale(scale)
    if working.empty:
        return pd.DataFrame({"Sentiment": order, "Count": [0] * len(order), "Grouped Stories": [0] * len(order), "Share": [0.0] * len(order)})
    counts = pd.to_numeric(working.get("Group Count", pd.Series(index=working.index)), errors="coerce").fillna(1)
    distribution_input = pd.DataFrame({"Sentiment": working["Rapid Insight Sentiment"], "Count": counts})
    summed = distribution_input.groupby("Sentiment", dropna=False)["Count"].sum().reset_index()
    grouped = working["Rapid Insight Sentiment"].value_counts().rename_axis("Sentiment").reset_index(name="Grouped Stories")
    out = pd.DataFrame({"Sentiment": order}).merge(summed, on="Sentiment", how="left").merge(grouped, on="Sentiment", how="left")
    out["Count"] = out["Count"].fillna(0).astype(int)
    out["Grouped Stories"] = out["Grouped Stories"].fillna(0).astype(int)
    total = int(out["Count"].sum())
    out["Share"] = out["Count"] / total if total > 0 else 0.0
    return out


def build_rapid_sentiment_observation_payload(
    df_unique: pd.DataFrame,
    *,
    scale: str,
    include_not_relevant: bool,
    per_sentiment_limit: int = RAPID_PRIMARY_EXAMPLE_LIMIT,
    aligned_evidence_limit: int = RAPID_ALIGNED_EVIDENCE_LIMIT,
    selected_prominence_column: str = "",
) -> dict[str, Any]:
    working = build_rapid_sentiment_insight_frame(df_unique, scale=scale)
    if not include_not_relevant:
        working = working[working["Rapid Insight Sentiment"].ne("NOT RELEVANT")].copy()
    if working.empty:
        return {"distribution": [], "examples_by_sentiment": {}, "aligned_evidence_by_sentiment": {}}

    working = _prepare_story_columns(working)
    working = _add_observation_scores(working, selected_prominence_column)
    distribution = build_rapid_sentiment_distribution(working, scale=scale)
    if not include_not_relevant:
        distribution = distribution[distribution["Sentiment"].ne("NOT RELEVANT")].copy()
    distribution = distribution[pd.to_numeric(distribution["Count"], errors="coerce").fillna(0).gt(0)].copy()

    examples_by_sentiment: dict[str, list[dict[str, Any]]] = {}
    aligned_evidence_by_sentiment: dict[str, list[dict[str, Any]]] = {}
    for sentiment, group in working.groupby("Rapid Insight Sentiment", dropna=False):
        ranked = group.sort_values(
            ["_obs_story_score", "Group Count", "Mentions", "Impressions", "Effective Reach"],
            ascending=[False, False, False, False, False],
        )
        primary = ranked.drop_duplicates(subset=["Headline"], keep="first").head(per_sentiment_limit)
        primary_group_ids = {str(row.get("Group ID", "") or "").strip() for _, row in primary.iterrows()}
        primary_headlines = {str(row.get("Headline", "") or "").strip() for _, row in primary.iterrows()}
        examples_by_sentiment[str(sentiment)] = [_story_example(row) for _, row in primary.iterrows()]

        aligned = ranked[ranked["Rapid Insight Machine Sentiment"].eq(ranked["Rapid Insight Sentiment"])].copy()
        if primary_group_ids:
            aligned = aligned[~aligned["Group ID"].astype(str).str.strip().isin(primary_group_ids)].copy()
        if primary_headlines:
            aligned = aligned[~aligned["Headline"].fillna("").astype(str).str.strip().isin(primary_headlines)].copy()
        aligned_evidence_by_sentiment[str(sentiment)] = [
            _story_example(row, compact=True)
            for _, row in aligned.drop_duplicates(subset=["Headline"], keep="first").head(aligned_evidence_limit).iterrows()
        ]

    return {
        "distribution": distribution.to_dict(orient="records"),
        "examples_by_sentiment": examples_by_sentiment,
        "aligned_evidence_by_sentiment": aligned_evidence_by_sentiment,
    }


def tag_column_for_mode(mode: str) -> str:
    return "Final Rapid Tags" if mode == RAPID_TAG_REVIEW_MODE_APPLICABLE else "Final Rapid Best Tag"


def effective_tag_column_for_mode(mode: str) -> str:
    return "Effective Rapid Tags" if mode == RAPID_TAG_REVIEW_MODE_APPLICABLE else "Effective Rapid Best Tag"


def build_rapid_tag_insight_frame(
    df_unique: pd.DataFrame,
    *,
    mode: str,
    tag_definitions: dict[str, str] | None = None,
) -> pd.DataFrame:
    if df_unique is None or df_unique.empty:
        return pd.DataFrame()
    final = build_final_rapid_tag_frame(df_unique, tag_definitions=tag_definitions)
    effective = build_effective_rapid_tag_frame(df_unique, tag_definitions=tag_definitions)
    working = pd.concat([df_unique.copy(), final, effective], axis=1)
    working["Rapid Insight Tags"] = _text_series(working, tag_column_for_mode(mode))
    working["Rapid Insight Machine Tags"] = _text_series(working, effective_tag_column_for_mode(mode))
    working = working[working["Rapid Insight Tags"].ne("")].copy()
    return working


def _expand_tag_rows(working: pd.DataFrame) -> pd.DataFrame:
    tags_expanded = working[["Group ID"]].assign(Tag=working["Rapid Insight Tags"].astype(str).str.split(";")).explode("Tag")
    tags_expanded["Tag"] = tags_expanded["Tag"].fillna("").astype(str).str.strip()
    tags_expanded = tags_expanded[tags_expanded["Tag"].ne("")].copy()
    return tags_expanded


def build_rapid_tag_distribution(
    df_unique: pd.DataFrame,
    *,
    mode: str,
    tag_definitions: dict[str, str] | None = None,
    include_other: bool = True,
) -> pd.DataFrame:
    working = build_rapid_tag_insight_frame(df_unique, mode=mode, tag_definitions=tag_definitions)
    if working.empty:
        return pd.DataFrame(columns=["Tag", "Count", "Grouped Stories", "Share", "Grouped Story Share"])
    working = _prepare_story_columns(working)
    expanded = _expand_tag_rows(working)
    if not include_other:
        expanded = expanded[expanded["Tag"].str.casefold().ne(RESERVED_OTHER_TAG.casefold())].copy()
    if expanded.empty:
        return pd.DataFrame(columns=["Tag", "Count", "Grouped Stories", "Share", "Grouped Story Share"])
    expanded = expanded.merge(
        working[["Group ID", "Group Count"]].drop_duplicates(subset=["Group ID"], keep="first"),
        on="Group ID",
        how="left",
    )
    expanded["Group Count"] = pd.to_numeric(expanded["Group Count"], errors="coerce").fillna(1)
    out = (
        expanded.groupby("Tag", dropna=False)
        .agg(Count=("Group Count", "sum"), Grouped_Stories=("Group ID", "nunique"))
        .reset_index()
        .sort_values(["Count", "Grouped_Stories", "Tag"], ascending=[False, False, True])
        .reset_index(drop=True)
    )
    out["Grouped Stories"] = out.pop("Grouped_Stories").astype(int)
    out["Count"] = out["Count"].astype(int)
    total_count = int(out["Count"].sum())
    denominator = max(1, int(working["Group ID"].nunique()))
    out["Share"] = out["Count"] / float(total_count) if total_count > 0 else 0.0
    out["Grouped Story Share"] = out["Grouped Stories"] / float(denominator)
    return out


def build_rapid_tag_observation_payload(
    df_unique: pd.DataFrame,
    *,
    mode: str,
    include_other: bool,
    tag_definitions: dict[str, str] | None = None,
    per_tag_limit: int = RAPID_PRIMARY_EXAMPLE_LIMIT,
    aligned_evidence_limit: int = RAPID_ALIGNED_EVIDENCE_LIMIT,
    selected_prominence_column: str = "",
) -> dict[str, Any]:
    working = build_rapid_tag_insight_frame(df_unique, mode=mode, tag_definitions=tag_definitions)
    if working.empty:
        return {"distribution": [], "examples_by_tag": {}, "aligned_evidence_by_tag": {}}
    working = _prepare_story_columns(working)
    expanded = _expand_tag_rows(working)
    if not include_other:
        expanded = expanded[expanded["Tag"].str.casefold().ne(RESERVED_OTHER_TAG.casefold())].copy()
    if expanded.empty:
        return {"distribution": [], "examples_by_tag": {}, "aligned_evidence_by_tag": {}}

    distribution = build_rapid_tag_distribution(df_unique, mode=mode, tag_definitions=tag_definitions, include_other=include_other)
    working = _add_observation_scores(working, selected_prominence_column)
    tagged = expanded.merge(working.drop(columns=["Group Count"], errors="ignore"), on="Group ID", how="left")
    tagged = tagged.merge(working[["Group ID", "Group Count"]].drop_duplicates(subset=["Group ID"], keep="first"), on="Group ID", how="left")

    examples_by_tag: dict[str, list[dict[str, Any]]] = {}
    aligned_evidence_by_tag: dict[str, list[dict[str, Any]]] = {}
    for tag_name, group in tagged.groupby("Tag", dropna=False):
        ranked = group.sort_values(
            ["_obs_story_score", "Group Count", "Mentions", "Impressions", "Effective Reach"],
            ascending=[False, False, False, False, False],
        )
        primary = ranked.drop_duplicates(subset=["Headline"], keep="first").head(per_tag_limit)
        primary_group_ids = {str(row.get("Group ID", "") or "").strip() for _, row in primary.iterrows()}
        primary_headlines = {str(row.get("Headline", "") or "").strip() for _, row in primary.iterrows()}
        examples_by_tag[str(tag_name)] = [_story_example(row) for _, row in primary.iterrows()]

        aligned_mask = ranked["Rapid Insight Machine Tags"].fillna("").astype(str).apply(
            lambda value: str(tag_name).casefold() in {tag.casefold() for tag in normalize_tag_list(value.replace(";", ","))}
        )
        aligned = ranked[aligned_mask].copy()
        if primary_group_ids:
            aligned = aligned[~aligned["Group ID"].astype(str).str.strip().isin(primary_group_ids)].copy()
        if primary_headlines:
            aligned = aligned[~aligned["Headline"].fillna("").astype(str).str.strip().isin(primary_headlines)].copy()
        aligned_evidence_by_tag[str(tag_name)] = [
            _story_example(row, compact=True)
            for _, row in aligned.drop_duplicates(subset=["Headline"], keep="first").head(aligned_evidence_limit).iterrows()
        ]

    return {
        "distribution": distribution.to_dict(orient="records"),
        "examples_by_tag": examples_by_tag,
        "aligned_evidence_by_tag": aligned_evidence_by_tag,
    }


def _story_example(row: pd.Series, *, compact: bool = False) -> dict[str, Any]:
    item = {
        "group_id": row.get("Group ID", ""),
        "headline": row.get("Headline", ""),
        "outlet": row.get("Display Outlet", ""),
        "date": row.get("Date", ""),
        "url": row.get("Display URL", ""),
        "example_type": row.get("Display Type", ""),
        "mentions": int(row.get("Mentions", 0) or 0),
        "group_count": int(row.get("Group Count", 0) or 0),
    }
    if not compact:
        item["impressions"] = int(row.get("Impressions", 0) or 0)
        item["effective_reach"] = int(row.get("Effective Reach", 0) or 0)
        item["snippet"] = _truncate_text(row.get("Display Snippet", ""), 420)
    return item


def rapid_observation_fingerprint(payload: dict[str, Any], *, settings: dict[str, Any], analysis_context: str) -> str:
    raw = json.dumps(
        {"payload": payload, "settings": settings, "analysis_context": analysis_context},
        sort_keys=True,
        ensure_ascii=True,
        default=str,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def generate_rapid_sentiment_observations(
    df_unique: pd.DataFrame,
    *,
    client_name: str,
    scale: str,
    include_not_relevant: bool,
    api_key: str,
    model: str = DEFAULT_SENTIMENT_OBSERVATION_MODEL,
    analysis_context: str = "",
    selected_prominence_column: str = "",
) -> tuple[dict[str, Any], int, int, str]:
    payload = build_rapid_sentiment_observation_payload(
        df_unique,
        scale=scale,
        include_not_relevant=include_not_relevant,
        selected_prominence_column=selected_prominence_column,
    )
    prompt = build_sentiment_observation_prompt(client_name, scale, include_not_relevant, payload, analysis_context)
    client = OpenAI(api_key=api_key)
    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": "You write concise, neutral media-intelligence summaries."},
            {"role": "user", "content": prompt},
        ],
        reasoning={"effort": "low"},
        text={
            "verbosity": "low",
            "format": {
                "type": "json_schema",
                "name": "sentiment_observations",
                "strict": True,
                "schema": build_sentiment_observation_schema(payload),
            },
        },
    )
    add_api_usage(response, model)
    in_tok, out_tok = extract_usage_tokens(response)
    parsed = _extract_json_payload(getattr(response, "output_text", "") or "")
    if parsed is None:
        raise ValueError("Model did not return valid JSON for sentiment observations.")
    parsed["_examples_by_sentiment"] = payload.get("examples_by_sentiment", {})
    fingerprint = rapid_observation_fingerprint(
        payload,
        settings={"family": "sentiment", "scale": scale, "include_not_relevant": include_not_relevant},
        analysis_context=analysis_context,
    )
    return parsed, in_tok, out_tok, fingerprint


def generate_rapid_tag_observations(
    df_unique: pd.DataFrame,
    *,
    client_name: str,
    mode: str,
    include_other: bool,
    api_key: str,
    model: str = DEFAULT_TAGGING_OBSERVATION_MODEL,
    analysis_context: str = "",
    tag_definitions: dict[str, str] | None = None,
    selected_prominence_column: str = "",
) -> tuple[dict[str, Any], int, int, str]:
    payload = build_rapid_tag_observation_payload(
        df_unique,
        mode=mode,
        include_other=include_other,
        tag_definitions=tag_definitions,
        selected_prominence_column=selected_prominence_column,
    )
    prompt = build_tag_observation_prompt(client_name, include_other, payload, analysis_context)
    client = OpenAI(api_key=api_key)
    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": "You write concise, neutral media-intelligence summaries."},
            {"role": "user", "content": prompt},
        ],
        reasoning={"effort": "low"},
        text={
            "verbosity": "low",
            "format": {
                "type": "json_schema",
                "name": "tag_observations",
                "strict": True,
                "schema": build_tag_observation_schema(payload),
            },
        },
    )
    add_api_usage(response, model)
    in_tok, out_tok = extract_usage_tokens(response)
    parsed = _extract_json_payload(getattr(response, "output_text", "") or "")
    if parsed is None:
        raise ValueError("Model did not return valid JSON for tag observations.")
    parsed["_examples_by_tag"] = payload.get("examples_by_tag", {})
    fingerprint = rapid_observation_fingerprint(
        payload,
        settings={"family": "tagging", "mode": mode, "include_other": include_other, "tag_definitions": tag_definitions or {}},
        analysis_context=analysis_context,
    )
    return parsed, in_tok, out_tok, fingerprint
