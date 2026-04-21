from __future__ import annotations

import json
import re
from typing import Any

import pandas as pd
from openai import OpenAI

from processing.top_stories import consolidate_top_story_candidates, parse_source_group_ids
from utils.api_meter import add_api_usage, extract_usage_tokens


METRIC_FIELD_MAP = {
    "Mentions": "Mentions",
    "Impressions": "Impressions",
    "Effective Reach": "Effective Reach",
}

DEFAULT_REGIONS_OBSERVATION_MODEL = "gpt-5.4-mini"
REGIONS_DEFAULT_EXCLUDED_FLAGS = [
    "Press Release",
    "Financial Outlet",
    "Advertorial",
]
REGIONS_HIDDEN_FILTER_FLAGS = {
    "Aggregators",
    "Good outlets",
}


REGIONS_REQUIRED_COLUMNS = [
    "Group ID",
    "Source Outlet",
    "Outlet",
    "Headline",
    "Snippet",
    "Media Type",
    "Coverage Flags",
    "Country",
    "Prov/State",
    "City",
    "Mentions",
    "Impressions",
    "Effective Reach",
    "URL",
    "Date",
]


def init_regions_state(session_state) -> None:
    session_state.setdefault("regions_step", "Setup")
    session_state.setdefault("regions_metric", "Mentions")
    session_state.setdefault("regions_geo_basis", "Outlet location")
    session_state.setdefault("regions_analysis_levels", ["Countries", "States / Provinces", "Cities"])
    session_state.setdefault("regions_exclude_coverage_flags", REGIONS_DEFAULT_EXCLUDED_FLAGS.copy())
    session_state.setdefault("regions_flag_defaults_initialized", False)
    session_state.setdefault("regions_include_countries", [])
    session_state.setdefault("regions_exclude_countries", [])
    session_state.setdefault("regions_prepared", False)
    session_state.setdefault("regions_generated_output", {})
    session_state.setdefault("regions_generated_signature", None)


# def _normalize_text_column(series: pd.Series) -> pd.Series:
#     return series.fillna("").astype(str).str.strip()

def _normalize_text_column(series: pd.Series) -> pd.Series:
    s = pd.Series(series)
    return s.astype("string").fillna("").str.strip()


def build_regions_source_df(
    df_traditional: pd.DataFrame,
    outlet_rollup_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    if df_traditional is None or df_traditional.empty:
        return pd.DataFrame(columns=REGIONS_REQUIRED_COLUMNS)

    outlet_map = {
        str(k).strip(): str(v).strip()
        for k, v in (outlet_rollup_map or {}).items()
        if str(k).strip()
    }

    working = df_traditional.copy()
    if "Group ID" in working.columns:
        working["Group ID"] = _normalize_text_column(working.get("Group ID", pd.Series(index=working.index, dtype="object")))
    else:
        fallback_ids = working.get("Headline", pd.Series(index=working.index, dtype="object")).fillna("").astype(str).str.strip()
        fallback_ids = fallback_ids.replace("", pd.NA)
        fallback_ids = fallback_ids.fillna(pd.Series(index=working.index, data=[f"ROW::{idx}" for idx in working.index]))
        working["Group ID"] = fallback_ids.astype(str)
    working["Source Outlet"] = _normalize_text_column(working.get("Outlet", pd.Series(index=working.index, dtype="object")))
    working["Outlet"] = working["Source Outlet"].map(lambda value: outlet_map.get(value, value))
    media_type_source = "Type" if "Type" in working.columns else "Media Type"
    working["Media Type"] = _normalize_text_column(working.get(media_type_source, pd.Series(index=working.index, dtype="object")))
    for col in ["Headline", "Snippet", "Coverage Flags", "Country", "Prov/State", "City", "URL"]:
        working[col] = _normalize_text_column(working.get(col, pd.Series(index=working.index, dtype="object")))

    working["Mentions"] = pd.to_numeric(
        working.get("Mentions", pd.Series(index=working.index, data=1)),
        errors="coerce",
    ).fillna(1)
    if (working["Mentions"] <= 0).all():
        working["Mentions"] = 1
    working["Impressions"] = pd.to_numeric(
        working.get("Impressions", pd.Series(index=working.index, data=0)),
        errors="coerce",
    ).fillna(0)
    effective_source = "Effective Reach" if "Effective Reach" in working.columns else "Effective_Reach"
    working["Effective Reach"] = pd.to_numeric(
        working.get(effective_source, pd.Series(index=working.index, data=0)),
        errors="coerce",
    ).fillna(0)
    working["Date"] = working.get("Date", pd.Series(index=working.index, dtype="object"))

    return working[REGIONS_REQUIRED_COLUMNS].copy()


def split_coverage_flags(value: object) -> list[str]:
    raw = str(value or "").strip()
    if not raw:
        return []
    parts = re.split(r"[;,]\s*|\|\s*", raw)
    return [part.strip() for part in parts if part.strip()]


def filter_regions_df(
    df: pd.DataFrame,
    exclude_coverage_flags: list[str] | None = None,
    include_countries: list[str] | None = None,
    exclude_countries: list[str] | None = None,
) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=REGIONS_REQUIRED_COLUMNS)

    filtered = df.copy()
    blocked_flags = {str(v).strip() for v in exclude_coverage_flags or [] if str(v).strip()}
    if blocked_flags:
        filtered = filtered[
            ~filtered["Coverage Flags"].apply(
                lambda value: any(flag in blocked_flags for flag in split_coverage_flags(value))
            )
        ].copy()

    allowed_countries = [str(v).strip() for v in include_countries or [] if str(v).strip()]
    if allowed_countries:
        filtered = filtered[filtered["Country"].isin(allowed_countries)].copy()

    blocked_countries = {str(v).strip() for v in exclude_countries or [] if str(v).strip()}
    if blocked_countries:
        filtered = filtered[~filtered["Country"].isin(blocked_countries)].copy()

    usable_mask = (
        filtered["Country"].ne("")
        | filtered["Prov/State"].ne("")
        | filtered["City"].ne("")
    )
    filtered = filtered[usable_mask].copy()

    return filtered.reset_index(drop=True)


def build_regions_health_summary(df: pd.DataFrame) -> dict[str, int]:
    if df is None or df.empty:
        return {
            "rows": 0,
            "usable_rows": 0,
            "missing_rows": 0,
            "countries": 0,
            "states": 0,
            "cities": 0,
        }

    usable_mask = (
        df["Country"].ne("")
        | df["Prov/State"].ne("")
        | df["City"].ne("")
    )
    return {
        "rows": int(len(df)),
        "usable_rows": int(usable_mask.sum()),
        "missing_rows": int((~usable_mask).sum()),
        "countries": int(df["Country"].replace("", pd.NA).dropna().nunique()),
        "states": int(df["Prov/State"].replace("", pd.NA).dropna().nunique()),
        "cities": int(df["City"].replace("", pd.NA).dropna().nunique()),
    }


def _build_state_label(row: pd.Series) -> str:
    state = str(row.get("Prov/State", "") or "").strip()
    country = str(row.get("Country", "") or "").strip()
    if not state:
        return ""
    return f"{state}, {country}" if country else state


def _build_city_label(row: pd.Series) -> str:
    city = str(row.get("City", "") or "").strip()
    state = str(row.get("Prov/State", "") or "").strip()
    country = str(row.get("Country", "") or "").strip()
    if not city:
        return ""
    if state:
        return f"{city}, {state}"
    if country:
        return f"{city}, {country}"
    return city


def build_region_rankings(df: pd.DataFrame, level: str, metric_label: str = "Mentions") -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["Region", "Mentions", "Impressions", "Effective Reach", "Outlet Count", "Example Outlets"])

    working = df.copy()
    if level == "Country":
        working["Region"] = working["Country"]
    elif level == "State / Province":
        working["Region"] = working.apply(_build_state_label, axis=1)
    elif level == "City":
        working["Region"] = working.apply(_build_city_label, axis=1)
    else:
        raise ValueError(f"Unsupported region level: {level}")

    working = working[working["Region"].ne("")].copy()
    if working.empty:
        return pd.DataFrame(columns=["Region", "Mentions", "Impressions", "Effective Reach", "Outlet Count", "Example Outlets"])

    grouped = (
        working.groupby("Region", as_index=False)
        .agg(
            Mentions=("Mentions", "sum"),
            Impressions=("Impressions", "sum"),
            **{"Effective Reach": ("Effective Reach", "sum")},
            **{"Outlet Count": ("Outlet", "nunique")},
        )
        .sort_values(
            [METRIC_FIELD_MAP.get(metric_label, "Mentions"), "Mentions", "Impressions", "Effective Reach", "Region"],
            ascending=[False, False, False, False, True],
        )
        .reset_index(drop=True)
    )

    outlet_examples = (
        working.groupby(["Region", "Outlet"], as_index=False)["Mentions"]
        .sum()
        .sort_values(["Region", "Mentions", "Outlet"], ascending=[True, False, True])
    )
    example_map = (
        outlet_examples.groupby("Region")["Outlet"]
        .apply(lambda values: ", ".join(list(values.head(3))))
        .to_dict()
    )
    grouped["Example Outlets"] = grouped["Region"].map(lambda region: example_map.get(region, ""))
    return grouped


def _assign_region_column(df: pd.DataFrame, level: str) -> pd.DataFrame:
    working = df.copy()
    if level == "Country":
        working["Region"] = working["Country"]
    elif level == "State / Province":
        working["Region"] = working.apply(_build_state_label, axis=1)
    elif level == "City":
        working["Region"] = working.apply(_build_city_label, axis=1)
    else:
        raise ValueError(f"Unsupported region level: {level}")
    return working


def _extract_response_text(response) -> str:
    if hasattr(response, "output_text") and response.output_text:
        return response.output_text

    try:
        parts = []
        for item in response.output:
            if getattr(item, "type", None) != "message":
                continue
            for content in getattr(item, "content", []):
                if getattr(content, "type", None) == "output_text":
                    parts.append(content.text)
        return "\n".join(parts).strip()
    except Exception:
        return ""


def _top_values(series: pd.Series, limit: int = 5) -> list[str]:
    cleaned = series.fillna("").astype(str).str.strip()
    cleaned = cleaned[cleaned != ""]
    if cleaned.empty:
        return []
    return cleaned.value_counts().head(limit).index.tolist()


def _first_nonblank(values: pd.Series) -> str:
    for value in values.tolist():
        cleaned = str(value or "").strip()
        if cleaned:
            return cleaned
    return ""


def _trim_text(value: object, limit: int = 240) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _format_date_value(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    if hasattr(value, "strftime"):
        try:
            return value.strftime("%Y-%m-%d")
        except Exception:
            return str(value)
    return str(value or "").strip()


def _sanitize_observation_text(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    text = re.sub(r"\bbuckets?\b", "regions", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip()


def _choose_story_label(headline: object, snippet: object, group_id: object) -> str:
    headline_text = str(headline or "").strip()
    if headline_text:
        return headline_text
    snippet_text = _trim_text(snippet, limit=140)
    if snippet_text:
        return snippet_text
    fallback = str(group_id or "").strip()
    return fallback or "Grouped story"


def _metric_value(series: pd.Series, metric_label: str) -> float:
    metric_col = METRIC_FIELD_MAP.get(metric_label, "Mentions")
    return float(series.get(metric_col, 0) or 0)


def _share_pct(value: float, total: float) -> float:
    if total <= 0:
        return 0.0
    return round((float(value) / float(total)) * 100, 1)


def _build_top_media_type_context(working: pd.DataFrame, metric_label: str, limit: int = 4) -> list[dict[str, Any]]:
    media_df = working[working["Media Type"].ne("")].copy()
    if media_df.empty:
        return []

    metric_col = METRIC_FIELD_MAP.get(metric_label, "Mentions")
    total_metric = float(media_df[metric_col].sum()) if metric_col in media_df.columns else 0.0
    grouped = (
        media_df.groupby("Media Type", as_index=False)
        .agg(
            Mentions=("Mentions", "sum"),
            Impressions=("Impressions", "sum"),
            **{"Effective Reach": ("Effective Reach", "sum")},
            **{"Outlet Count": ("Outlet", "nunique")},
        )
        .sort_values(["Mentions", "Impressions", "Effective Reach", "Media Type"], ascending=[False, False, False, True])
        .head(limit)
        .reset_index(drop=True)
    )
    grouped = grouped.sort_values(
        [metric_col, "Mentions", "Impressions", "Effective Reach", "Media Type"],
        ascending=[False, False, False, False, True],
    ).reset_index(drop=True)
    grouped["Metric Share Pct"] = grouped[metric_col].apply(lambda value: _share_pct(value, total_metric))
    return grouped.to_dict(orient="records")


def _build_region_story_candidates(working: pd.DataFrame) -> pd.DataFrame:
    if working.empty:
        return pd.DataFrame(
            columns=[
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
        )

    rows: list[dict[str, Any]] = []
    for group_id, group in working.groupby("Group ID", dropna=False):
        if group.empty:
            continue

        group_working = group.copy()
        group_working["_snippet_len"] = group_working["Snippet"].fillna("").astype(str).str.len()
        group_working["_has_url"] = group_working["URL"].fillna("").astype(str).str.strip().ne("")
        group_working["_headline_len"] = group_working["Headline"].fillna("").astype(str).str.len()
        best_row = group_working.sort_values(
            by=["_snippet_len", "Mentions", "Impressions", "_has_url", "_headline_len", "Date"],
            ascending=[False, False, False, False, False, False],
            na_position="last",
        ).iloc[0]

        group_id_str = "" if pd.isna(group_id) else str(group_id).strip()
        rows.append(
            {
                "Group ID": group_id,
                "Headline": best_row.get("Headline", ""),
                "Date": best_row.get("Date", pd.NaT),
                "Mentions": int(pd.to_numeric(group["Mentions"], errors="coerce").fillna(0).sum()),
                "Impressions": int(pd.to_numeric(group["Impressions"], errors="coerce").fillna(0).sum()),
                "Effective Reach": int(pd.to_numeric(group["Effective Reach"], errors="coerce").fillna(0).sum()),
                "Example Outlet": best_row.get("Outlet", ""),
                "Example URL": best_row.get("URL", ""),
                "Example Type": best_row.get("Media Type", ""),
                "Example Snippet": best_row.get("Snippet", ""),
                "Source Group IDs": group_id_str,
            }
        )

    if not rows:
        return pd.DataFrame()

    candidates = pd.DataFrame(rows)
    candidates["Date"] = pd.to_datetime(candidates["Date"], errors="coerce").dt.date
    candidates["Mentions"] = pd.to_numeric(candidates["Mentions"], errors="coerce").fillna(0).astype(int)
    candidates["Impressions"] = pd.to_numeric(candidates["Impressions"], errors="coerce").fillna(0).astype(int)
    candidates["Effective Reach"] = pd.to_numeric(candidates["Effective Reach"], errors="coerce").fillna(0).astype(int)
    candidates["Source Group IDs"] = candidates["Source Group IDs"].fillna("").astype(str)
    return consolidate_top_story_candidates(candidates)


def _build_story_outlet_context_map(working: pd.DataFrame, story_candidates: pd.DataFrame) -> dict[str, dict[str, Any]]:
    if working.empty or story_candidates.empty:
        return {}

    working_ids = working.copy()
    working_ids["_group_id_key"] = working_ids["Group ID"].fillna("").astype(str).str.strip()
    outlet_map: dict[str, dict[str, Any]] = {}

    for _, row in story_candidates.iterrows():
        story_key = str(row.get("Group ID", "") or "").strip()
        source_ids = parse_source_group_ids(row.get("Source Group IDs", ""), fallback_group_id=row.get("Group ID"))
        if not source_ids:
            outlet_map[story_key] = {"example_outlets": "", "outlet_count": 0}
            continue

        story_rows = working_ids[working_ids["_group_id_key"].isin(source_ids)].copy()
        if story_rows.empty:
            outlet_map[story_key] = {"example_outlets": "", "outlet_count": 0}
            continue

        ranked_outlets = (
            story_rows.groupby("Outlet", as_index=False)["Mentions"]
            .sum()
            .sort_values(["Mentions", "Outlet"], ascending=[False, True])
        )
        outlet_map[story_key] = {
            "example_outlets": ", ".join(ranked_outlets["Outlet"].head(3).tolist()),
            "outlet_count": int(ranked_outlets["Outlet"].nunique()),
        }

    return outlet_map


def _build_top_story_context(working: pd.DataFrame, metric_label: str, limit: int = 10) -> list[dict[str, Any]]:
    story_candidates = _build_region_story_candidates(working)
    if story_candidates.empty:
        return []

    metric_col = METRIC_FIELD_MAP.get(metric_label, "Mentions")
    total_metric = float(story_candidates[metric_col].sum()) if metric_col in story_candidates.columns else 0.0
    outlet_context_map = _build_story_outlet_context_map(working, story_candidates)

    grouped = story_candidates.copy()
    grouped["Headline"] = grouped.apply(
        lambda row: _choose_story_label(row.get("Headline", ""), row.get("Example Snippet", ""), row.get("Group ID", "")),
        axis=1,
    )
    grouped["Snippet"] = grouped["Example Snippet"].map(_trim_text)
    grouped["Outlet Count"] = grouped["Group ID"].map(
        lambda value: int((outlet_context_map.get(str(value), {}) or {}).get("outlet_count", 0))
    )
    grouped["Date"] = grouped["Date"].map(_format_date_value)
    grouped["Example Outlets"] = grouped["Group ID"].map(
        lambda value: str((outlet_context_map.get(str(value), {}) or {}).get("example_outlets", ""))
    )
    grouped["Metric Share Pct"] = grouped[metric_col].apply(lambda value: _share_pct(value, total_metric))
    grouped = grouped.sort_values(
        [metric_col, "Mentions", "Impressions", "Effective Reach", "Headline"],
        ascending=[False, False, False, False, True],
    ).head(limit).reset_index(drop=True)
    return grouped.to_dict(orient="records")


def build_region_profile_context(
    df: pd.DataFrame,
    level: str,
    region_name: str,
    metric_label: str,
) -> dict[str, Any]:
    if df is None or df.empty:
        return {}

    working = _assign_region_column(df, level)
    working = working[working["Region"].eq(str(region_name or "").strip())].copy()
    if working.empty:
        return {}

    metric_col = METRIC_FIELD_MAP.get(metric_label, "Mentions")
    total_metric = float(working[metric_col].sum()) if metric_col in working.columns else 0.0
    outlet_rank = (
        working.groupby("Outlet", as_index=False)
        .agg(
            Mentions=("Mentions", "sum"),
            Impressions=("Impressions", "sum"),
            **{"Effective Reach": ("Effective Reach", "sum")},
            **{"Row Count": ("Headline", "size")},
        )
        .sort_values([metric_col, "Mentions", "Impressions", "Effective Reach", "Outlet"], ascending=[False, False, False, False, True])
        .head(5)
    )
    if not outlet_rank.empty:
        outlet_rank["Metric Share Pct"] = outlet_rank[metric_col].apply(lambda value: _share_pct(value, total_metric))

    top_stories = _build_top_story_context(working, metric_label=metric_label, limit=10)
    top_media_types = _build_top_media_type_context(working, metric_label=metric_label, limit=5)
    story_count = len(_build_region_story_candidates(working))
    unique_outlet_count = int(working["Outlet"].replace("", pd.NA).dropna().nunique())
    top_outlet_share_pct = float(outlet_rank["Metric Share Pct"].iloc[0]) if not outlet_rank.empty else 0.0
    top_three_outlet_share_pct = round(float(outlet_rank[metric_col].head(3).sum()) / total_metric * 100, 1) if total_metric > 0 and not outlet_rank.empty else 0.0
    top_story_share_pct = float(top_stories[0].get("Metric Share Pct", 0.0)) if top_stories else 0.0
    top_three_story_share_pct = round(sum(float(item.get(metric_col, 0) or 0) for item in top_stories[:3]) / total_metric * 100, 1) if total_metric > 0 and top_stories else 0.0

    return {
        "region": region_name,
        "mentions": int(working["Mentions"].sum()),
        "impressions": int(working["Impressions"].sum()),
        "effective_reach": int(working["Effective Reach"].sum()),
        "outlet_count": unique_outlet_count,
        "story_count": story_count,
        "top_media_types": top_media_types,
        "top_outlets": outlet_rank.to_dict(orient="records"),
        "top_stories": top_stories,
        "concentration_summary": {
            "top_outlet_share_pct": top_outlet_share_pct,
            "top_3_outlet_share_pct": top_three_outlet_share_pct,
            "top_story_share_pct": top_story_share_pct,
            "top_3_story_share_pct": top_three_story_share_pct,
            "unique_outlet_count": unique_outlet_count,
            "unique_story_count": story_count,
        },
    }


def build_region_level_prompt_payload(
    filtered_df: pd.DataFrame,
    ranking_df: pd.DataFrame,
    level: str,
    metric_label: str,
    tail_n: int = 7,
) -> dict[str, Any]:
    top_n_profiles = 3
    top_regions = ranking_df.head(top_n_profiles)
    tail_regions = ranking_df.iloc[top_n_profiles : top_n_profiles + tail_n]
    metric_col = METRIC_FIELD_MAP.get(metric_label, "Mentions")
    total_metric = float(ranking_df[metric_col].sum()) if metric_col in ranking_df.columns else 0.0
    top_metric = float(top_regions[metric_col].iloc[0]) if not top_regions.empty else 0.0
    top_three_metric = float(top_regions[metric_col].sum()) if not top_regions.empty else 0.0

    return {
        "level": level,
        "metric": metric_label,
        "level_summary": {
            "regions_in_view": int(len(ranking_df)),
            "total_metric": int(total_metric),
            "top_region_share_pct": round((top_metric / total_metric) * 100, 1) if total_metric > 0 else 0.0,
            "top_three_share_pct": round((top_three_metric / total_metric) * 100, 1) if total_metric > 0 else 0.0,
            "single_region_in_view": bool(len(ranking_df) == 1),
            "single_region_name": str(ranking_df.iloc[0]["Region"]).strip() if len(ranking_df) == 1 else "",
        },
        "top_table": ranking_df.head(10)[
            ["Region", "Mentions", "Impressions", "Effective Reach", "Outlet Count", "Example Outlets"]
        ].to_dict(orient="records"),
        "top_regions": top_regions[
            ["Region", "Mentions", "Impressions", "Effective Reach", "Outlet Count", "Example Outlets"]
        ].to_dict(orient="records"),
        "tail_table": tail_regions[
            ["Region", "Mentions", "Impressions", "Effective Reach", "Outlet Count", "Example Outlets"]
        ].to_dict(orient="records"),
    }


def _build_representative_rows_context(
    filtered_df: pd.DataFrame,
    level: str,
    region_name: str,
    limit: int = 10,
) -> list[dict[str, Any]]:
    rows_df = build_region_example_rows(filtered_df, level, region_name, limit=limit)
    if rows_df.empty:
        return []
    working = rows_df.copy()
    for col in ["Mentions", "Impressions", "Effective Reach"]:
        if col in working.columns:
            working[col] = working[col].fillna(0).astype(int)
    return working.to_dict(orient="records")


def generate_region_level_overview(
    *,
    client_name: str,
    analysis_context: str,
    level: str,
    metric_label: str,
    ranking_df: pd.DataFrame,
    filtered_df: pd.DataFrame,
    api_key: str,
    model: str = DEFAULT_REGIONS_OBSERVATION_MODEL,
) -> tuple[dict[str, Any], int, int]:
    if ranking_df is None or ranking_df.empty:
        return {
            "overall_observation": "",
            "tail_observation": "",
        }, 0, 0

    payload = build_region_level_prompt_payload(
        filtered_df=filtered_df,
        ranking_df=ranking_df,
        level=level,
        metric_label=metric_label,
    )

    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "overall_observation": {"type": "string"},
            "tail_observation": {"type": "string"},
        },
        "required": ["overall_observation", "tail_observation"],
    }

    single_region_name = str(payload.get("level_summary", {}).get("single_region_name", "") or "").strip()
    single_region_instruction = ""
    if single_region_name:
        single_region_instruction = (
            f"- Only one region is in view here: {single_region_name}. Start the overall observation plainly with "
            f"'All coverage in this view came from {single_region_name}.' or an equally direct sentence, then move "
            "straight into what appears to be driving coverage there.\n"
            "- Do not frame that as a limitation, caveat, or apology.\n"
        )

    prompt = (
        "Write concise regional media-analysis observations for reporting use.\n\n"
        f"Client: {client_name or '[not provided]'}\n"
        f"Analysis context:\n{analysis_context or '[none provided]'}\n\n"
        f"Region level: {level}\n"
        f"Ranking metric: {metric_label}\n\n"
        "Your job is to explain the big regional pattern at this level.\n"
        "Focus on things like:\n"
        "- whether attention is concentrated in a few leading regions or spread more broadly\n"
        "- whether the next tier suggests broader pickup or a steep drop-off\n"
        "- what kinds of outlets appear to be shaping the leading regions\n\n"
        "Evidence rules:\n"
        "- Use only the structured evidence provided.\n"
        "- Do not infer a dominant storyline from this level-wide payload alone.\n"
        f"{single_region_instruction}"
        "- Keep the overall observation focused on distribution and broad character, not specific story claims.\n\n"
        "Output goals:\n"
        "- Overall observation: explain the big regional pattern at this level.\n"
        "- Tail observation: explain what the next tier of regions suggests about broader spread, syndication, or long-tail pickup.\n\n"
        "- If there is no meaningful next tier of regions, return an empty string for tail_observation.\n"
        "Do not simply restate that one region ranks above another.\n"
        "Do not use words like 'bucket'.\n"
        "Do not narrate the table line by line.\n"
        "Avoid generic phrasing like 'visibility is led by'.\n"
        "Keep each blurb compact, specific, and report-ready.\n"
        "If the evidence is thin, be cautious and say so plainly.\n\n"
        f"Structured region data:\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )

    client = OpenAI(api_key=api_key)
    response = client.responses.create(
        model=model,
        temperature=0,
        input=[
            {
                "role": "system",
                "content": (
                    "You are a precise media analyst writing regional coverage insights. "
                    "Return only structured JSON."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        text={
            "verbosity": "low",
            "format": {
                "type": "json_schema",
                "name": "regional_observations",
                "strict": True,
                "schema": schema,
            },
        },
    )

    add_api_usage(response, model)
    in_tok, out_tok = extract_usage_tokens(response)
    raw_text = _extract_response_text(response).strip()
    if not raw_text:
        raise ValueError("No structured output text was returned.")

    parsed = json.loads(raw_text)
    return {
        "overall_observation": _sanitize_observation_text(parsed.get("overall_observation", "")),
        "tail_observation": _sanitize_observation_text(parsed.get("tail_observation", "")),
    }, in_tok, out_tok


def generate_region_profile_observation(
    *,
    client_name: str,
    analysis_context: str,
    level: str,
    metric_label: str,
    region_profile: dict[str, Any],
    level_payload: dict[str, Any],
    representative_rows: list[dict[str, Any]],
    api_key: str,
    model: str = DEFAULT_REGIONS_OBSERVATION_MODEL,
) -> tuple[str, int, int]:
    region_name = str(region_profile.get("region", "") or "").strip()
    if not region_name:
        return "", 0, 0

    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "blurb": {"type": "string"},
        },
        "required": ["blurb"],
    }

    prompt = (
        "Write one concise regional profile blurb for reporting use.\n\n"
        f"Client: {client_name or '[not provided]'}\n"
        f"Analysis context:\n{analysis_context or '[none provided]'}\n\n"
        f"Region level: {level}\n"
        f"Region name: {region_name}\n"
        f"Ranking metric: {metric_label}\n\n"
        "Your job is to explain what appears to be driving visibility in this specific region.\n"
        "Focus on:\n"
        "- leading outlets\n"
        "- dominant media types\n"
        "- whether story attention looks concentrated or varied\n"
        "- the strongest recurring themes visible in the grouped stories and representative rows\n\n"
        "Evidence rules:\n"
        "- Treat concentration_summary as the guardrails for how strong your claims can be.\n"
        "- Do not say a story dominates unless the shares are clearly high and the story appears across multiple rows and multiple outlets.\n"
        "- If the top story appears only once, treat it as one example rather than a dominant driver.\n"
        "- Do not overstate trend, momentum, or audience intent.\n"
        "- If the evidence is mixed, say the coverage is mixed, varied, or distributed.\n\n"
        "Style:\n"
        "- One compact paragraph.\n"
        "- Specific, analyst-facing, and report-ready.\n"
        "- Do not just paraphrase the ranking table.\n\n"
        f"Level context:\n{json.dumps(level_payload, ensure_ascii=False, indent=2)}\n\n"
        f"Region profile context:\n{json.dumps(region_profile, ensure_ascii=False, indent=2)}\n\n"
        f"Representative rows:\n{json.dumps(representative_rows, ensure_ascii=False, indent=2)}"
    )

    client = OpenAI(api_key=api_key)
    response = client.responses.create(
        model=model,
        temperature=0,
        input=[
            {
                "role": "system",
                "content": (
                    "You are a precise media analyst writing one regional profile blurb. "
                    "Return only structured JSON."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        text={
            "verbosity": "low",
            "format": {
                "type": "json_schema",
                "name": "regional_profile_observation",
                "strict": True,
                "schema": schema,
            },
        },
    )

    add_api_usage(response, model)
    in_tok, out_tok = extract_usage_tokens(response)
    raw_text = _extract_response_text(response).strip()
    if not raw_text:
        raise ValueError("No structured output text was returned.")
    parsed = json.loads(raw_text)
    return _sanitize_observation_text(parsed.get("blurb", "")), in_tok, out_tok


def generate_regions_level_output(
    *,
    client_name: str,
    analysis_context: str,
    metric_label: str,
    filtered_df: pd.DataFrame,
    level_label: str,
    level_key: str,
    ranking_df: pd.DataFrame,
    api_key: str,
    model: str = DEFAULT_REGIONS_OBSERVATION_MODEL,
) -> tuple[dict[str, Any], int, int]:
    if ranking_df is None or ranking_df.empty:
        return {
            "overall_observation": "",
            "top_region_profiles": [],
            "tail_observation": "",
        }, 0, 0

    level_payload = build_region_level_prompt_payload(
        filtered_df=filtered_df,
        ranking_df=ranking_df,
        level=level_key,
        metric_label=metric_label,
    )

    overview_payload, total_in_tokens, total_out_tokens = generate_region_level_overview(
        client_name=client_name,
        analysis_context=analysis_context,
        level=level_key,
        metric_label=metric_label,
        ranking_df=ranking_df,
        filtered_df=filtered_df,
        api_key=api_key,
        model=model,
    )

    top_region_profiles: list[dict[str, str]] = []
    for region_name in ranking_df.head(3)["Region"].tolist():
        region_profile = build_region_profile_context(filtered_df, level_key, str(region_name), metric_label)
        representative_rows = _build_representative_rows_context(filtered_df, level_key, str(region_name), limit=10)
        blurb, in_tok, out_tok = generate_region_profile_observation(
            client_name=client_name,
            analysis_context=analysis_context,
            level=level_key,
            metric_label=metric_label,
            region_profile=region_profile,
            level_payload=level_payload,
            representative_rows=representative_rows,
            api_key=api_key,
            model=model,
        )
        total_in_tokens += in_tok
        total_out_tokens += out_tok
        top_region_profiles.append({"region": str(region_name), "blurb": blurb})

    output = {
        "label": level_label,
        "overall_observation": overview_payload.get("overall_observation", ""),
        "top_region_profiles": top_region_profiles,
        "tail_observation": overview_payload.get("tail_observation", ""),
    }
    return output, total_in_tokens, total_out_tokens


def generate_regions_insight_output(
    *,
    client_name: str,
    analysis_context: str,
    metric_label: str,
    filtered_df: pd.DataFrame,
    level_label: str,
    level_key: str,
    ranking_df: pd.DataFrame,
    api_key: str,
    model: str = DEFAULT_REGIONS_OBSERVATION_MODEL,
) -> tuple[dict[str, Any], int, int]:
    output, total_in_tokens, total_out_tokens = generate_regions_level_output(
        client_name=client_name,
        analysis_context=analysis_context,
        metric_label=metric_label,
        filtered_df=filtered_df,
        level_label=level_label,
        level_key=level_key,
        ranking_df=ranking_df,
        api_key=api_key,
        model=model,
    )
    return {level_label: output}, total_in_tokens, total_out_tokens


def build_region_example_rows(df: pd.DataFrame, level: str, region_name: str, limit: int = 5) -> pd.DataFrame:
    if df is None or df.empty or not str(region_name or "").strip():
        return pd.DataFrame(columns=["Outlet", "Headline", "Country", "Prov/State", "City", "Mentions", "Impressions", "Effective Reach"])

    working = _assign_region_column(df, level)
    working = working[working["Region"].eq(region_name)].copy()
    if working.empty:
        return pd.DataFrame(columns=["Outlet", "Headline", "Country", "Prov/State", "City", "Mentions", "Impressions", "Effective Reach"])

    ranked = (
        working.sort_values(["Mentions", "Impressions", "Effective Reach", "Date"], ascending=[False, False, False, False], na_position="last")
        .loc[:, ["Outlet", "Headline", "Country", "Prov/State", "City", "Mentions", "Impressions", "Effective Reach"]]
        .head(limit)
        .reset_index(drop=True)
    )
    return ranked


def build_region_story_group_examples(
    df: pd.DataFrame,
    level: str,
    region_name: str,
    metric_label: str = "Mentions",
    limit: int = 10,
) -> pd.DataFrame:
    if df is None or df.empty or not str(region_name or "").strip():
        return pd.DataFrame(
            columns=[
                "Headline",
                "Mentions",
                "Impressions",
                "Effective Reach",
                "Date",
                "Example Outlets",
            ]
        )

    working = _assign_region_column(df, level)
    working = working[working["Region"].eq(str(region_name or "").strip())].copy()
    if working.empty:
        return pd.DataFrame(
            columns=[
                "Headline",
                "Mentions",
                "Impressions",
                "Effective Reach",
                "Date",
                "Example Outlets",
            ]
        )

    grouped_story_dicts = _build_top_story_context(working, metric_label=metric_label, limit=limit)
    if not grouped_story_dicts:
        return pd.DataFrame(
            columns=[
                "Headline",
                "Mentions",
                "Impressions",
                "Effective Reach",
                "Date",
                "Example Outlets",
            ]
        )

    grouped_df = pd.DataFrame(grouped_story_dicts)
    if "Latest Date" in grouped_df.columns:
        grouped_df["Date"] = grouped_df["Latest Date"]
    elif "Date" not in grouped_df.columns:
        grouped_df["Date"] = ""
    keep_cols = [
        "Headline",
        "Mentions",
        "Impressions",
        "Effective Reach",
        "Date",
        "Example Outlets",
    ]
    return grouped_df[[col for col in keep_cols if col in grouped_df.columns]].copy()
