from __future__ import annotations

import json
import re
from typing import Any

import pandas as pd
from openai import OpenAI

from utils.api_meter import add_api_usage, extract_usage_tokens


DEFAULT_OUTLET_SUMMARY_MODEL = "gpt-5.4-mini"


def init_outlet_workflow_state(session_state) -> None:
    session_state.setdefault("outlets_section", "Cleanup")
    session_state.setdefault("outlets_rank_by", "Mentions")
    session_state.setdefault("outlet_insights_selected_outlets", [])
    session_state.setdefault("outlet_insights_summaries", {})
    session_state.setdefault("outlet_insights_active_outlet", "")
    session_state.setdefault("outlet_rollup_map", {})
    session_state.setdefault("outlet_cleanup_manual_selection", [])
    session_state.setdefault("outlet_cleanup_manual_target", "")
    session_state.setdefault("outlet_cleanup_rule_mode", "Contains")
    session_state.setdefault("outlet_cleanup_rule_pattern", "")
    session_state.setdefault("outlet_cleanup_rule_target", "")


def _normalize_outlet_key(text: str) -> str:
    text = str(text or "").strip().lower()
    text = re.sub(r"^the\s+", "", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _clean_outlet_df(df: pd.DataFrame) -> pd.DataFrame:
    working = df.copy()

    defaults: dict[str, Any] = {
        "Outlet": "",
        "Headline": "",
        "Author": "",
        "URL": "",
        "Type": "",
        "Coverage Flags": "",
        "Mentions": 0,
        "Impressions": 0,
        "Effective Reach": 0,
        "Prime Example": 0,
    }
    for col, default in defaults.items():
        if col not in working.columns:
            working[col] = default

    text_cols = ["Outlet", "Headline", "Author", "URL", "Type", "Coverage Flags"]
    for col in text_cols:
        working[col] = working[col].fillna("").astype(str).str.strip()

    for col in ["Mentions", "Impressions", "Effective Reach", "Prime Example"]:
        working[col] = pd.to_numeric(working[col], errors="coerce").fillna(0)

    if "Date" in working.columns:
        working["Date"] = pd.to_datetime(working["Date"], errors="coerce")
    else:
        working["Date"] = pd.NaT

    if "Group ID" not in working.columns:
        working["Group ID"] = working["Headline"].replace("", pd.NA).fillna(working.index.astype(str))

    working = working[working["Outlet"] != ""].copy()
    working["_is_good_outlet"] = working["Coverage Flags"].eq("Good Outlet")
    working["_normalized_key"] = working["Outlet"].apply(_normalize_outlet_key)
    return working


def _normalize_network_canonical(text: str) -> str | None:
    key = _normalize_outlet_key(text)
    rules = [
        ("ctv", "CTV"),
        ("cbc", "CBC"),
        ("global news", "Global News"),
        ("global", "Global News"),
        ("citynews", "CityNews"),
        ("the canadian press", "The Canadian Press"),
        ("canadian press", "The Canadian Press"),
        ("radio canada", "Radio-Canada"),
        ("bnn bloomberg", "BNN Bloomberg"),
    ]
    for needle, canonical in rules:
        if needle in key:
            return canonical
    return None


def build_outlet_workflow_df(df_traditional: pd.DataFrame, outlet_rollup_map: dict[str, str] | None = None) -> pd.DataFrame:
    working = _clean_outlet_df(df_traditional)
    if working.empty:
        return working

    rollup_map = {str(k).strip(): str(v).strip() for k, v in (outlet_rollup_map or {}).items() if str(k).strip() and str(v).strip()}
    working["Original Outlet"] = working["Outlet"]
    working["Canonical Outlet"] = working["Original Outlet"].map(lambda name: rollup_map.get(str(name).strip(), str(name).strip()))
    working["Outlet"] = working["Canonical Outlet"]
    return working


def _pick_story_row(group: pd.DataFrame) -> pd.Series:
    ordered = group.sort_values(
        by=["Prime Example", "_is_good_outlet", "Impressions", "Mentions", "Date"],
        ascending=[False, False, False, False, True],
        na_position="last",
    )
    return ordered.iloc[0]


def _build_outlet_story_rows(df: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []

    for (outlet_name, group_id), group in df.groupby(["Outlet", "Group ID"], dropna=False):
        rep = _pick_story_row(group)
        records.append({
            "Outlet": outlet_name,
            "Group ID": group_id,
            "Headline": rep.get("Headline", ""),
            "Date": rep.get("Date"),
            "Author": rep.get("Author", ""),
            "Type": rep.get("Type", ""),
            "Representative URL": rep.get("URL", ""),
            "Representative Flag": rep.get("Coverage Flags", ""),
            "Prime Example Story": int(rep.get("Prime Example", 0) or 0),
            "Story Mentions": int(pd.to_numeric(group["Mentions"], errors="coerce").fillna(0).sum()),
            "Story Impressions": int(pd.to_numeric(group["Impressions"], errors="coerce").fillna(0).sum()),
            "Story Effective Reach": int(pd.to_numeric(group["Effective Reach"], errors="coerce").fillna(0).sum()),
        })

    out = pd.DataFrame(records)
    if not out.empty and "Date" in out.columns:
        out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    return out


def build_outlet_metrics(df_traditional: pd.DataFrame, outlet_rollup_map: dict[str, str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    working = build_outlet_workflow_df(df_traditional, outlet_rollup_map=outlet_rollup_map)
    story_level = _build_outlet_story_rows(working)

    if story_level.empty:
        return pd.DataFrame(), pd.DataFrame()

    summary = (
        story_level.groupby("Outlet", as_index=False)
        .agg(
            Unique_Mentions=("Group ID", "nunique"),
            Mention_Total=("Story Mentions", "sum"),
            Impressions=("Story Impressions", "sum"),
            Effective_Reach=("Story Effective Reach", "sum"),
            Prime_Example_Stories=("Prime Example Story", "sum"),
            Source_Outlets=("Outlet", "size"),
        )
    )
    source_counts = (
        working.groupby("Outlet", as_index=False)["Original Outlet"]
        .nunique()
        .rename(columns={"Original Outlet": "Source_Outlet_Count"})
    )
    summary = summary.merge(source_counts, on="Outlet", how="left")

    good_rate = (
        story_level.assign(_is_good=story_level["Representative Flag"].eq("Good Outlet"))
        .groupby("Outlet", as_index=False)["_is_good"]
        .mean()
        .rename(columns={"_is_good": "Good_Outlet_Rate"})
    )
    summary = summary.merge(good_rate, on="Outlet", how="left")
    summary["Good_Outlet_Rate"] = summary["Good_Outlet_Rate"].fillna(0.0) * 100

    type_rollup = (
        working.groupby(["Outlet", "Type"], as_index=False)["Mentions"]
        .sum()
        .sort_values(["Outlet", "Mentions"], ascending=[True, False])
    )
    if not type_rollup.empty:
        top_types = (
            type_rollup.groupby("Outlet")["Type"]
            .apply(lambda s: ", ".join([str(x) for x in s.head(2) if str(x).strip()]))
            .reset_index(name="Top_Types")
        )
        summary = summary.merge(top_types, on="Outlet", how="left")
    else:
        summary["Top_Types"] = ""

    summary = summary.sort_values(["Mention_Total", "Impressions", "Unique_Mentions"], ascending=False).reset_index(drop=True)
    return summary, story_level


def build_outlet_headline_table(story_level_df: pd.DataFrame, outlet_name: str, limit: int = 5) -> pd.DataFrame:
    if story_level_df.empty:
        return pd.DataFrame()

    outlet_rows = story_level_df[story_level_df["Outlet"] == outlet_name].copy()
    if outlet_rows.empty:
        return pd.DataFrame()

    outlet_rows = outlet_rows.sort_values(
        ["Prime Example Story", "Story Mentions", "Story Impressions"],
        ascending=[False, False, False],
    ).head(limit)

    display_cols = [
        "Headline",
        "Date",
        "Author",
        "Type",
        "Story Mentions",
        "Story Impressions",
        "Story Effective Reach",
        "Representative URL",
    ]
    return outlet_rows[[c for c in display_cols if c in outlet_rows.columns]].copy()


def build_outlet_top_authors(
    df_traditional: pd.DataFrame,
    outlet_name: str,
    limit: int = 5,
    outlet_rollup_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    working = build_outlet_workflow_df(df_traditional, outlet_rollup_map=outlet_rollup_map)
    outlet_rows = working[working["Outlet"] == outlet_name].copy()
    if outlet_rows.empty:
        return pd.DataFrame()

    author_df = (
        outlet_rows[outlet_rows["Author"] != ""]
        .groupby("Author", as_index=False)
        .agg(
            Mentions=("Mentions", "sum"),
            Impressions=("Impressions", "sum"),
        )
        .sort_values(["Mentions", "Impressions"], ascending=False)
        .head(limit)
    )
    return author_df


def build_outlet_variant_candidates(df_traditional: pd.DataFrame, outlet_rollup_map: dict[str, str] | None = None) -> pd.DataFrame:
    working = build_outlet_workflow_df(df_traditional, outlet_rollup_map=outlet_rollup_map)
    if working.empty:
        return pd.DataFrame()

    variants = (
        working.groupby(["_normalized_key", "Outlet"], as_index=False)
        .agg(
            Mentions=("Mentions", "sum"),
            Impressions=("Impressions", "sum"),
            Effective_Reach=("Effective Reach", "sum"),
        )
    )
    group_sizes = variants.groupby("_normalized_key").size().rename("Variant Count")
    variants = variants.merge(group_sizes, on="_normalized_key", how="left")
    variants = variants[variants["Variant Count"] > 1].copy()
    if variants.empty:
        return pd.DataFrame()

    display = (
        variants.sort_values(["_normalized_key", "Mentions", "Impressions"], ascending=[True, False, False])
        .groupby("_normalized_key", as_index=False)
        .agg(
            Variants=("Outlet", lambda s: " | ".join([str(x) for x in s.head(6)])),
            Mentions=("Mentions", "sum"),
            Impressions=("Impressions", "sum"),
            Effective_Reach=("Effective_Reach", "sum"),
            Variant_Count=("Variant Count", "max"),
        )
        .rename(columns={"_normalized_key": "Variant Key"})
        .sort_values(["Mentions", "Impressions"], ascending=False)
    )
    return display


def build_rollup_suggestions(df_traditional: pd.DataFrame, outlet_rollup_map: dict[str, str] | None = None) -> pd.DataFrame:
    working = _clean_outlet_df(df_traditional)
    if working.empty:
        return pd.DataFrame()

    raw_outlets = (
        working.groupby("Outlet", as_index=False)
        .agg(
            Mentions=("Mentions", "sum"),
            Impressions=("Impressions", "sum"),
            Effective_Reach=("Effective Reach", "sum"),
        )
    )
    rollup_map = {str(k).strip(): str(v).strip() for k, v in (outlet_rollup_map or {}).items() if str(k).strip() and str(v).strip()}

    suggestions: dict[str, list[str]] = {}
    for outlet in raw_outlets["Outlet"].tolist():
        suggested = _normalize_network_canonical(outlet)
        if suggested and suggested != outlet:
            suggestions.setdefault(suggested, []).append(outlet)

    rows: list[dict[str, Any]] = []
    for canonical, outlets in suggestions.items():
        unique_outlets = sorted(set(outlets))
        if len(unique_outlets) < 2:
            continue
        subset = raw_outlets[raw_outlets["Outlet"].isin(unique_outlets)].copy()
        mapped_already = [outlet for outlet in unique_outlets if rollup_map.get(outlet, outlet) == canonical]
        if len(mapped_already) >= len(unique_outlets):
            continue
        rows.append({
            "Suggested Rollup": canonical,
            "Source Outlet Count": len(unique_outlets),
            "Already Mapped": len(mapped_already),
            "Mentions": int(subset["Mentions"].sum()),
            "Impressions": int(subset["Impressions"].sum()),
            "Effective Reach": int(subset["Effective_Reach"].sum()),
            "Source Outlets": " | ".join(unique_outlets[:8]),
        })

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values(["Mentions", "Impressions"], ascending=False).reset_index(drop=True)


def build_outlet_rollup_preview(df_traditional: pd.DataFrame, outlet_rollup_map: dict[str, str] | None = None) -> pd.DataFrame:
    working = _clean_outlet_df(df_traditional)
    if working.empty:
        return pd.DataFrame()

    rollup_map = {str(k).strip(): str(v).strip() for k, v in (outlet_rollup_map or {}).items() if str(k).strip() and str(v).strip()}
    preview = (
        working.groupby("Outlet", as_index=False)
        .agg(
            Mentions=("Mentions", "sum"),
            Impressions=("Impressions", "sum"),
            Effective_Reach=("Effective Reach", "sum"),
        )
        .sort_values(["Mentions", "Impressions"], ascending=False)
        .reset_index(drop=True)
    )
    preview["Canonical Outlet"] = preview["Outlet"].map(lambda name: rollup_map.get(str(name).strip(), str(name).strip()))
    preview["Rollup Applied"] = preview["Outlet"] != preview["Canonical Outlet"]
    return preview[["Outlet", "Canonical Outlet", "Rollup Applied", "Mentions", "Impressions", "Effective_Reach"]]


def apply_outlet_rollup_map(session_state, outlet_names: list[str], canonical_name: str) -> None:
    canonical_name = str(canonical_name or "").strip()
    names = [str(name).strip() for name in outlet_names if str(name).strip()]
    if not canonical_name or not names:
        return

    mapping = dict(session_state.get("outlet_rollup_map", {}))
    for name in names:
        mapping[name] = canonical_name
    session_state.outlet_rollup_map = mapping

    selected = [mapping.get(outlet, outlet) for outlet in session_state.get("outlet_insights_selected_outlets", [])]
    session_state.outlet_insights_selected_outlets = list(dict.fromkeys(selected))

    active = str(session_state.get("outlet_insights_active_outlet", "") or "").strip()
    if active in names:
        session_state.outlet_insights_active_outlet = canonical_name

    summaries = dict(session_state.get("outlet_insights_summaries", {}))
    migrated = False
    for name in names:
        if name in summaries and canonical_name not in summaries:
            summaries[canonical_name] = summaries[name]
            migrated = True
        summaries.pop(name, None)
    if migrated or names:
        session_state.outlet_insights_summaries = summaries


def remove_outlet_rollup_map(session_state, outlet_names: list[str]) -> None:
    mapping = dict(session_state.get("outlet_rollup_map", {}))
    changed = False
    for name in outlet_names:
        key = str(name).strip()
        if key in mapping:
            mapping.pop(key, None)
            changed = True
    if changed:
        session_state.outlet_rollup_map = mapping


def build_outlet_prompt(
    outlet_name: str,
    client_name: str,
    outlet_row: pd.Series,
    headline_df: pd.DataFrame,
    top_authors_df: pd.DataFrame,
) -> str:
    story_json = []
    for _, row in headline_df.drop_duplicates(subset=["Headline"]).head(6).iterrows():
        story_json.append({
            "headline": row.get("Headline", ""),
            "author": row.get("Author", ""),
            "mentions": int(row.get("Story Mentions", 0) or 0),
            "impressions": int(row.get("Story Impressions", 0) or 0),
        })

    top_authors = top_authors_df["Author"].dropna().astype(str).head(5).tolist() if not top_authors_df.empty else []

    return f"""
You are helping a media intelligence analyst summarize why an outlet matters in coverage relevant to {client_name or 'the client brand'}.

Write 1-2 concise sentences, about 35-80 words total.

Requirements:
- Keep it factual and report-ready.
- Focus on recurring themes or angles visible in the example stories.
- Keep the emphasis on how this outlet's coverage relates to {client_name or 'the client brand'}.
- Mention specific contributing authors only if materially helpful.
- Do not invent significance, editorial stance, or motives.
- Do not speculate about whether the outlet originated a story.
- Do not use bullets.

Outlet: {outlet_name}
Top media types: {outlet_row.get("Top_Types", "")}
Top contributing authors: {", ".join(top_authors)}
Total mentions: {int(outlet_row.get("Mention_Total", 0) or 0)}
Unique mentions: {int(outlet_row.get("Unique_Mentions", 0) or 0)}
Impressions: {int(outlet_row.get("Impressions", 0) or 0)}

Representative stories:
{json.dumps(story_json, ensure_ascii=True)}
""".strip()


def generate_outlet_summary(
    outlet_name: str,
    client_name: str,
    outlet_row: pd.Series,
    headline_df: pd.DataFrame,
    top_authors_df: pd.DataFrame,
    api_key: str,
    model: str = DEFAULT_OUTLET_SUMMARY_MODEL,
) -> tuple[str, int, int]:
    prompt = build_outlet_prompt(outlet_name, client_name, outlet_row, headline_df, top_authors_df)
    client = OpenAI(api_key=api_key)

    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": "You write concise, neutral media-intelligence summaries."},
            {"role": "user", "content": prompt},
        ],
        text={"verbosity": "low"},
    )

    add_api_usage(response, model)
    in_tok, out_tok = extract_usage_tokens(response)
    summary = getattr(response, "output_text", "") or ""
    return summary.strip(), in_tok, out_tok
