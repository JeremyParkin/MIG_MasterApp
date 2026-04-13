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
    session_state.setdefault("outlet_cleanup_source", "")
    session_state.setdefault("outlet_cleanup_target", "")


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


def build_outlet_metrics(df_traditional: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    working = _clean_outlet_df(df_traditional)
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
        )
    )

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
        "Story Mentions",
        "Story Impressions",
        "Representative URL",
    ]
    return outlet_rows[[c for c in display_cols if c in outlet_rows.columns]].copy()


def build_outlet_top_authors(df_traditional: pd.DataFrame, outlet_name: str, limit: int = 5) -> pd.DataFrame:
    working = _clean_outlet_df(df_traditional)
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


def build_outlet_variant_candidates(df_traditional: pd.DataFrame) -> pd.DataFrame:
    working = _clean_outlet_df(df_traditional)
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


def apply_outlet_name_cleanup(session_state, old_name: str, new_name: str) -> None:
    old_name = str(old_name or "").strip()
    new_name = str(new_name or "").strip()
    if not old_name or not new_name or old_name == new_name:
        return

    for key, value in list(session_state.items()):
        if isinstance(value, pd.DataFrame) and not value.empty:
            updated = value.copy()
            changed = False
            for col in ["Outlet", "Example Outlet"]:
                if col in updated.columns:
                    updated.loc[updated[col].fillna("").astype(str).str.strip() == old_name, col] = new_name
                    changed = True
            if changed:
                session_state[key] = updated

    selected = session_state.get("outlet_insights_selected_outlets", [])
    session_state.outlet_insights_selected_outlets = [new_name if outlet == old_name else outlet for outlet in selected]

    active = str(session_state.get("outlet_insights_active_outlet", "") or "").strip()
    if active == old_name:
        session_state.outlet_insights_active_outlet = new_name

    summaries = dict(session_state.get("outlet_insights_summaries", {}))
    if old_name in summaries:
        if new_name not in summaries:
            summaries[new_name] = summaries[old_name]
        summaries.pop(old_name, None)
        session_state.outlet_insights_summaries = summaries


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
