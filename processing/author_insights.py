from __future__ import annotations

import math
import json
from typing import Any

import pandas as pd
from openai import OpenAI

from utils.api_meter import add_api_usage, extract_usage_tokens


DEFAULT_AUTHOR_SUMMARY_MODEL = "gpt-5.4-mini"

def init_author_insights_state(session_state) -> None:
    session_state.setdefault("author_insights_selected_authors", [])
    session_state.setdefault("author_insights_summaries", {})
    session_state.setdefault("author_insights_target_count", 10)


def _clean_author_df(df: pd.DataFrame) -> pd.DataFrame:
    working = df.copy()

    for col in ["Author", "Headline", "Outlet", "Coverage Flags", "Type", "URL", "Snippet"]:
        if col not in working.columns:
            working[col] = ""
        working[col] = working[col].fillna("").astype(str).str.strip()

    if "Mentions" not in working.columns:
        working["Mentions"] = 1
    if "Impressions" not in working.columns:
        working["Impressions"] = 0
    if "Effective Reach" not in working.columns:
        working["Effective Reach"] = 0
    if "Prime Example" not in working.columns:
        working["Prime Example"] = 0

    working["Mentions"] = pd.to_numeric(working["Mentions"], errors="coerce").fillna(0)
    working["Impressions"] = pd.to_numeric(working["Impressions"], errors="coerce").fillna(0)
    working["Effective Reach"] = pd.to_numeric(working["Effective Reach"], errors="coerce").fillna(0)
    working["Prime Example"] = pd.to_numeric(working["Prime Example"], errors="coerce").fillna(0)

    if "Date" in working.columns:
        working["Date"] = pd.to_datetime(working["Date"], errors="coerce")
    else:
        working["Date"] = pd.NaT

    working["Author"] = working["Author"].str.strip()
    working = working[working["Author"] != ""].copy()

    if "Group ID" not in working.columns:
        working["Group ID"] = working["Headline"].replace("", pd.NA).fillna(working.index.astype(str))

    working["Coverage Flags"] = working["Coverage Flags"].fillna("")
    working["_is_good_outlet"] = working["Coverage Flags"].eq("Good Outlet")

    return working


def _pick_story_row(group: pd.DataFrame) -> pd.Series:
    ordered = group.sort_values(
        by=["Prime Example", "_is_good_outlet", "Impressions", "Date"],
        ascending=[False, False, False, True],
        na_position="last",
    )
    return ordered.iloc[0]


def _build_story_level_rows(df: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []

    for (author_name, group_id), group in df.groupby(["Author", "Group ID"], dropna=False):
        rep = _pick_story_row(group)
        records.append({
            "Author": author_name,
            "Group ID": group_id,
            "Headline": rep.get("Headline", ""),
            "Date": rep.get("Date"),
            "Type": rep.get("Type", ""),
            "Representative Outlet": rep.get("Outlet", ""),
            "Representative URL": rep.get("URL", ""),
            "Representative Flag": rep.get("Coverage Flags", ""),
            "Story Mentions": int(pd.to_numeric(group["Mentions"], errors="coerce").fillna(0).sum()),
            "Story Impressions": int(pd.to_numeric(group["Impressions"], errors="coerce").fillna(0).sum()),
            "Story Effective Reach": int(pd.to_numeric(group["Effective Reach"], errors="coerce").fillna(0).sum()),
            "Syndicated Pickups": max(len(group) - 1, 0),
            "Outlet Count": int(group["Outlet"].replace("", pd.NA).dropna().nunique()),
            "Good Outlet Pickups": int(group["_is_good_outlet"].sum()),
            "Outlets": ", ".join(group["Outlet"].replace("", pd.NA).dropna().astype(str).drop_duplicates().head(6)),
        })

    out = pd.DataFrame(records)
    if not out.empty and "Date" in out.columns:
        out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    return out


def build_author_metrics(
    df_traditional: pd.DataFrame,
    auth_outlet_table: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    working = _clean_author_df(df_traditional)
    story_level = _build_story_level_rows(working)

    if story_level.empty:
        return pd.DataFrame(), pd.DataFrame()

    summary = (
        story_level.groupby("Author", as_index=False)
        .agg(
            Unique_Stories=("Group ID", "nunique"),
            Syndicated_Pickups=("Syndicated Pickups", "sum"),
            Mention_Total=("Story Mentions", "sum"),
            Impressions=("Story Impressions", "sum"),
            Effective_Reach=("Story Effective Reach", "sum"),
            Good_Outlet_Stories=("Good Outlet Pickups", lambda s: int((pd.Series(s) > 0).sum())),
            Notable_Outlet_Count=("Outlet Count", "sum"),
        )
    )

    first_dates = story_level.groupby("Author")["Date"].min().rename("First_Date")
    last_dates = story_level.groupby("Author")["Date"].max().rename("Last_Date")
    summary = summary.merge(first_dates, on="Author", how="left").merge(last_dates, on="Author", how="left")

    outlet_rollup = (
        working[working["Outlet"] != ""]
        .groupby(["Author", "Outlet"], as_index=False)["Mentions"]
        .sum()
        .sort_values(["Author", "Mentions"], ascending=[True, False])
    )
    primary_outlet = outlet_rollup.drop_duplicates(subset=["Author"], keep="first")
    primary_outlet = primary_outlet.rename(columns={"Outlet": "Coverage_Primary_Outlet", "Mentions": "Coverage_Primary_Outlet_Mentions"})
    summary = summary.merge(primary_outlet[["Author", "Coverage_Primary_Outlet", "Coverage_Primary_Outlet_Mentions"]], on="Author", how="left")

    if auth_outlet_table is not None and not auth_outlet_table.empty and "Outlet" in auth_outlet_table.columns:
        assignment_map = auth_outlet_table[["Author", "Outlet"]].copy()
        assignment_map["Outlet"] = assignment_map["Outlet"].fillna("").astype(str).str.strip()
        assignment_map = assignment_map.rename(columns={"Outlet": "Assigned Outlet"})
        summary = summary.merge(assignment_map, on="Author", how="left")
    else:
        summary["Assigned Outlet"] = ""

    summary["Assigned Outlet"] = summary["Assigned Outlet"].fillna("")
    summary["Syndication Ratio"] = (
        summary["Syndicated_Pickups"] / summary["Unique_Stories"].replace(0, pd.NA)
    ).fillna(0.0)
    summary = summary.sort_values(["Mention_Total", "Impressions", "Unique_Stories"], ascending=False).reset_index(drop=True)
    return summary, story_level


def build_author_headline_table(story_level_df: pd.DataFrame, author_name: str, limit: int = 12) -> pd.DataFrame:
    if story_level_df.empty:
        return pd.DataFrame()

    author_rows = story_level_df[story_level_df["Author"] == author_name].copy()
    if author_rows.empty:
        return pd.DataFrame()

    author_rows = author_rows.sort_values(
        ["Story Mentions", "Story Impressions", "Good Outlet Pickups"],
        ascending=False,
    ).head(limit)

    display_cols = [
        "Headline",
        "Date",
        "Story Mentions",
        "Story Impressions",
        "Representative Outlet",
        "Representative URL",
    ]
    existing = [c for c in display_cols if c in author_rows.columns]
    return author_rows[existing].copy()


def build_author_prompt(
    author_name: str,
    client_name: str,
    author_row: pd.Series,
    headline_df: pd.DataFrame,
) -> str:
    story_json = []
    for _, row in headline_df.drop_duplicates(subset=["Headline", "Representative Outlet"]).head(8).iterrows():
        story_json.append({
            "headline": row.get("Headline", ""),
            "mentions": int(row.get("Story Mentions", 0) or 0),
            "impressions": int(row.get("Story Impressions", 0) or 0),
            "outlet": row.get("Representative Outlet", ""),
            "flag": row.get("Representative Flag", ""),
        })

    return f"""
You are helping a media intelligence analyst summarize an author's coverage themes that are relevant to {client_name or 'the client brand'}.

Write 1-2 concise sentences, about 35-80 words total.

Requirements:
- Keep it factual and report-ready.
- Focus on recurring themes, angles, or beats visible in the headlines and example stories.
- Keep the emphasis on how this author's coverage relates to {client_name or 'the client brand'}.
- Mention notable outlets only when they materially help describe the author's footprint.
- Do not invent expertise, intent, or biography.
- Do not list raw counts unless they are essential.
- Do not use bullets.

Author: {author_name}
Total mentions: {int(author_row.get("Mention_Total", 0) or 0)}
Total impressions: {int(author_row.get("Impressions", 0) or 0)}

Representative stories:
{json.dumps(story_json, ensure_ascii=True)}
""".strip()


def generate_author_summary(
    author_name: str,
    client_name: str,
    author_row: pd.Series,
    headline_df: pd.DataFrame,
    api_key: str,
    model: str = DEFAULT_AUTHOR_SUMMARY_MODEL,
) -> tuple[str, int, int]:
    prompt = build_author_prompt(author_name, client_name, author_row, headline_df)
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
