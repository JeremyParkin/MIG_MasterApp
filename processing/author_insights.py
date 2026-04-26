from __future__ import annotations

import math
import json
from typing import Any

import pandas as pd
from openai import OpenAI

from processing.prominence import get_prominence_weight_series
from utils.api_meter import add_api_usage, extract_usage_tokens


DEFAULT_AUTHOR_SUMMARY_MODEL = "gpt-5.4-mini"
DEFAULT_AUTHOR_PRIMARY_EXAMPLE_LIMIT = 10
DEFAULT_AUTHOR_SUPPORTING_EVIDENCE_LIMIT = 40

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


def _normalized_rank_score(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").fillna(0)
    if numeric.empty:
        return pd.Series(dtype="float64")

    rank_series = numeric.rank(method="dense", ascending=False).astype(float)
    max_rank = float(rank_series.max()) if not rank_series.empty else 1.0
    if max_rank <= 1:
        return pd.Series(1.0, index=numeric.index, dtype="float64")
    return 1.0 - ((rank_series - 1.0) / (max_rank - 1.0))


def _truncate_text(text: str, limit: int = 320) -> str:
    text = " ".join(str(text or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _score_author_story_rows(
    story_rows: pd.DataFrame,
    selected_prominence_column: str = "",
) -> pd.DataFrame:
    working = story_rows.copy()
    for col in ["Story Mentions", "Story Impressions", "Story Effective Reach", "Syndicated Pickups", "Good Outlet Pickups"]:
        if col not in working.columns:
            working[col] = 0
        working[col] = pd.to_numeric(working[col], errors="coerce").fillna(0)

    working["_has_url"] = working.get("Representative URL", pd.Series(index=working.index, dtype="object")).fillna("").astype(str).str.strip().ne("")
    working["_is_online"] = working.get("Type", pd.Series(index=working.index, dtype="object")).fillna("").astype(str).str.upper().eq("ONLINE")
    working["_Story_Mentions_rank_score"] = _normalized_rank_score(working["Story Mentions"])
    working["_Story_Impressions_rank_score"] = _normalized_rank_score(working["Story Impressions"])
    working["_Story_Effective_Reach_rank_score"] = _normalized_rank_score(working["Story Effective Reach"])
    working["_Syndicated_Pickups_rank_score"] = _normalized_rank_score(working["Syndicated Pickups"])
    working["_Good_Outlet_Pickups_rank_score"] = _normalized_rank_score(working["Good Outlet Pickups"])
    working["_prominence_bonus"] = get_prominence_weight_series(working, selected_prominence_column)
    working["_prompt_story_score"] = (
        working["_Story_Mentions_rank_score"] * 3.0
        + working["_Story_Impressions_rank_score"] * 2.0
        + working["_Story_Effective_Reach_rank_score"] * 2.0
        + working["_Syndicated_Pickups_rank_score"] * 1.5
        + working["_Good_Outlet_Pickups_rank_score"] * 1.0
        + working["_prominence_bonus"]
        + working["_has_url"].astype(float) * 0.2
        + working["_is_online"].astype(float) * 0.35
    )
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
    prominence_cols = [c for c in df.columns if isinstance(c, str) and c.startswith("Prominence")]

    for (author_name, group_id), group in df.groupby(["Author", "Group ID"], dropna=False):
        rep = _pick_story_row(group)
        record = {
            "Author": author_name,
            "Group ID": group_id,
            "Headline": rep.get("Headline", ""),
            "Date": rep.get("Date"),
            "Type": rep.get("Type", ""),
            "Representative Outlet": rep.get("Outlet", ""),
            "Representative URL": rep.get("URL", ""),
            "Representative Snippet": rep.get("Snippet", ""),
            "Representative Flag": rep.get("Coverage Flags", ""),
            "Story Mentions": int(pd.to_numeric(group["Mentions"], errors="coerce").fillna(0).sum()),
            "Story Impressions": int(pd.to_numeric(group["Impressions"], errors="coerce").fillna(0).sum()),
            "Story Effective Reach": int(pd.to_numeric(group["Effective Reach"], errors="coerce").fillna(0).sum()),
            "Syndicated Pickups": max(len(group) - 1, 0),
            "Outlet Count": int(group["Outlet"].replace("", pd.NA).dropna().nunique()),
            "Good Outlet Pickups": int(group["_is_good_outlet"].sum()),
            "Outlets": ", ".join(group["Outlet"].replace("", pd.NA).dropna().astype(str).drop_duplicates().head(6)),
        }
        for col in prominence_cols:
            record[col] = pd.to_numeric(group[col], errors="coerce").fillna(0).max()
        records.append(record)

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


def build_author_headline_table(
    story_level_df: pd.DataFrame,
    author_name: str,
    limit: int = 12,
    selected_prominence_column: str = "",
) -> pd.DataFrame:
    if story_level_df.empty:
        return pd.DataFrame()

    author_rows = story_level_df[story_level_df["Author"] == author_name].copy()
    if author_rows.empty:
        return pd.DataFrame()

    author_rows = _score_author_story_rows(
        author_rows,
        selected_prominence_column=selected_prominence_column,
    ).sort_values(
        ["_prompt_story_score", "Story Mentions", "Story Impressions", "Story Effective Reach"],
        ascending=[False, False, False, False],
    ).head(limit)

    display_cols = [
        "Headline",
        "Date",
        "Type",
        "Story Mentions",
        "Story Impressions",
        "Story Effective Reach",
        "Representative Outlet",
        "Representative URL",
        "Representative Snippet",
        "Representative Flag",
    ]
    existing = [c for c in display_cols if c in author_rows.columns]
    return author_rows[existing].copy()


def build_author_prompt_payload(
    headline_df: pd.DataFrame,
    primary_limit: int = DEFAULT_AUTHOR_PRIMARY_EXAMPLE_LIMIT,
    supporting_limit: int = DEFAULT_AUTHOR_SUPPORTING_EVIDENCE_LIMIT,
    selected_prominence_column: str = "",
) -> dict[str, Any]:
    if headline_df is None or headline_df.empty:
        return {"primary_examples": [], "supporting_evidence": []}

    working = _score_author_story_rows(
        headline_df,
        selected_prominence_column=selected_prominence_column,
    )
    ranked = (
        working.sort_values(
            ["_prompt_story_score", "Story Mentions", "Story Impressions", "Story Effective Reach"],
            ascending=[False, False, False, False],
        )
        .drop_duplicates(subset=["Headline", "Representative Outlet"], keep="first")
        .copy()
    )

    primary_df = ranked.head(primary_limit).copy()
    supporting_df = ranked.iloc[primary_limit:].head(supporting_limit).copy()

    primary_examples = []
    for _, row in primary_df.iterrows():
        primary_examples.append(
            {
                "headline": row.get("Headline", ""),
                "date": str(row.get("Date", "") or ""),
                "type": row.get("Type", ""),
                "mentions": int(row.get("Story Mentions", 0) or 0),
                "impressions": int(row.get("Story Impressions", 0) or 0),
                "effective_reach": int(row.get("Story Effective Reach", 0) or 0),
                "syndicated_pickups": int(row.get("Syndicated Pickups", 0) or 0),
                "outlet": row.get("Representative Outlet", ""),
                "url": row.get("Representative URL", ""),
                "snippet": _truncate_text(row.get("Representative Snippet", ""), 320),
            }
        )

    supporting_evidence = []
    for _, row in supporting_df.iterrows():
        supporting_evidence.append(
            {
                "headline": row.get("Headline", ""),
                "date": str(row.get("Date", "") or ""),
                "outlet": row.get("Representative Outlet", ""),
                "mentions": int(row.get("Story Mentions", 0) or 0),
                "impressions": int(row.get("Story Impressions", 0) or 0),
                "syndicated_pickups": int(row.get("Syndicated Pickups", 0) or 0),
                "snippet": _truncate_text(row.get("Representative Snippet", ""), 200),
            }
        )

    return {
        "primary_examples": primary_examples,
        "supporting_evidence": supporting_evidence,
    }


def build_author_prompt(
    author_name: str,
    client_name: str,
    author_row: pd.Series,
    headline_df: pd.DataFrame,
    analysis_context: str = "",
    selected_prominence_column: str = "",
) -> str:
    prompt_payload = build_author_prompt_payload(
        headline_df,
        selected_prominence_column=selected_prominence_column,
    )

    return f"""
You are helping a media intelligence analyst summarize an author's coverage themes that are relevant to this analysis focus:
{analysis_context or client_name or 'the client brand'}

Write 1-2 concise sentences, about 35-80 words total.

Requirements:
- Write in English only, even if the source stories or author names are in another language.
- Keep it factual and report-ready.
- Base the summary on the actual substance visible in the representative stories, especially their headlines and snippets.
- Use the primary examples to anchor the clearest patterns, and use the supporting evidence to judge whether those patterns recur more broadly.
- Assume the shared analysis focus above is already understood.
- Do not repeat the author's name in the response; it is already shown in the UI.
- Start directly with the distinctive beat, angle, or role visible in the stories.
- Do not waste opening words restating that the author's coverage is "related to" or "focused on" the topic.
- Describe observable patterns in the sampled stories, not generic category labels.
- Vary sentence structure naturally across authors.
- Prioritize what is distinctive in the author's sample over repeating the broad topic.
- Use cautious wording when the sample is small or thin.
- Keep the emphasis on the distinctive substance of the author's coverage, not on re-framing the assignment.
- Mention notable outlets only when they materially help describe the author's footprint.
- Do not invent expertise, intent, or biography.
- Do not speculate beyond what is visible in the sample.
- Do not list raw counts unless they are essential.
- Do not use bullets.
- Be specific about recurring angles, subjects, or framing patterns visible in the examples.
- Avoid canned openings and repeated opener patterns across entries.
- Avoid padded prose or empty qualifiers.
- Favor concrete, analyst-facing phrasing over generic profile language.
- Do not mention that this is a "sample" or refer to "the sampled stories" explicitly.
- A concise sentence fragment is acceptable if it reads cleanly and gets to the substance faster.
- Avoid boilerplate phrasing such as:
  - "[Author name]'s coverage..."
  - "Writes mainly about..."
  - "In this sample..."
  - "The sampled coverage..."
  - "Coverage centers on..." when a more concrete phrasing would work
  - "Quebec-related coverage..."
  - "Coverage of [topic]..."
  - "This author's coverage centers on..." when a more direct opening would work.

Author: {author_name}
Total mentions: {int(author_row.get("Mention_Total", 0) or 0)}
Total impressions: {int(author_row.get("Impressions", 0) or 0)}

Representative stories:
{json.dumps(prompt_payload, ensure_ascii=True)}
""".strip()


def generate_author_summary(
    author_name: str,
    client_name: str,
    author_row: pd.Series,
    headline_df: pd.DataFrame,
    api_key: str,
    model: str = DEFAULT_AUTHOR_SUMMARY_MODEL,
    analysis_context: str = "",
    selected_prominence_column: str = "",
) -> tuple[str, int, int]:
    prompt = build_author_prompt(
        author_name,
        client_name,
        author_row,
        headline_df,
        analysis_context=analysis_context,
        selected_prominence_column=selected_prominence_column,
    )
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
