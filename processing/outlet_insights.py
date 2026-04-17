from __future__ import annotations

import json
import re
from urllib.parse import urlparse
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
    session_state.setdefault("outlet_cleanup_cluster_index", 0)
    session_state.setdefault("outlet_cleanup_selected_candidates", {})


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
        "Country": "",
        "Coverage Flags": "",
        "Mentions": 0,
        "Impressions": 0,
        "Effective Reach": 0,
        "Prime Example": 0,
    }
    for col, default in defaults.items():
        if col not in working.columns:
            working[col] = default

    text_cols = ["Outlet", "Headline", "Author", "URL", "Type", "Country", "Coverage Flags"]
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
    if not key:
        return None

    direct_rules: list[tuple[str, str]] = [
        (r"\byahoo\b", "Yahoo!"),
        (r"\bmsn\b", "MSN"),
        (r"\bassociated press\b", "Associated Press"),
        (r"\bap planner\b", "Associated Press"),
        (r"\bap state local wire\b", "Associated Press"),
        (r"\bassociated press international\b", "Associated Press"),
        (r"\breuters\b", "Reuters"),
        (r"\bthe canadian press\b", "The Canadian Press"),
        (r"\bcanadian press\b", "The Canadian Press"),
        (r"\bcp newsalert\b", "The Canadian Press"),
        (r"\bradio canada\b", "Radio-Canada"),
        (r"\bcbc\b", "CBC"),
        (r"\bctv\b", "CTV"),
        (r"\bcitynews\b", "CityNews"),
        (r"\bcity news\b", "CityNews"),
        (r"\bglobal news\b", "Global"),
        (r"\bbbc\b", "BBC"),
        (r"\bcnn\b", "CNN"),
        (r"\bsky news\b", "Sky News"),
        (r"\bfrance 24\b", "France 24"),
        (r"\bbnn bloomberg\b", "BNN Bloomberg"),
    ]
    for pattern, canonical in direct_rules:
        if re.search(pattern, key):
            return canonical

    broadcast_indicators = r"\b(news|tv|radio|fm|am|channel|eyewitness|newsradio)\b"
    broadcast_family_rules: list[tuple[str, str]] = [
        (r"\babc(?:\d{1,2})?\b", "ABC"),
        (r"\bcbs(?:\d{1,2})?\b", "CBS"),
        (r"\bnbc(?:\d{1,2})?\b", "NBC"),
        (r"\bfox(?:\d{1,2})?\b", "FOX"),
    ]
    for pattern, canonical in broadcast_family_rules:
        if re.search(pattern, key) and re.search(broadcast_indicators, key):
            return canonical

    if re.search(r"\bglobal\b", key) and re.search(broadcast_indicators, key):
        return "Global"

    return None


def _extract_domain(url: str) -> str:
    text = str(url or "").strip()
    if not text:
        return ""
    try:
        parsed = urlparse(text if "://" in text else f"https://{text}")
        domain = (parsed.netloc or "").lower().strip()
        domain = re.sub(r"^www\.", "", domain)
        if "tveyes" in domain or "tvey" in domain:
            return ""
        return domain
    except Exception:
        return ""


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
            "Representative Snippet": rep.get("Snippet", ""),
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
        "Representative Snippet",
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


def build_outlet_cleanup_clusters(
    df_traditional: pd.DataFrame,
    outlet_rollup_map: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    working = _clean_outlet_df(df_traditional)
    if working.empty:
        return []

    rollup_map = {
        str(k).strip(): str(v).strip()
        for k, v in (outlet_rollup_map or {}).items()
        if str(k).strip() and str(v).strip()
    }

    raw_outlets = (
        working.groupby("Outlet", as_index=False)
        .agg(
            Mentions=("Mentions", "sum"),
            Impressions=("Impressions", "sum"),
            Effective_Reach=("Effective Reach", "sum"),
        )
    )
    if raw_outlets.empty:
        return []

    url_ranked = working.copy()
    url_ranked["_domain"] = url_ranked["URL"].apply(_extract_domain)
    url_ranked = url_ranked[url_ranked["_domain"] != ""].copy()
    if not url_ranked.empty:
        top_domains = (
            url_ranked.sort_values(
                ["Mentions", "Impressions", "Date"],
                ascending=[False, False, False],
                na_position="last",
            )
            .drop_duplicates(subset=["Outlet"], keep="first")
            .rename(columns={"_domain": "Domain"})
        )[["Outlet", "Domain"]]
        raw_outlets = raw_outlets.merge(top_domains, on="Outlet", how="left")
    else:
        raw_outlets["Domain"] = ""

    type_rollup = (
        working[working["Type"] != ""]
        .groupby(["Outlet", "Type"], as_index=False)["Mentions"]
        .sum()
        .sort_values(["Outlet", "Mentions", "Type"], ascending=[True, False, True])
    )
    if not type_rollup.empty:
        top_type = (
            type_rollup.drop_duplicates(subset=["Outlet"], keep="first")
            .rename(columns={"Type": "Media Type"})
        )[["Outlet", "Media Type"]]
        raw_outlets = raw_outlets.merge(top_type, on="Outlet", how="left")
    else:
        raw_outlets["Media Type"] = ""

    country_rollup = (
        working[working["Country"] != ""]
        .groupby(["Outlet", "Country"], as_index=False)["Mentions"]
        .sum()
        .sort_values(["Outlet", "Mentions", "Country"], ascending=[True, False, True])
    )
    if not country_rollup.empty:
        top_country = (
            country_rollup.drop_duplicates(subset=["Outlet"], keep="first")
        )[["Outlet", "Country"]]
        raw_outlets = raw_outlets.merge(top_country, on="Outlet", how="left")
    else:
        raw_outlets["Country"] = ""

    raw_outlets["_normalized_key"] = raw_outlets["Outlet"].apply(_normalize_outlet_key)

    clusters: list[dict[str, Any]] = []
    seen_member_sets: set[tuple[str, ...]] = set()

    def _add_cluster(cluster_id: str, reason: str, members: list[str], suggested_master: str) -> None:
        unique_members = sorted({str(member).strip() for member in members if str(member).strip()})
        if len(unique_members) < 2:
            return

        member_key = tuple(unique_members)
        if member_key in seen_member_sets:
            return

        current_targets = {rollup_map.get(member, member) for member in unique_members}
        if len(current_targets) == 1:
            return

        subset = raw_outlets[raw_outlets["Outlet"].isin(unique_members)].copy()
        if subset.empty:
            return

        ranked_subset = subset.sort_values(
            ["Mentions", "Impressions", "Effective_Reach", "Outlet"],
            ascending=[False, False, False, True],
        ).reset_index(drop=True)

        clusters.append(
            {
                "cluster_id": cluster_id,
                "reason": reason,
                "suggested_master": suggested_master or str(ranked_subset.iloc[0]["Outlet"]).strip(),
                "candidate_count": len(unique_members),
                "mentions": int(subset["Mentions"].sum()),
                "impressions": int(subset["Impressions"].sum()),
                "effective_reach": int(subset["Effective_Reach"].sum()),
                "candidates": ranked_subset[["Outlet", "Media Type", "Country", "Domain", "Mentions", "Impressions", "Effective_Reach"]]
                .rename(columns={"Effective_Reach": "Effective Reach"})
                .to_dict("records"),
            }
        )
        seen_member_sets.add(member_key)

    network_groups: dict[str, list[str]] = {}
    for outlet in raw_outlets["Outlet"].tolist():
        canonical = _normalize_network_canonical(outlet)
        if canonical and canonical != outlet:
            network_groups.setdefault(canonical, []).append(outlet)

    for canonical, members in network_groups.items():
        _add_cluster(
            cluster_id=f"network::{_normalize_outlet_key(canonical)}",
            reason="Suggested network rollup",
            members=members,
            suggested_master=canonical,
        )

    for normalized_key, group in raw_outlets.groupby("_normalized_key", dropna=False):
        members = group["Outlet"].astype(str).tolist()
        if len(members) < 2:
            continue
        ranked_group = group.sort_values(
            ["Mentions", "Impressions", "Effective_Reach", "Outlet"],
            ascending=[False, False, False, True],
        ).reset_index(drop=True)
        suggested_master = str(ranked_group.iloc[0]["Outlet"]).strip()
        _add_cluster(
            cluster_id=f"variant::{normalized_key}",
            reason="Possible naming variants",
            members=members,
            suggested_master=suggested_master,
        )

    clusters.sort(key=lambda row: (row["mentions"], row["impressions"], row["candidate_count"]), reverse=True)
    return clusters


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
    analysis_context: str = "",
) -> str:
    story_json = []
    for _, row in headline_df.drop_duplicates(subset=["Headline"]).head(6).iterrows():
        story_json.append({
            "headline": row.get("Headline", ""),
            "date": str(row.get("Date", "") or ""),
            "author": row.get("Author", ""),
            "type": row.get("Type", ""),
            "mentions": int(row.get("Story Mentions", 0) or 0),
            "impressions": int(row.get("Story Impressions", 0) or 0),
            "snippet": row.get("Representative Snippet", ""),
        })

    top_authors = top_authors_df["Author"].dropna().astype(str).head(5).tolist() if not top_authors_df.empty else []

    return f"""
You are helping a media intelligence analyst summarize why an outlet matters in coverage relevant to this analysis focus:
{analysis_context or client_name or 'the client brand'}.

Write 1-2 concise sentences, about 35-80 words total.

Requirements:
- Write in English only, even if the source stories, outlet names, or example authors are in another language.
- Keep it factual and report-ready.
- Base the summary on the actual substance visible in the representative stories, especially their headlines and snippets.
- Assume the shared analysis focus above is already understood.
- Do not repeat the outlet name in the response; it is already shown in the UI.
- Start directly with the outlet's actual coverage role, topics, or framing style.
- Do not spend opening words explaining that the outlet is "relevant to" or "matters for" the topic.
- Describe observable patterns in the sampled stories, not generic category labels.
- Vary sentence structure naturally across outlets.
- Prioritize what is distinctive in the outlet's sample over repeating the broad topic.
- Use cautious wording when the sample is small or thin.
- Keep the emphasis on what kinds of stories, angles, and framing patterns the outlet tends to publish within the analysis focus.
- Mention specific contributing authors only if materially helpful.
- Do not invent significance, editorial stance, or motives.
- Do not speculate about whether the outlet originated a story.
- Do not speculate beyond what is visible in the sample.
- Do not use bullets.
- Be specific about the recurring subject matter, framing, or story formats visible in the examples.
- Avoid canned openings and repeated opener patterns across entries.
- Avoid padded prose or empty qualifiers.
- Favor concrete, analyst-facing phrasing over generic profile language.
- Do not mention that this is a "sample" or refer to "the sampled stories" explicitly.
- A concise sentence fragment is acceptable if it reads cleanly and gets to the substance faster.
- Avoid boilerplate phrasing such as:
  - "[Outlet name]..."
  - "Regularly surfaces..."
  - "Packages [topic] through..."
  - "Shows up most around..."
  - "In this sample..."
  - "The sampled coverage..."
  - "X is relevant because..."
  - "X matters for [topic] because..."
  - "Coverage of [topic] in X..."

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
    analysis_context: str = "",
) -> tuple[str, int, int]:
    prompt = build_outlet_prompt(
        outlet_name,
        client_name,
        outlet_row,
        headline_df,
        top_authors_df,
        analysis_context=analysis_context,
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
