"""Fingerprints of the inputs used to generate established report observations."""

import pandas as pd

from processing.ai_sentiment import build_sentiment_observation_payload, build_sentiment_observation_prompt
from processing.ai_tagging import build_tag_observation_payload, build_tag_observation_prompt
from processing.analysis_context import (
    apply_session_coverage_flag_policy,
    build_analysis_context_text,
    build_sentiment_analysis_context_text,
    get_analysis_context_payload,
    get_outlet_insight_coverage_flag_exclusions,
    get_qualitative_coverage_flag_exclusions,
)
from processing.author_insights import build_author_headline_table, build_author_metrics, build_author_prompt
from processing.outlet_insights import (
    build_outlet_headline_table,
    build_outlet_metrics,
    build_outlet_prompt,
    build_outlet_top_authors,
)
from processing.top_story_summaries import build_top_story_observation_payload, build_top_story_observation_prompt
from utils.download_freshness import content_fingerprint


def sentiment_observation_fingerprint(state, *, include_not_relevant: bool) -> str:
    prominence = get_analysis_context_payload(state).get("selected_prominence_column", "")
    sentiment_type = state.get("sentiment_type")
    payload = build_sentiment_observation_payload(
        state.get("df_sentiment_unique", pd.DataFrame()), state.get("df_sentiment_grouped_rows"),
        sentiment_type, include_not_relevant, selected_prominence_column=prominence,
    )
    return content_fingerprint(build_sentiment_observation_prompt(
        client_name=str(state.get("client_name", "")).strip(),
        sentiment_type=sentiment_type,
        include_not_relevant=include_not_relevant,
        payload=payload,
        analysis_context=build_sentiment_analysis_context_text(state),
    ))


def tagging_observation_fingerprint(state, *, include_other: bool) -> str:
    prominence = get_analysis_context_payload(state).get("selected_prominence_column", "")
    payload = build_tag_observation_payload(
        state.get("df_tagging_unique"), include_other, selected_prominence_column=prominence,
    )
    return content_fingerprint(build_tag_observation_prompt(
        client_name=state.get("client_name", ""),
        include_other=include_other,
        payload=payload,
        analysis_context=build_analysis_context_text(state),
    ))


def top_story_observation_fingerprint(state) -> str:
    payload = build_top_story_observation_payload(state.get("added_df", pd.DataFrame()))
    return content_fingerprint(build_top_story_observation_prompt(
        get_analysis_context_payload(state)["primary_name"], payload,
        analysis_context=build_analysis_context_text(state),
    ))


def established_report_regeneration_needed(state) -> list[str]:
    checks = {
        "sentiment": (
            "Sentiment observations in Sentiment → Insights",
            lambda: sentiment_observation_fingerprint(
                state, include_not_relevant=state.get("sentiment_observation_include_nr", False),
            ),
        ),
        "tagging": (
            "tag observations in Tagging → Insights",
            lambda: tagging_observation_fingerprint(
                state, include_other=state.get("tagging_observation_include_other", False),
            ),
        ),
        "top_story": ("analysis in Top Stories → Top Story Insights", lambda: top_story_observation_fingerprint(state)),
    }
    regenerate = []
    sections = state.get("_report_copy_sections", set())
    previous_section = {"sentiment": "sentiment", "tagging": "tags", "top_story": "top_stories"}
    for family, (action, current_fingerprint) in checks.items():
        output = state.get(f"{family}_observation_output") or {}
        if output.get("_error"):
            if previous_section[family] in sections:
                regenerate.append(action)
        elif output:
            saved = state.get(f"{family}_observation_fingerprint")
            if not saved or saved != current_fingerprint():
                regenerate.append(action)
    return regenerate


def stale_coverage_themes(state, *, author_blocks, outlet_blocks) -> list[str]:
    """Check only themes actually included in the Word report."""
    raw = state.get("df_traditional", pd.DataFrame())
    context = build_analysis_context_text(state)
    prominence = get_analysis_context_payload(state).get("selected_prominence_column", "")
    client = str(state.get("client_name", "")).strip()
    authors = [block["title"] for block in author_blocks if block["summary"]]
    outlets = [block["title"] for block in outlet_blocks if block["summary"]]
    stale = []

    if authors:
        filtered = apply_session_coverage_flag_policy(raw, state, get_qualitative_coverage_flag_exclusions(state))
        assignments = state.get("auth_outlet_table")
        metrics, stories = build_author_metrics(filtered, auth_outlet_table=assignments)
        saved = state.get("author_insights_summary_fingerprints", {})
        for name in authors:
            row = metrics.loc[metrics["Author"].eq(name)] if not metrics.empty else metrics
            if row.empty or saved.get(name) != content_fingerprint(build_author_prompt(
                name, client, row.iloc[0],
                build_author_headline_table(stories, name, limit=60, selected_prominence_column=prominence),
                analysis_context=context, selected_prominence_column=prominence,
            )):
                stale.append("author coverage themes in Authors → Insights")
                break

    if outlets:
        filtered = apply_session_coverage_flag_policy(raw, state, get_outlet_insight_coverage_flag_exclusions(state))
        rollup = state.get("outlet_rollup_map", {})
        metrics, stories = build_outlet_metrics(filtered, outlet_rollup_map=rollup)
        saved = state.get("outlet_insights_summary_fingerprints", {})
        for name in outlets:
            row = metrics.loc[metrics["Outlet"].eq(name)] if not metrics.empty else metrics
            if row.empty or saved.get(name) != content_fingerprint(build_outlet_prompt(
                name, client, row.iloc[0],
                build_outlet_headline_table(stories, name, limit=60, selected_prominence_column=prominence),
                build_outlet_top_authors(raw, name, limit=5, outlet_rollup_map=rollup),
                analysis_context=context, selected_prominence_column=prominence,
            )):
                stale.append("outlet coverage themes in Outlets → Insights")
                break

    return stale
