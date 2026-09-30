from __future__ import annotations

import json
import time
import warnings
import html

import altair as alt
import pandas as pd
import streamlit as st

from processing.analysis_context import (
    apply_session_coverage_flag_policy,
    build_analysis_context_text,
    build_analysis_context_caption,
    build_analysis_context_required_message,
    build_sentiment_analysis_context_text,
    format_qualitative_exclusion_caption,
    get_analysis_context_payload,
    get_qualitative_coverage_flag_exclusions,
    has_saved_analysis_context,
)
from processing.ai_sentiment import ensure_ai_sentiment_columns
from processing.jev_sentiment import (
    DEFAULT_JEV_MAX_WORKERS,
    DEFAULT_JEV_MODEL,
    DEFAULT_JEV_TAGGING_MODE,
    JEV_3WAY_SENTIMENT_ORDER,
    JEV_5WAY_SENTIMENT_ORDER,
    JEV_COMBINED_SCHEMES,
    JEV_TAGGING_MODES,
    build_jev_tag_color_scale,
    build_jev_all_applicable_tag_distribution,
    build_jev_combined_sentiment_payload,
    build_jev_best_tag_distribution,
    build_jev_results_display,
    build_jev_sentiment_distribution,
    build_jev_tag_config_fingerprint,
    build_jev_tag_question_map,
    ensure_jev_sentiment_columns,
    filter_jev_all_applicable_tag_distribution,
    filter_jev_best_tag_distribution,
    filter_jev_sentiment_distribution,
    get_remaining_jev_sentiment_rows,
    get_openrouter_api_key,
    parse_jev_tag_definitions,
    rapid_labeling_column_name,
    recompute_jev_official_tag_assignments,
    rename_jev_columns_to_rapid,
    run_jev_sentiment_batch,
)
from processing.rapid_second_opinion import (
    RAPID_REVIEW_MAX_WORKERS,
    RAPID_REVIEW_MODEL,
    build_rapid_review_candidates,
    clear_rapid_review_columns,
    ensure_rapid_review_columns,
    get_openai_api_key,
    normalize_first_pass_sentiment,
    resolve_rapid_review_recommendation,
    run_rapid_review_batch,
)
from processing.rapid_resolution import (
    HUMAN_SENTIMENT_SCALE_3_WAY,
    HUMAN_SENTIMENT_SCALE_5_WAY,
    HUMAN_STATE_ACCEPTED,
    HUMAN_STATE_ASSIGNED,
    RELEVANCE_NOT_RELEVANT,
    RELEVANCE_RELEVANT,
    LEGACY_RAPID_TAGGING_MODE_APPLICABLE,
    LEGACY_RAPID_TAGGING_MODE_BEST,
    RAPID_TAG_REVIEW_MODE_APPLICABLE,
    RAPID_TAG_REVIEW_MODE_BEST,
    apply_rapid_sentiment_human_selection,
    apply_rapid_tag_human_selection,
    build_final_rapid_sentiment_frame,
    build_final_rapid_tag_frame,
    build_effective_rapid_sentiment_frame,
    build_effective_rapid_tag_frame,
    build_rapid_highlight_keywords,
    ensure_rapid_human_review_columns,
    format_rapid_numeric,
    summarize_rapid_label_provenance,
)
from processing.rapid_insights import (
    RAPID_INSIGHT_FIELD_OPTIONS,
    RAPID_SENTIMENT_SCALE_3_WAY,
    RAPID_SENTIMENT_SCALE_5_WAY,
    build_rapid_sentiment_distribution,
    build_rapid_sentiment_observation_payload,
    build_rapid_sentiment_insight_frame,
    build_rapid_tag_distribution,
    build_rapid_tag_observation_payload,
    build_rapid_tag_insight_frame,
    generate_rapid_sentiment_observations,
    generate_rapid_tag_observations,
    rapid_observation_fingerprint,
    sentiment_order_for_scale,
)
from processing.ai_tagging import (
    RESERVED_OTHER_DEFINITION,
    RESERVED_OTHER_TAG,
    normalize_tag_list,
    remove_reserved_tag_from_text,
)
from processing.sentiment_config import (
    DEFAULT_MAX_FULL_ROWS,
    build_tolerant_regex_str,
    calculate_representative_sample_size,
    prepare_sentiment_datasets,
    get_sentiment_source_rows,
)
from processing.spot_checks import (
    apply_translation_to_group,
    escape_markdown,
    highlight_with_tolerant_regex,
    translate_text,
)
from ui.page_help import set_page_help_context
from ui.insight_blocks import build_linked_example_blocks_html
from utils.api_meter import apply_usage_to_session

warnings.filterwarnings("ignore")


def init_jev_sentiment_state() -> None:
    defaults = {
        "jev_sentiment_section": "Prepare Sample",
        "jev_sentiment_config_step": False,
        "jev_sentiment_sample_mode": "representative",
        "jev_sentiment_sample_size": None,
        "jev_sentiment_full_override": False,
        "jev_sentiment_elapsed_time": 0.0,
        "jev_sentiment_excluded_flags": [],
        "jev_tags_text": "",
        "jev_tag_definitions": {},
        "jev_tagging_mode": DEFAULT_JEV_TAGGING_MODE,
        "jev_tag_config_fingerprint": "",
        "jev_analysis_context_fingerprint": "",
        "df_jev_sentiment_rows": pd.DataFrame(),
        "df_jev_sentiment_grouped_rows": pd.DataFrame(),
        "df_jev_sentiment_unique": pd.DataFrame(),
        "rapid_sentiment_observation_output": {},
        "rapid_sentiment_observation_fingerprint": "",
        "rapid_sentiment_observation_scale": RAPID_SENTIMENT_SCALE_3_WAY,
        "rapid_sentiment_observation_include_nr": False,
        "rapid_tag_observation_output": {},
        "rapid_tag_observation_fingerprint": "",
        "rapid_tag_observation_mode": RAPID_TAG_REVIEW_MODE_BEST,
        "rapid_tag_observation_include_other": False,
        "rapid_spot_review_domain": "Sentiment",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_jev_sentiment_state() -> None:
    for key in [
        "jev_sentiment_config_step",
        "jev_sentiment_sample_mode",
        "jev_sentiment_sample_size",
        "jev_sentiment_full_override",
        "jev_sentiment_elapsed_time",
        "jev_sentiment_excluded_flags",
        "jev_tags_text",
        "jev_tag_definitions",
        "jev_tagging_mode",
        "jev_tag_config_fingerprint",
        "jev_analysis_context_fingerprint",
        "df_jev_sentiment_rows",
        "df_jev_sentiment_grouped_rows",
        "df_jev_sentiment_unique",
        "__last_jev_sentiment_batch_summary__",
        "rapid_sentiment_observation_output",
        "rapid_sentiment_observation_fingerprint",
        "rapid_sentiment_observation_scale",
        "rapid_sentiment_observation_include_nr",
        "rapid_tag_observation_output",
        "rapid_tag_observation_fingerprint",
        "rapid_tag_observation_mode",
        "rapid_tag_observation_include_other",
    ]:
        st.session_state.pop(key, None)
    init_jev_sentiment_state()
    st.session_state.jev_sentiment_section = "Prepare Sample"


def reset_jev_results() -> None:
    unique = st.session_state.get("df_jev_sentiment_unique", pd.DataFrame()).copy()
    if unique.empty:
        return
    for column in [column for column in unique.columns if column.startswith("Jev")]:
        if column in unique.columns:
            unique[column] = pd.NA
    unique = clear_rapid_review_columns(unique)
    st.session_state.df_jev_sentiment_unique = unique
    st.session_state.pop("__last_jev_sentiment_batch_summary__", None)
    st.session_state.pop("__last_rapid_review_batch_summary__", None)
    st.session_state.pop("rapid_review_target_batch", None)
    st.session_state.pop("rapid_review_target_source_count", None)


def format_sample_mode(mode: str) -> str:
    return {
        "reuse_sentiment_sample": "Reused Sentiment sample",
        "reuse_other_sample": "Reused tagging sample",
        "full": "Full eligible dataset",
        "representative": "Representative sample",
        "custom": "Custom sample",
    }.get(mode, str(mode))


def build_analysis_context_fingerprint(payload: dict) -> str:
    try:
        return json.dumps(payload or {}, ensure_ascii=False, sort_keys=True, default=str)
    except Exception:
        return str(payload or "")


def metric_counts() -> tuple[int, int, int, int, int, int]:
    source_rows = get_sentiment_source_rows(st.session_state.df_traditional)
    excluded_flags = get_qualitative_coverage_flag_exclusions(st.session_state)
    eligible_rows = apply_session_coverage_flag_policy(source_rows, st.session_state, excluded_flags)
    tag_definitions = st.session_state.get("jev_tag_definitions", {})
    unique = ensure_jev_sentiment_columns(
        st.session_state.get("df_jev_sentiment_unique", pd.DataFrame()),
        tag_definitions=tag_definitions,
    )
    if unique.empty:
        return len(eligible_rows), 0, 0, 0, 0, 0
    errors = unique["Jev Error"].astype("string").fillna("").str.strip()
    completed = (
        unique["Jev Sentiment"].astype("string").fillna("").str.strip().ne("")
        & unique["Jev 5-Way Sentiment"].astype("string").fillna("").str.strip().ne("")
        & unique["Jev Sentiment Score"].astype("string").fillna("").str.strip().ne("")
        & unique["Jev Mixture Label"].astype("string").fillna("").str.strip().ne("")
    )
    if build_jev_tag_question_map(tag_definitions):
        completed = completed & unique["Jev Best Tag"].astype("string").fillna("").str.strip().ne("")
        completed = completed & unique["Jev Tags"].astype("string").fillna("").str.strip().ne("")
    processed = int(completed.sum())
    error_count = int((errors != "").sum())
    remaining = int(((~completed) & (errors == "")).sum())
    sample_used = len(st.session_state.get("df_jev_sentiment_rows", pd.DataFrame()))
    grouped = len(unique)
    return len(eligible_rows), sample_used, grouped, processed, error_count, remaining


def percent_display(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for column in out.columns:
        if (
            "Probability" not in column
            and "Relevant Probability" not in column
            and not column.endswith("Confidence")
        ):
            continue
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce") * 100
    return out


def probability_column_config(df: pd.DataFrame) -> dict:
    config = {}
    for column in df.columns:
        if "Probability" in column or column.endswith("Confidence"):
            config[column] = st.column_config.NumberColumn(format="%.1f%%")
        elif column.endswith("Cost USD"):
            config[column] = st.column_config.NumberColumn(format="$%.6f")
        elif column.endswith("Score"):
            config[column] = st.column_config.NumberColumn(format="%.2f")
    return config


def safe_display_text(value: object) -> str:
    try:
        missing = pd.isna(value)
    except Exception:
        missing = False
    if isinstance(missing, bool) and missing:
        return ""
    return str(value).strip()


def display_model_name(model: str) -> str:
    text = str(model or "").strip()
    return text.split("/", 1)[1] if text.startswith("typesafe/") else text


def get_sentiment_color_mapping() -> dict[str, str]:
    return {
        "POSITIVE": "#2ecc71",
        "BALANCED": "#38bdf8",
        "NEUTRAL": "#f1c40f",
        "NEGATIVE": "#e74c3c",
        "VERY POSITIVE": "#0f9d58",
        "SOMEWHAT POSITIVE": "#72cc4a",
        "SOMEWHAT NEGATIVE": "#e67e22",
        "VERY NEGATIVE": "#8e1f1f",
        "NOT RELEVANT": "#6b7280",
    }


def filter_sentiment_distribution(dist: pd.DataFrame, order: list[str], include_not_relevant: bool) -> pd.DataFrame:
    return filter_jev_sentiment_distribution(dist, order=order, include_not_relevant=include_not_relevant)


def filter_exclusive_tag_distribution(dist: pd.DataFrame, include_other: bool) -> pd.DataFrame:
    return filter_jev_best_tag_distribution(dist, include_other=include_other)


def filter_all_applicable_tag_distribution(dist: pd.DataFrame, include_other: bool) -> pd.DataFrame:
    return filter_jev_all_applicable_tag_distribution(dist, include_other=include_other)


def build_donut_chart(
    dist: pd.DataFrame,
    *,
    label_column: str,
    count_column: str,
    color_domain: list[str] | None = None,
    color_range: list[str] | None = None,
    show_legend: bool = False,
) -> alt.Chart:
    working = dist.copy()
    total_count = pd.to_numeric(working.get(count_column, pd.Series(dtype=float)), errors="coerce").fillna(0).sum()
    if working.empty or total_count <= 0:
        working = pd.DataFrame({label_column: ["No data"], count_column: [1], "Share": [0.0]})
        color_domain = ["No data"]
        color_range = ["#374151"]
        show_legend = False
    working["Share Label"] = (pd.to_numeric(working.get("Share", 0), errors="coerce").fillna(0) * 100).map(lambda x: f"{x:.1f}%")
    if color_domain:
        order_lookup = {label: index for index, label in enumerate(color_domain)}
        working["__SortOrder"] = working[label_column].astype(str).map(order_lookup).fillna(len(order_lookup)).astype(int)
    else:
        working["__SortOrder"] = range(len(working))
    legend = (
        alt.Legend(
            orient="right",
            title=None,
            labelColor="#E5E7EB",
            labelLimit=180,
            symbolStrokeWidth=0,
        )
        if show_legend
        else None
    )
    color = alt.Color(f"{label_column}:N", legend=legend)
    if color_domain and color_range:
        color = alt.Color(f"{label_column}:N", scale=alt.Scale(domain=color_domain, range=color_range), legend=legend)
    return (
        alt.Chart(working)
        .mark_arc(innerRadius=52, outerRadius=92, stroke="#0f172a", strokeWidth=2)
        .encode(
            theta=alt.Theta(f"{count_column}:Q", stack=True),
            color=color,
            order=alt.Order("__SortOrder:Q", sort="ascending"),
            tooltip=[
                alt.Tooltip(f"{label_column}:N", title=label_column),
                alt.Tooltip(f"{count_column}:Q", format=",", title="Underlying stories"),
                alt.Tooltip("Grouped Stories:Q", format=",", title="Grouped stories")
                if "Grouped Stories" in working.columns
                else alt.Tooltip(f"{count_column}:Q", format=",", title="Count"),
                alt.Tooltip("Share:Q", format=".1%", title="Share"),
            ],
        )
        .properties(height=220)
        .configure_view(strokeWidth=0)
    )


def build_tag_bar_chart(tag_dist: pd.DataFrame, *, share_column: str = "Grouped Story Share") -> alt.Chart:
    working = tag_dist.copy()
    working["Share Label"] = (pd.to_numeric(working.get(share_column, 0), errors="coerce").fillna(0) * 100).map(lambda x: f"{x:.1f}%")
    y_order = working["Tag"].astype(str).tolist()
    base = alt.Chart(working).encode(
        y=alt.Y("Tag:N", sort=y_order, axis=alt.Axis(title=None, labelLimit=240, labelPadding=10))
    )
    bars = base.mark_bar(cornerRadiusEnd=3, color="#636E95").encode(
        x=alt.X("Count:Q", axis=alt.Axis(title=None, grid=True, tickMinStep=1)),
        tooltip=[
            "Tag",
            alt.Tooltip("Count:Q", format=",", title="Underlying stories"),
            alt.Tooltip("Grouped Stories:Q", format=",", title="Grouped stories"),
            alt.Tooltip(share_column + ":Q", format=".1%", title="% processed grouped stories"),
        ],
    )
    text = base.mark_text(
        align="left",
        baseline="middle",
        dx=6,
        color="#F8FAFC",
        fontWeight=600,
    ).encode(x="Count:Q", text="Share Label:N")
    return (
        (bars + text)
        .properties(height=max(220, 38 * len(working)))
        .configure_view(strokeWidth=0)
        .configure_axis(
            gridColor="rgba(148, 163, 184, 0.18)",
            domain=False,
            tickColor="rgba(148, 163, 184, 0.35)",
            labelColor="#E5E7EB",
            titleColor="#E5E7EB",
        )
    )


def format_distribution_table(dist: pd.DataFrame, label_column: str, share_column: str = "Share") -> pd.DataFrame:
    working = dist.copy()
    if share_column in working.columns:
        working["Share"] = (pd.to_numeric(working[share_column], errors="coerce").fillna(0) * 100).map(lambda x: f"{x:.1f}%")
    columns = [label_column, "Count", "Grouped Stories"]
    if "Share" in working.columns:
        columns.append("Share")
    return working.reindex(columns=columns).rename(
        columns={
            "Count": "Underlying stories",
            "Grouped Stories": "Grouped stories",
            "Share": "Share",
        }
    )


def rapid_insight_display_fields(prefix: str) -> set[str]:
    key = f"{prefix}_selected_fields"
    previous_key = f"{prefix}_previous_fields"
    if key not in st.session_state:
        st.session_state[key] = RAPID_INSIGHT_FIELD_OPTIONS.copy()
    if previous_key not in st.session_state:
        st.session_state[previous_key] = RAPID_INSIGHT_FIELD_OPTIONS.copy()

    child_fields = {"Outlet", "Date", "Media type", "Mentions", "Impressions", "Effective reach"}

    def normalize_fields() -> None:
        current = st.session_state.get(key, []) or []
        previous = st.session_state.get(previous_key, []) or []
        current_set = set(current)
        previous_set = set(previous)
        if "Examples" not in current_set and current_set & child_fields:
            if "Examples" in previous_set:
                current_set -= child_fields
            else:
                current_set.add("Examples")
        normalized = [field for field in RAPID_INSIGHT_FIELD_OPTIONS if field in current_set]
        st.session_state[key] = normalized
        st.session_state[previous_key] = normalized.copy()

    preset_col, fields_col = st.columns([0.18, 0.82], gap="small")
    with preset_col:
        all_col, none_col = st.columns(2, gap="small")
        with all_col:
            if st.button("All", key=f"{prefix}_select_all", use_container_width=True):
                st.session_state[key] = RAPID_INSIGHT_FIELD_OPTIONS.copy()
                st.session_state[previous_key] = RAPID_INSIGHT_FIELD_OPTIONS.copy()
                st.rerun()
        with none_col:
            if st.button("None", key=f"{prefix}_select_none", use_container_width=True):
                st.session_state[key] = []
                st.session_state[previous_key] = []
                st.rerun()
    with fields_col:
        st.pills(
            "Fields",
            options=RAPID_INSIGHT_FIELD_OPTIONS,
            selection_mode="multi",
            default=st.session_state.get(key, RAPID_INSIGHT_FIELD_OPTIONS),
            key=key,
            on_change=normalize_fields,
            label_visibility="collapsed",
        )
    selected = st.session_state.get(key, []) or []
    st.session_state[previous_key] = list(selected)
    return set(selected)


def render_rapid_example_blocks(examples: list[dict], selected_fields: set[str]) -> None:
    if "Examples" not in selected_fields:
        return
    blocks = build_linked_example_blocks_html(
        examples[:5],
        show_outlet="Outlet" in selected_fields,
        show_date="Date" in selected_fields,
        show_media_type="Media type" in selected_fields,
        show_mentions="Mentions" in selected_fields,
        show_impressions="Impressions" in selected_fields,
        show_effective_reach="Effective reach" in selected_fields,
    )
    if blocks:
        st.markdown(blocks, unsafe_allow_html=True)


def rapid_observation_current_fingerprint(
    *,
    family: str,
    unique_df: pd.DataFrame,
    scale: str = RAPID_SENTIMENT_SCALE_3_WAY,
    tag_mode: str = RAPID_TAG_REVIEW_MODE_BEST,
    include_not_relevant: bool = False,
    include_other: bool = False,
    tag_definitions: dict[str, str] | None = None,
    analysis_context: str = "",
    selected_prominence_column: str = "",
) -> str:
    if family == "sentiment":
        payload = build_rapid_sentiment_observation_payload(
            unique_df,
            scale=scale,
            include_not_relevant=include_not_relevant,
            selected_prominence_column=selected_prominence_column,
        )
        return rapid_observation_fingerprint(
            payload,
            settings={"family": "sentiment", "scale": scale, "include_not_relevant": include_not_relevant},
            analysis_context=analysis_context,
        )
    payload = build_rapid_tag_observation_payload(
        unique_df,
        mode=tag_mode,
        include_other=include_other,
        tag_definitions=tag_definitions,
        selected_prominence_column=selected_prominence_column,
    )
    return rapid_observation_fingerprint(
        payload,
        settings={"family": "tagging", "mode": tag_mode, "include_other": include_other, "tag_definitions": tag_definitions or {}},
        analysis_context=analysis_context,
    )


def rapid_unresolved_sentiment_quality_count(unique_df: pd.DataFrame) -> int:
    final = build_final_rapid_sentiment_frame(unique_df)
    effective = build_effective_rapid_sentiment_frame(unique_df)
    status = effective.get("Rapid Sentiment Resolution Status", pd.Series(index=effective.index, dtype="object")).fillna("").astype(str)
    source = final.get("Final Rapid Sentiment Source", pd.Series(index=final.index, dtype="object")).fillna("").astype(str)
    unresolved = status.isin({"Cross-model disagreement", "Cross-model partial agreement", "First-pass internal conflict"}) & ~source.str.contains("Human", case=False, na=False)
    return int(unresolved.sum())


def rapid_unresolved_tag_quality_count(unique_df: pd.DataFrame, tag_definitions: dict[str, str]) -> int:
    final = build_final_rapid_tag_frame(unique_df, tag_definitions=tag_definitions)
    effective = build_effective_rapid_tag_frame(unique_df, tag_definitions=tag_definitions)
    status = effective.get("Rapid Tag Resolution Status", pd.Series(index=effective.index, dtype="object")).fillna("").astype(str)
    source = final.get("Final Rapid Tag Source", pd.Series(index=final.index, dtype="object")).fillna("").astype(str)
    unresolved = status.isin({"Cross-model disagreement", "Cross-model partial agreement", "First-pass internal conflict"}) & ~source.str.contains("Human", case=False, na=False)
    return int(unresolved.sum())


def rapid_processed_mask(df: pd.DataFrame) -> pd.Series:
    if df is None or df.empty:
        return pd.Series(dtype=bool)
    return (
        df.get("Jev Sentiment", pd.Series(index=df.index, dtype="object"))
        .fillna("")
        .astype(str)
        .str.strip()
        .ne("")
    )


def build_rapid_review_candidates_for_mode(
    unique_df: pd.DataFrame,
    *,
    review_domain: str,
    review_mode: str,
    tag_review_mode: str = RAPID_TAG_REVIEW_MODE_BEST,
    tag_definitions: dict[str, str] | None = None,
) -> pd.DataFrame:
    if unique_df is None or unique_df.empty:
        return pd.DataFrame()

    working = ensure_rapid_human_review_columns(unique_df).copy()
    processed = rapid_processed_mask(working)
    sentiment_effective = build_effective_rapid_sentiment_frame(working)
    tag_effective = build_effective_rapid_tag_frame(working, tag_definitions=tag_definitions)
    final_sentiment = build_final_rapid_sentiment_frame(working)
    final_tags = build_final_rapid_tag_frame(working, tag_definitions=tag_definitions)
    working = pd.concat([working, sentiment_effective, tag_effective, final_sentiment, final_tags], axis=1)

    if review_domain == "Sentiment":
        review_state = working["Rapid Human Sentiment Review State"].fillna("").astype(str).str.strip()
        status = working["Rapid Sentiment Resolution Status"].fillna("").astype(str).str.strip()
        disagreement = status.isin(["Cross-model disagreement", "Cross-model partial agreement", "Second opinion error"])
        internal = status.eq("First-pass internal conflict")
        available = processed | review_state.ne("")
    else:
        review_state = rapid_tag_review_state_series(working, tag_review_mode)
        status = working["Rapid Tag Resolution Status"].fillna("").astype(str).str.strip()
        disagreement = status.isin(["Cross-model disagreement", "Cross-model partial agreement", "Second opinion error"])
        internal = status.eq("First-pass internal conflict")
        first_pass_column = "Jev Tags" if tag_review_mode == RAPID_TAG_REVIEW_MODE_APPLICABLE else "Jev Best Tag"
        available = (
            working.get(first_pass_column, pd.Series(index=working.index, dtype="object"))
            .fillna("")
            .astype(str)
            .str.strip()
            .ne("")
            | review_state.ne("")
        )

    if "Rapid Review Priority Tier" not in working.columns or "Rapid Review Priority Reason" not in working.columns:
        working["Rapid Review Priority Tier"] = ""
        working["Rapid Review Priority Reason"] = ""
    recommended = (
        internal
        | disagreement
        | working["Rapid Review Priority Tier"].fillna("").astype(str).str.contains("Tier 1|Tier 2|Tier 3|Tier 4", regex=True)
    )
    unresolved = review_state.eq("")

    if review_mode == "Recommended / flagged for review":
        pool = working[available & recommended].copy()
        if pool.empty:
            pool = working[available & unresolved].copy()
    elif review_mode == "Cross-model disagreements":
        pool = working[available & disagreement].copy()
    elif review_mode == "Internal conflicts":
        pool = working[available & internal].copy()
    elif review_mode.startswith("All unresolved"):
        pool = working[available & unresolved].copy()
    elif review_mode in {"All Rapid-labeled sentiment", "All Rapid-tagged coverage"}:
        pool = working[available].copy()
    else:
        pool = working.copy()

    if pool.empty:
        return pool

    for column in ["Group Count", "Effective Reach", "Mentions", "Impressions"]:
        pool[column] = pd.to_numeric(pool.get(column, 0), errors="coerce").fillna(0)
    status_priority = {
        "First-pass internal conflict": 0,
        "Cross-model disagreement": 1,
        "Cross-model partial agreement": 2,
        "Second opinion error": 3,
        "First pass only": 4,
        "Cross-model agreement": 5,
    }
    pool["_rapid_spot_status_priority"] = status.map(status_priority).fillna(6).loc[pool.index]
    pool["_rapid_spot_unreviewed"] = review_state.eq("").loc[pool.index]
    return (
        pool.sort_values(
            [
                "_rapid_spot_unreviewed",
                "_rapid_spot_status_priority",
                "Group Count",
                "Effective Reach",
                "Mentions",
                "Impressions",
            ],
            ascending=[False, True, False, False, False, False],
        )
        .reset_index(drop=True)
    )


def rapid_sentiment_button(label: str, *, group_id: int, scale: str) -> bool:
    style_key = f"spot_btn_{label.replace(' ', '_').lower()}_rapid_{group_id}_{scale}"
    with st.container(key=style_key):
        return st.button(
            label,
            key=f"rapid_assign_sentiment_{group_id}_{scale}_{label}",
            use_container_width=True,
        )


def rapid_evidence_line(label: str, value: object) -> None:
    text = safe_display_text(value)
    if text:
        st.caption(f"{label}: {text}")


def default_rapid_human_tag_review_mode(configured_mode: str) -> str:
    return (
        RAPID_TAG_REVIEW_MODE_APPLICABLE
        if configured_mode == LEGACY_RAPID_TAGGING_MODE_APPLICABLE
        else RAPID_TAG_REVIEW_MODE_BEST
    )


def rapid_tag_review_state_series(unique_df: pd.DataFrame, review_mode: str) -> pd.Series:
    column = (
        "Rapid Human Applicable Tags Review State"
        if review_mode == RAPID_TAG_REVIEW_MODE_APPLICABLE
        else "Rapid Human Best Tag Review State"
    )
    return unique_df.get(column, pd.Series(index=unique_df.index, dtype="object")).fillna("").astype(str).str.strip()


def rapid_tag_review_state_for_row(row: pd.Series, review_mode: str) -> str:
    column = (
        "Rapid Human Applicable Tags Review State"
        if review_mode == RAPID_TAG_REVIEW_MODE_APPLICABLE
        else "Rapid Human Best Tag Review State"
    )
    state = safe_display_text(row.get(column, ""))
    if state:
        return state

    legacy_mode = safe_display_text(row.get("Rapid Human Tagging Mode", ""))
    legacy_state = safe_display_text(row.get("Rapid Human Tag Review State", ""))
    if not legacy_state:
        return ""
    if review_mode == RAPID_TAG_REVIEW_MODE_APPLICABLE and legacy_mode == LEGACY_RAPID_TAGGING_MODE_APPLICABLE:
        return legacy_state
    if review_mode == RAPID_TAG_REVIEW_MODE_BEST and legacy_mode == LEGACY_RAPID_TAGGING_MODE_BEST:
        return legacy_state
    return ""


def _clear_rapid_human_review(
    unique_df: pd.DataFrame,
    group_id: int,
    *,
    review_domain: str,
    tag_review_mode: str | None = None,
) -> pd.DataFrame:
    out = ensure_rapid_human_review_columns(unique_df)
    mask = out["Group ID"] == group_id
    if review_domain == "Sentiment":
        columns = [
            column
            for column in out.columns
            if (
                column.startswith("Rapid Human Sentiment")
                or column.startswith("Rapid Accepted Machine Sentiment")
                or column == "Rapid Accepted Machine Relevance"
            )
        ]
    elif tag_review_mode == RAPID_TAG_REVIEW_MODE_APPLICABLE:
        columns = [
            "Rapid Human Applicable Tags Review State",
            "Rapid Human Applicable Tags Assignment",
            "Rapid Human Applicable Tags Source",
            "Rapid Accepted Machine Tags",
        ]
    elif tag_review_mode == RAPID_TAG_REVIEW_MODE_BEST:
        columns = [
            "Rapid Human Best Tag Review State",
            "Rapid Human Best Tag Assignment",
            "Rapid Human Best Tag Source",
            "Rapid Accepted Machine Best Tag",
        ]
    else:
        columns = [
            column
            for column in out.columns
            if (
                column.startswith("Rapid Human Tag")
                or column.startswith("Rapid Accepted Machine Best Tag")
                or column.startswith("Rapid Accepted Machine Tags")
            )
        ]
    columns = [column for column in columns if column in out.columns]
    out.loc[mask, columns] = pd.NA
    return out


st.title("Rapid Labeling")
st.caption("Quickly apply structured sentiment and tagging labels to grouped stories.")
st.markdown(
    """
    <style>
    [class*="spot_btn_positive"] button,
    [class*="spot_btn_very_positive"] button,
    [class*="spot_btn_somewhat_positive"] button,
    [class*="spot_btn_balanced"] button,
    [class*="spot_btn_neutral"] button,
    [class*="spot_btn_somewhat_negative"] button,
    [class*="spot_btn_negative"] button,
    [class*="spot_btn_very_negative"] button,
    [class*="spot_btn_not_relevant"] button {
        width: 100%;
        color: black !important;
        border: 0 !important;
        padding: 0.16rem 0.6rem !important;
        font-weight: 700 !important;
        font-size: 14px !important;
        border-radius: 5px !important;
        margin-bottom: 0 !important;
        box-shadow: none !important;
    }
    [class*="spot_btn_positive"],
    [class*="spot_btn_very_positive"],
    [class*="spot_btn_somewhat_positive"],
    [class*="spot_btn_balanced"],
    [class*="spot_btn_neutral"],
    [class*="spot_btn_somewhat_negative"],
    [class*="spot_btn_negative"],
    [class*="spot_btn_very_negative"],
    [class*="spot_btn_not_relevant"] {
        margin-bottom: 0 !important;
        margin-bottom: -5px !important;
    }
    [class*="spot_btn_positive"] div[data-testid="stButton"],
    [class*="spot_btn_very_positive"] div[data-testid="stButton"],
    [class*="spot_btn_somewhat_positive"] div[data-testid="stButton"],
    [class*="spot_btn_balanced"] div[data-testid="stButton"],
    [class*="spot_btn_neutral"] div[data-testid="stButton"],
    [class*="spot_btn_somewhat_negative"] div[data-testid="stButton"],
    [class*="spot_btn_negative"] div[data-testid="stButton"],
    [class*="spot_btn_very_negative"] div[data-testid="stButton"],
    [class*="spot_btn_not_relevant"] div[data-testid="stButton"] {
        margin: 0 !important;
    }
    [class*="spot_btn_positive"] div[data-testid="stVerticalBlock"],
    [class*="spot_btn_very_positive"] div[data-testid="stVerticalBlock"],
    [class*="spot_btn_somewhat_positive"] div[data-testid="stVerticalBlock"],
    [class*="spot_btn_balanced"] div[data-testid="stVerticalBlock"],
    [class*="spot_btn_neutral"] div[data-testid="stVerticalBlock"],
    [class*="spot_btn_somewhat_negative"] div[data-testid="stVerticalBlock"],
    [class*="spot_btn_negative"] div[data-testid="stVerticalBlock"],
    [class*="spot_btn_very_negative"] div[data-testid="stVerticalBlock"],
    [class*="spot_btn_not_relevant"] div[data-testid="stVerticalBlock"] {
        gap: 0 !important;
    }
    [class*="spot_btn_positive"] button { background-color: #2ecc71 !important; }
    [class*="spot_btn_very_positive"] button { background-color: #10ad82 !important; }
    [class*="spot_btn_somewhat_positive"] button { background-color: #72cc4a !important; }
    [class*="spot_btn_balanced"] button { background-color: #38bdf8 !important; }
    [class*="spot_btn_neutral"] button { background-color: #f1c40f !important; }
    [class*="spot_btn_somewhat_negative"] button { background-color: #e67e22 !important; }
    [class*="spot_btn_negative"] button { background-color: #e74c3c !important; }
    [class*="spot_btn_very_negative"] button { background-color: #c0392b !important; }
    [class*="spot_btn_not_relevant"] button { background-color: #7f8c8d !important; }
    [class*="spot_btn_positive"] button:hover,
    [class*="spot_btn_very_positive"] button:hover,
    [class*="spot_btn_somewhat_positive"] button:hover,
    [class*="spot_btn_balanced"] button:hover,
    [class*="spot_btn_neutral"] button:hover,
    [class*="spot_btn_somewhat_negative"] button:hover,
    [class*="spot_btn_negative"] button:hover,
    [class*="spot_btn_very_negative"] button:hover,
    [class*="spot_btn_not_relevant"] button:hover {
        color: black !important;
        filter: brightness(0.98);
    }
    [class*="spot_btn_positive"] div[data-testid="element-container"],
    [class*="spot_btn_very_positive"] div[data-testid="element-container"],
    [class*="spot_btn_somewhat_positive"] div[data-testid="element-container"],
    [class*="spot_btn_balanced"] div[data-testid="element-container"],
    [class*="spot_btn_neutral"] div[data-testid="element-container"],
    [class*="spot_btn_somewhat_negative"] div[data-testid="element-container"],
    [class*="spot_btn_negative"] div[data-testid="element-container"],
    [class*="spot_btn_very_negative"] div[data-testid="element-container"],
    [class*="spot_btn_not_relevant"] div[data-testid="element-container"] {
        margin: 0 !important;
        padding: 0 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
init_jev_sentiment_state()
rapid_help_step_labels = {
    "Prepare Sample": "Prepare & Configure",
    "Run Jev Analysis": "Run Rapid Labeling",
    "AI Second Opinion": "AI Second Opinion",
    "Spot Checks": "Spot Checks",
    "Insights": "Insights",
}
set_page_help_context(
    st.session_state,
    "Rapid Labeling",
    rapid_help_step_labels.get(st.session_state.jev_sentiment_section, ""),
)

if not st.session_state.get("standard_step", False):
    st.error("Please complete Basic Cleaning before trying this step.")
    st.stop()

if not has_saved_analysis_context(st.session_state):
    st.warning(build_analysis_context_required_message("Rapid Labeling"))
    st.stop()

step1, step2, step3, step4, step5 = st.columns(5, gap="small")
with step1:
    if st.button(
        "1. Prepare & Configure",
        key="jev_sentiment_nav_prepare",
        use_container_width=True,
        type="primary" if st.session_state.jev_sentiment_section == "Prepare Sample" else "secondary",
    ):
        st.session_state.jev_sentiment_section = "Prepare Sample"
        st.rerun()
with step2:
    if st.button(
        "2. Run Rapid Labeling",
        key="jev_sentiment_nav_run",
        use_container_width=True,
        type="primary" if st.session_state.jev_sentiment_section == "Run Jev Analysis" else "secondary",
        disabled=not st.session_state.get("jev_sentiment_config_step", False),
    ):
        st.session_state.jev_sentiment_section = "Run Jev Analysis"
        st.rerun()
with step3:
    if st.button(
        "3. AI Second Opinion",
        key="jev_sentiment_nav_review",
        use_container_width=True,
        type="primary" if st.session_state.jev_sentiment_section == "AI Second Opinion" else "secondary",
        disabled=not st.session_state.get("jev_sentiment_config_step", False),
    ):
        st.session_state.jev_sentiment_section = "AI Second Opinion"
        st.rerun()
with step4:
    if st.button(
        "4. Spot Checks",
        key="jev_sentiment_nav_spot_checks",
        use_container_width=True,
        type="primary" if st.session_state.jev_sentiment_section == "Spot Checks" else "secondary",
        disabled=not st.session_state.get("jev_sentiment_config_step", False),
    ):
        st.session_state.jev_sentiment_section = "Spot Checks"
        st.rerun()
with step5:
    if st.button(
        "5. Insights",
        key="jev_sentiment_nav_insights",
        use_container_width=True,
        type="primary" if st.session_state.jev_sentiment_section == "Insights" else "secondary",
        disabled=not st.session_state.get("jev_sentiment_config_step", False),
    ):
        st.session_state.jev_sentiment_section = "Insights"
        st.rerun()

source_rows = get_sentiment_source_rows(st.session_state.df_traditional)
analysis_payload = get_analysis_context_payload(st.session_state)

if st.session_state.jev_sentiment_section == "Prepare Sample":
    st.caption("Sampling mirrors the production Sentiment setup, then Rapid Labeling writes only separate Rapid result columns.")

    reusable_sentiment_sample = st.session_state.get("df_sentiment_rows", None)
    reusable_sentiment_grouped = st.session_state.get("df_sentiment_grouped_rows", None)
    reusable_sentiment_unique = st.session_state.get("df_sentiment_unique", None)
    has_reusable_sentiment = (
        isinstance(reusable_sentiment_sample, pd.DataFrame)
        and not reusable_sentiment_sample.empty
        and isinstance(reusable_sentiment_grouped, pd.DataFrame)
        and not reusable_sentiment_grouped.empty
        and isinstance(reusable_sentiment_unique, pd.DataFrame)
        and not reusable_sentiment_unique.empty
    )
    reusable_tagging_sample = st.session_state.get("df_tagging_rows", None)
    has_reusable = isinstance(reusable_tagging_sample, pd.DataFrame) and not reusable_tagging_sample.empty

    options = [
        "Use full eligible dataset",
        "Use representative sample",
        "Set custom sample size",
    ]
    if has_reusable_sentiment:
        options.insert(0, "Reuse prepared Sentiment sample")
    if has_reusable:
        insert_at = 1 if has_reusable_sentiment else 0
        options.insert(insert_at, "Reuse tagging sample")

    excluded_flags = get_qualitative_coverage_flag_exclusions(st.session_state)
    working_source_rows = apply_session_coverage_flag_policy(source_rows, st.session_state, excluded_flags)
    population_size = len(working_source_rows)
    default_index = 1 if population_size > DEFAULT_MAX_FULL_ROWS else 0
    if has_reusable_sentiment:
        default_index = 0
    elif has_reusable:
        default_index = 1 if has_reusable_sentiment else 0

    mode_label = st.radio(
        "Rapid Labeling dataset mode",
        options=options,
        index=default_index,
    )
    if mode_label == "Reuse prepared Sentiment sample":
        sample_mode = "reuse_sentiment_sample"
    elif mode_label == "Reuse tagging sample":
        sample_mode = "reuse_other_sample"
    elif mode_label == "Use full eligible dataset":
        sample_mode = "full"
    elif mode_label == "Use representative sample":
        sample_mode = "representative"
    else:
        sample_mode = "custom"

    if sample_mode not in {"reuse_other_sample", "reuse_sentiment_sample"} and excluded_flags:
        st.caption(format_qualitative_exclusion_caption(excluded_flags))

    recommended_sample = calculate_representative_sample_size(population_size) if population_size > 0 else 0
    custom_sample_size = None
    if sample_mode == "custom":
        custom_sample_size = st.number_input(
            "Custom sample size",
            min_value=1,
            max_value=max(1, population_size),
            value=min(400, max(1, population_size)),
            step=1,
        )

    full_override = False
    if sample_mode == "full" and population_size > DEFAULT_MAX_FULL_ROWS:
        st.warning(
            f"Full Rapid Labeling is limited to {DEFAULT_MAX_FULL_ROWS:,} row-level mentions by default."
        )
        full_override = st.checkbox(
            f"I understand the risk and want to allow full Rapid Labeling over {DEFAULT_MAX_FULL_ROWS:,} mentions",
            value=False,
        )

    if sample_mode == "reuse_sentiment_sample":
        preview_rows = len(reusable_sentiment_sample) if has_reusable_sentiment else 0
    elif sample_mode == "reuse_other_sample":
        preview_rows = len(reusable_tagging_sample) if has_reusable else 0
    elif sample_mode == "full":
        preview_rows = population_size if (population_size <= DEFAULT_MAX_FULL_ROWS or full_override) else DEFAULT_MAX_FULL_ROWS
    elif sample_mode == "representative":
        preview_rows = recommended_sample
    else:
        preview_rows = int(custom_sample_size or 0)

    stat1, stat2 = st.columns(2)
    with stat1:
        st.metric("Eligible mentions", f"{population_size:,}")
    with stat2:
        st.metric("Estimated mentions used", f"{preview_rows:,}")

    if sample_mode == "representative":
        st.caption(f"Representative sample size estimate: {recommended_sample:,}")
    if sample_mode == "reuse_other_sample" and has_reusable:
        st.caption(f"Using existing tagging sample: {len(reusable_tagging_sample):,} rows")
    if sample_mode == "reuse_sentiment_sample" and has_reusable_sentiment:
        st.caption(
            f"Using prepared Sentiment workflow sample: {len(reusable_sentiment_sample):,} rows, "
            f"{len(reusable_sentiment_unique):,} grouped stories"
        )

    st.divider()
    st.write("**Analysis context**")
    caption = build_analysis_context_caption(st.session_state)
    if caption:
        st.caption(caption)
    if analysis_payload.get("general_guidance"):
        st.caption(f"Shared guidance: {analysis_payload['general_guidance']}")
    if analysis_payload.get("sentiment_guidance"):
        st.caption(f"Sentiment-specific guidance: {analysis_payload['sentiment_guidance']}")

    st.divider()
    st.write("**Optional Rapid Tags**")
    st.caption(
        f"One tag per line: `Tag Name: rationale`. Leave blank for sentiment-only Rapid Labeling. "
        f"`{RESERVED_OTHER_TAG}` is added automatically when tags are configured."
    )
    existing_tags_text = remove_reserved_tag_from_text(st.session_state.get("jev_tags_text", ""))
    tags_text = st.text_area(
        "Define Rapid Tags",
        value=existing_tags_text,
        height=180,
        key="jev_tags_text_input",
        help="One per line, in the format: TagName: rationale",
    )
    parsed_tag_definitions = {}
    tag_validation_error = ""
    if str(tags_text or "").strip():
        try:
            parsed_tag_definitions = parse_jev_tag_definitions(tags_text)
            explicit_count = len(build_jev_tag_question_map(parsed_tag_definitions))
            st.caption(
                f"{explicit_count:,} explicit tag(s) configured. "
                f"`{RESERVED_OTHER_TAG}` fallback: {RESERVED_OTHER_DEFINITION}"
            )
        except ValueError as exc:
            tag_validation_error = str(exc)
            st.error(tag_validation_error)

    if parsed_tag_definitions:
        current_mode = st.session_state.get("jev_tagging_mode", DEFAULT_JEV_TAGGING_MODE)
        tagging_mode = st.radio(
            "Tagging mode",
            list(JEV_TAGGING_MODES),
            index=list(JEV_TAGGING_MODES).index(current_mode)
            if current_mode in JEV_TAGGING_MODES
            else 0,
        )
    else:
        tagging_mode = DEFAULT_JEV_TAGGING_MODE

    prep_clicked = st.button("Prepare Rapid Labeling Sample", type="primary")
    if prep_clicked:
        if tag_validation_error:
            st.error("Fix the Rapid Tag definitions before preparing.")
            st.stop()
        primary_name = str(analysis_payload.get("primary_name", "") or "").strip()
        if not primary_name:
            st.warning("Add at least one Primary name in Analysis Context before preparing the sample.")
            st.stop()

        start = time.time()
        if sample_mode == "reuse_sentiment_sample":
            if not has_reusable_sentiment:
                st.warning("No prepared Sentiment workflow sample is available to reuse.")
                st.stop()
            sampled_rows = reusable_sentiment_sample.copy()
            grouped = reusable_sentiment_grouped.copy()
            unique = reusable_sentiment_unique.copy()
            sample_size_used = len(sampled_rows)
        else:
            reused_rows = reusable_tagging_sample if sample_mode == "reuse_other_sample" else None
            results = prepare_sentiment_datasets(
                df_traditional=st.session_state.df_traditional,
                sample_mode=sample_mode,
                excluded_flags=excluded_flags if sample_mode != "reuse_other_sample" else [],
                custom_sample_size=custom_sample_size,
                max_full_rows=DEFAULT_MAX_FULL_ROWS,
                full_override=full_override,
                reused_rows=reused_rows,
            )
            grouped, unique = ensure_ai_sentiment_columns(
                results["df_sentiment_grouped_rows"],
                results["df_sentiment_unique"],
            )
            sampled_rows = results["df_sentiment_rows"]
            sample_size_used = results["sample_size_used"]

        previous_fingerprint = str(st.session_state.get("jev_tag_config_fingerprint", "") or "")
        next_fingerprint = build_jev_tag_config_fingerprint(parsed_tag_definitions)

        st.session_state.df_jev_sentiment_rows = sampled_rows.copy()
        st.session_state.df_jev_sentiment_grouped_rows = grouped.copy()
        prepared_unique = unique.copy()
        for column in [
            column
            for column in prepared_unique.columns
            if column.startswith("Jev") or column.startswith("Rapid Review") or column.startswith("Rapid Sentiment") or column.startswith("Rapid Tag")
        ]:
            prepared_unique = prepared_unique.drop(columns=[column])
        for scheme in JEV_COMBINED_SCHEMES:
            prepared_unique = ensure_jev_sentiment_columns(prepared_unique, scheme, tag_definitions=parsed_tag_definitions)
        prepared_unique = ensure_rapid_review_columns(prepared_unique)
        prepared_unique = ensure_rapid_human_review_columns(prepared_unique)
        st.session_state.df_jev_sentiment_unique = prepared_unique
        st.session_state.jev_tags_text = remove_reserved_tag_from_text(tags_text)
        st.session_state.jev_tag_definitions = parsed_tag_definitions
        st.session_state.jev_tagging_mode = tagging_mode
        st.session_state.jev_tag_config_fingerprint = next_fingerprint
        st.session_state.jev_analysis_context_fingerprint = build_analysis_context_fingerprint(analysis_payload)
        st.session_state.jev_sentiment_sample_mode = sample_mode
        st.session_state.jev_sentiment_sample_size = sample_size_used
        st.session_state.jev_sentiment_full_override = full_override
        st.session_state.jev_sentiment_excluded_flags = (
            excluded_flags if sample_mode not in {"reuse_other_sample", "reuse_sentiment_sample"} else []
        )
        st.session_state.jev_sentiment_elapsed_time = time.time() - start
        st.session_state.jev_sentiment_config_step = True
        st.session_state.pop("__last_jev_sentiment_batch_summary__", None)
        st.session_state.pop("__last_rapid_review_batch_summary__", None)
        st.session_state.pop("rapid_review_target_batch", None)
        st.session_state.pop("rapid_review_target_source_count", None)
        if previous_fingerprint and previous_fingerprint != next_fingerprint:
            st.toast("Rapid Tag configuration changed; previous Rapid results were reset.")
        st.session_state.jev_sentiment_section = "Run Jev Analysis"
        st.rerun()

    if st.session_state.get("jev_sentiment_config_step", False):
        st.info("A Rapid Labeling sample is already prepared. You can prepare again or move to Run Rapid Labeling.")

    st.stop()

if not st.session_state.get("jev_sentiment_config_step", False):
    st.info("Prepare a Rapid Labeling sample before running.")
    st.stop()

tag_definitions = st.session_state.get("jev_tag_definitions", {})
tagging_enabled = bool(build_jev_tag_question_map(tag_definitions))
current_tagging_mode = st.session_state.get("jev_tagging_mode", DEFAULT_JEV_TAGGING_MODE)
st.session_state.df_jev_sentiment_unique = ensure_rapid_human_review_columns(
    st.session_state.get("df_jev_sentiment_unique", pd.DataFrame())
)

if st.session_state.jev_sentiment_section == "Insights":
    st.caption("Generate report-ready observations from the current final/effective Rapid labels.")

    unique_for_insights = st.session_state.df_jev_sentiment_unique.copy()
    prepared_count = len(unique_for_insights)
    selected_prominence_column = get_analysis_context_payload(st.session_state).get("selected_prominence_column", "")

    sentiment_default = st.session_state.get("rapid_human_sentiment_scale", RAPID_SENTIMENT_SCALE_3_WAY)
    if sentiment_default not in {RAPID_SENTIMENT_SCALE_3_WAY, RAPID_SENTIMENT_SCALE_5_WAY}:
        sentiment_default = RAPID_SENTIMENT_SCALE_3_WAY
    tag_default = st.session_state.get(
        "rapid_human_tag_review_mode",
        default_rapid_human_tag_review_mode(current_tagging_mode),
    )
    if tag_default not in {RAPID_TAG_REVIEW_MODE_BEST, RAPID_TAG_REVIEW_MODE_APPLICABLE}:
        tag_default = default_rapid_human_tag_review_mode(current_tagging_mode)

    sentiment_scale = st.radio(
        "Sentiment insight scale",
        [RAPID_SENTIMENT_SCALE_3_WAY, RAPID_SENTIMENT_SCALE_5_WAY],
        index=[RAPID_SENTIMENT_SCALE_3_WAY, RAPID_SENTIMENT_SCALE_5_WAY].index(sentiment_default),
        horizontal=True,
        key="rapid_insight_sentiment_scale",
    )
    include_not_relevant = st.toggle(
        "Include Not Relevant in sentiment percentages",
        value=False,
        key="rapid_insight_include_not_relevant",
    )

    sentiment_frame = build_rapid_sentiment_insight_frame(unique_for_insights, scale=sentiment_scale)
    if not include_not_relevant and not sentiment_frame.empty:
        sentiment_usable = sentiment_frame[sentiment_frame["Rapid Insight Sentiment"].ne("NOT RELEVANT")].copy()
    else:
        sentiment_usable = sentiment_frame.copy()
    sentiment_usable_count = len(sentiment_usable)
    unresolved_sentiment = rapid_unresolved_sentiment_quality_count(unique_for_insights)

    st.write("**Sentiment Insights**")
    sent_metric_cols = st.columns(4)
    sent_metric_cols[0].metric("Prepared stories", f"{prepared_count:,}")
    sent_metric_cols[1].metric("Usable sentiment", f"{sentiment_usable_count:,}")
    sent_metric_cols[2].metric("Not usable", f"{max(0, prepared_count - sentiment_usable_count):,}")
    sent_metric_cols[3].metric("Unresolved QA flags", f"{unresolved_sentiment:,}")
    if sentiment_usable_count:
        st.caption(
            f"{sentiment_usable_count:,} of {prepared_count:,} Rapid stories currently have usable "
            f"{sentiment_scale} sentiment for this view. Observations will use those stories."
        )
        if unresolved_sentiment:
            st.warning(
                f"{unresolved_sentiment:,} usable sentiment row(s) still have unresolved machine disagreement or conflict "
                "after the human/final layer. You can generate observations now, but more Spot Checks may improve quality."
            )
    else:
        st.info("No usable Rapid sentiment labels are available for the current sentiment view.")

    sentiment_dist = build_rapid_sentiment_distribution(unique_for_insights, scale=sentiment_scale)
    sentiment_dist = filter_sentiment_distribution(
        sentiment_dist,
        sentiment_order_for_scale(sentiment_scale),
        include_not_relevant,
    )
    sentiment_colors = get_sentiment_color_mapping()
    sentiment_order = [item for item in sentiment_order_for_scale(sentiment_scale) if include_not_relevant or item != "NOT RELEVANT"]
    sent_chart_col, sent_table_col = st.columns([1.35, 1], gap="large")
    with sent_chart_col:
        st.altair_chart(
            build_donut_chart(
                sentiment_dist,
                label_column="Sentiment",
                count_column="Count",
                color_domain=sentiment_order,
                color_range=[sentiment_colors.get(item, "#9ca3af") for item in sentiment_order],
            ),
            use_container_width=True,
        )
    with sent_table_col:
        st.dataframe(format_distribution_table(sentiment_dist, "Sentiment"), use_container_width=True, hide_index=True)

    sent_obs_col, sent_button_col = st.columns([2.6, 1.4], gap="medium")
    with sent_obs_col:
        st.subheader("Sentiment Observations")
    with sent_button_col:
        if st.button(
            "Generate sentiment observations",
            type="primary",
            key="rapid_generate_sentiment_observations",
            use_container_width=True,
            disabled=sentiment_usable_count == 0,
        ):
            try:
                api_key = st.secrets["key"]
            except Exception:
                api_key = ""
            if not api_key:
                st.warning("Add `key` to Streamlit secrets to generate Rapid sentiment observations.")
            else:
                with st.spinner("Generating Rapid sentiment observations..."):
                    try:
                        output, _in_tok, _out_tok, fingerprint = generate_rapid_sentiment_observations(
                            unique_for_insights,
                            client_name=str(st.session_state.get("client_name", "") or "").strip(),
                            scale=sentiment_scale,
                            include_not_relevant=include_not_relevant,
                            api_key=api_key,
                            analysis_context=build_sentiment_analysis_context_text(st.session_state),
                            selected_prominence_column=selected_prominence_column,
                        )
                        st.session_state.rapid_sentiment_observation_output = output
                        st.session_state.rapid_sentiment_observation_fingerprint = fingerprint
                        st.session_state.rapid_sentiment_observation_scale = sentiment_scale
                        st.session_state.rapid_sentiment_observation_include_nr = include_not_relevant
                        st.rerun()
                    except Exception as exc:
                        st.session_state.rapid_sentiment_observation_output = {"_error": str(exc)}
                        st.rerun()

    sentiment_output = st.session_state.get("rapid_sentiment_observation_output", {})
    if sentiment_output and sentiment_output.get("_error"):
        st.error(f"Could not generate Rapid sentiment observations: {sentiment_output['_error']}")
    elif sentiment_output:
        current_fingerprint = rapid_observation_current_fingerprint(
            family="sentiment",
            unique_df=unique_for_insights,
            scale=sentiment_scale,
            include_not_relevant=include_not_relevant,
            analysis_context=build_sentiment_analysis_context_text(st.session_state),
            selected_prominence_column=selected_prominence_column,
        )
        if st.session_state.get("rapid_sentiment_observation_fingerprint", "") != current_fingerprint:
            st.info("The current Rapid sentiment inputs differ from the generated observations. Regenerate to refresh this section.")
        selected_fields = rapid_insight_display_fields("rapid_sentiment_obs")
        overall = str(sentiment_output.get("overall_observation", "") or "").strip()
        if overall:
            st.markdown("### Overall Observations")
            st.write(overall)
        for section in sentiment_output.get("sentiment_sections", []):
            label = str(section.get("sentiment", "") or "").strip()
            observation = str(section.get("observation", "") or "").strip()
            if not label:
                continue
            st.markdown(f"### {html.escape(label.title())}", unsafe_allow_html=True)
            if observation:
                st.write(observation)
            render_rapid_example_blocks(
                sentiment_output.get("_examples_by_sentiment", {}).get(label, []),
                selected_fields,
            )

    if tagging_enabled:
        st.divider()
        st.write("**Tagging Insights**")
        tag_mode = st.radio(
            "Tag insight formulation",
            [RAPID_TAG_REVIEW_MODE_BEST, RAPID_TAG_REVIEW_MODE_APPLICABLE],
            index=[RAPID_TAG_REVIEW_MODE_BEST, RAPID_TAG_REVIEW_MODE_APPLICABLE].index(tag_default),
            horizontal=True,
            key="rapid_insight_tag_mode",
        )
        include_other = st.toggle(
            "Include Other in tag percentages",
            value=False,
            key="rapid_insight_include_other",
        )
        tag_frame = build_rapid_tag_insight_frame(unique_for_insights, mode=tag_mode, tag_definitions=tag_definitions)
        if not include_other and not tag_frame.empty:
            if tag_mode == RAPID_TAG_REVIEW_MODE_APPLICABLE:
                tag_usable_count = int(
                    tag_frame["Rapid Insight Tags"]
                    .fillna("")
                    .astype(str)
                    .apply(lambda value: any(tag.casefold() != RESERVED_OTHER_TAG.casefold() for tag in normalize_tag_list(value.replace(";", ","))))
                    .sum()
                )
            else:
                tag_usable_count = int(tag_frame["Rapid Insight Tags"].fillna("").astype(str).str.casefold().ne(RESERVED_OTHER_TAG.casefold()).sum())
        else:
            tag_usable_count = len(tag_frame)
        unresolved_tags = rapid_unresolved_tag_quality_count(unique_for_insights, tag_definitions)
        tag_metric_cols = st.columns(4)
        tag_metric_cols[0].metric("Prepared stories", f"{prepared_count:,}")
        tag_metric_cols[1].metric("Usable tags", f"{tag_usable_count:,}")
        tag_metric_cols[2].metric("Not usable", f"{max(0, prepared_count - tag_usable_count):,}")
        tag_metric_cols[3].metric("Unresolved QA flags", f"{unresolved_tags:,}")
        if tag_usable_count:
            st.caption(
                f"{tag_usable_count:,} of {prepared_count:,} Rapid stories currently have usable "
                f"{tag_mode.lower()} tag labels for this view."
            )
            if unresolved_tags:
                st.warning(
                    f"{unresolved_tags:,} usable tag row(s) still have unresolved machine disagreement or conflict "
                    "after the human/final layer. You can generate observations now, but more Spot Checks may improve quality."
                )
        else:
            st.info("No usable Rapid tag labels are available for the current tag view.")

        tag_dist = build_rapid_tag_distribution(
            unique_for_insights,
            mode=tag_mode,
            tag_definitions=tag_definitions,
            include_other=include_other,
        )
        tag_chart_col, tag_table_col = st.columns([1.35, 1], gap="large")
        with tag_chart_col:
            if tag_dist.empty:
                st.info("No Rapid tag distribution is available for this view.")
            elif tag_mode == RAPID_TAG_REVIEW_MODE_BEST:
                tag_color_domain, tag_color_range = build_jev_tag_color_scale(
                    [item["tag"] for item in build_jev_tag_question_map(tag_definitions)]
                    + tag_dist["Tag"].astype(str).tolist(),
                    include_other=include_other,
                )
                st.altair_chart(
                    build_donut_chart(
                        tag_dist,
                        label_column="Tag",
                        count_column="Count",
                        color_domain=tag_color_domain,
                        color_range=tag_color_range,
                        show_legend=True,
                    ),
                    use_container_width=True,
                )
            else:
                st.altair_chart(build_tag_bar_chart(tag_dist), use_container_width=True)
        with tag_table_col:
            if tag_dist.empty:
                st.dataframe(pd.DataFrame(columns=["Tag", "Underlying stories", "Grouped stories", "Share"]), hide_index=True)
            else:
                share_column = "Grouped Story Share" if tag_mode == RAPID_TAG_REVIEW_MODE_APPLICABLE else "Share"
                table = format_distribution_table(tag_dist, "Tag", share_column=share_column)
                if tag_mode == RAPID_TAG_REVIEW_MODE_APPLICABLE:
                    table = table.rename(columns={"Share": "% processed grouped stories"})
                st.dataframe(table, use_container_width=True, hide_index=True)

        tag_obs_col, tag_button_col = st.columns([2.6, 1.4], gap="medium")
        with tag_obs_col:
            st.subheader("Tag Observations")
        with tag_button_col:
            if st.button(
                "Generate tag observations",
                type="primary",
                key="rapid_generate_tag_observations",
                use_container_width=True,
                disabled=tag_usable_count == 0,
            ):
                try:
                    api_key = st.secrets["key"]
                except Exception:
                    api_key = ""
                if not api_key:
                    st.warning("Add `key` to Streamlit secrets to generate Rapid tag observations.")
                else:
                    with st.spinner("Generating Rapid tag observations..."):
                        try:
                            output, _in_tok, _out_tok, fingerprint = generate_rapid_tag_observations(
                                unique_for_insights,
                                client_name=str(st.session_state.get("client_name", "") or "").strip(),
                                mode=tag_mode,
                                include_other=include_other,
                                api_key=api_key,
                                analysis_context=build_analysis_context_text(st.session_state),
                                tag_definitions=tag_definitions,
                                selected_prominence_column=selected_prominence_column,
                            )
                            st.session_state.rapid_tag_observation_output = output
                            st.session_state.rapid_tag_observation_fingerprint = fingerprint
                            st.session_state.rapid_tag_observation_mode = tag_mode
                            st.session_state.rapid_tag_observation_include_other = include_other
                            st.rerun()
                        except Exception as exc:
                            st.session_state.rapid_tag_observation_output = {"_error": str(exc)}
                            st.rerun()

        tag_output = st.session_state.get("rapid_tag_observation_output", {})
        if tag_output and tag_output.get("_error"):
            st.error(f"Could not generate Rapid tag observations: {tag_output['_error']}")
        elif tag_output:
            current_fingerprint = rapid_observation_current_fingerprint(
                family="tagging",
                unique_df=unique_for_insights,
                tag_mode=tag_mode,
                include_other=include_other,
                tag_definitions=tag_definitions,
                analysis_context=build_analysis_context_text(st.session_state),
                selected_prominence_column=selected_prominence_column,
            )
            if st.session_state.get("rapid_tag_observation_fingerprint", "") != current_fingerprint:
                st.info("The current Rapid tag inputs differ from the generated observations. Regenerate to refresh this section.")
            selected_fields = rapid_insight_display_fields("rapid_tag_obs")
            overall = str(tag_output.get("overall_observation", "") or "").strip()
            if overall:
                st.markdown("### Overall Observations")
                st.write(overall)
            for section in tag_output.get("tag_sections", []):
                label = str(section.get("tag", "") or "").strip()
                observation = str(section.get("observation", "") or "").strip()
                if not label:
                    continue
                st.markdown(f"### {html.escape(label)}", unsafe_allow_html=True)
                if observation:
                    st.write(observation)
                render_rapid_example_blocks(tag_output.get("_examples_by_tag", {}).get(label, []), selected_fields)
    else:
        st.info("Rapid Tags are not configured for this sample, so only sentiment insights are available.")

    st.stop()

if st.session_state.jev_sentiment_section == "Spot Checks":
    st.caption("Review, accept, and correct Rapid sentiment or tagging decisions on grouped stories.")

    review_domain = st.segmented_control(
        "Rapid review type",
        ["Sentiment", "Tagging"],
        default=st.session_state.get("rapid_spot_review_domain", "Sentiment"),
        format_func=lambda mode: f"{mode} review",
        key="rapid_spot_review_domain",
        label_visibility="collapsed",
    )
    if review_domain == "Tagging":
        default_tag_review_mode = default_rapid_human_tag_review_mode(current_tagging_mode)
        if st.session_state.get("rapid_human_tag_review_mode") not in {
            RAPID_TAG_REVIEW_MODE_BEST,
            RAPID_TAG_REVIEW_MODE_APPLICABLE,
        }:
            st.session_state.rapid_human_tag_review_mode = default_tag_review_mode
    view_options = (
        [
            "Recommended / flagged for review",
            "Cross-model disagreements",
            "Internal conflicts",
            "All unresolved sentiment",
            "All Rapid-labeled sentiment",
            "All Rapid sample coverage",
        ]
        if review_domain == "Sentiment"
        else [
            "Recommended / flagged for review",
            "Cross-model disagreements",
            "Internal conflicts",
            "All unresolved tagging",
            "All Rapid-tagged coverage",
            "All Rapid sample coverage",
        ]
    )
    if st.session_state.get("rapid_spot_review_view") not in view_options:
        st.session_state.rapid_spot_review_view = view_options[0]

    unique_for_review = st.session_state.df_jev_sentiment_unique.copy()
    sentiment_state = unique_for_review["Rapid Human Sentiment Review State"].fillna("").astype(str).str.strip()
    tag_review_mode = st.session_state.get(
        "rapid_human_tag_review_mode",
        default_rapid_human_tag_review_mode(current_tagging_mode),
    )
    tag_state = rapid_tag_review_state_series(unique_for_review, tag_review_mode)
    active_state = sentiment_state if review_domain == "Sentiment" else tag_state
    processed_mask = rapid_processed_mask(unique_for_review)
    accepted_count = int(active_state.eq(HUMAN_STATE_ACCEPTED).sum())
    assigned_count = int(active_state.eq(HUMAN_STATE_ASSIGNED).sum())
    reviewed_count = accepted_count + assigned_count
    rapid_labeled = int(processed_mask.sum())

    metric_cols = st.columns(5)
    metric_cols[0].metric("Rapid-labeled", f"{rapid_labeled:,}")
    metric_cols[1].metric("Human-reviewed", f"{reviewed_count:,}")
    metric_cols[2].metric("Accepted machine", f"{accepted_count:,}")
    metric_cols[3].metric("Assigned", f"{assigned_count:,}")

    view_col, scale_col = st.columns([1.25, 1], gap="medium")
    with view_col:
        review_view = st.selectbox("Spot check view", view_options, key="rapid_spot_review_view")
    sentiment_scale = HUMAN_SENTIMENT_SCALE_3_WAY
    with scale_col:
        if review_domain == "Sentiment":
            sentiment_scale = st.radio(
                "Human sentiment review scale",
                [HUMAN_SENTIMENT_SCALE_3_WAY, HUMAN_SENTIMENT_SCALE_5_WAY],
                horizontal=True,
                key="rapid_human_sentiment_scale",
            )
        else:
            tag_review_mode = st.radio(
                "Human tagging review mode",
                [RAPID_TAG_REVIEW_MODE_BEST, RAPID_TAG_REVIEW_MODE_APPLICABLE],
                horizontal=True,
                key="rapid_human_tag_review_mode",
            )

    candidates = build_rapid_review_candidates_for_mode(
        unique_for_review,
        review_domain=review_domain,
        review_mode=review_view,
        tag_review_mode=tag_review_mode,
        tag_definitions=tag_definitions,
    )
    metric_cols[4].metric("In review queue", f"{len(candidates):,}")
    st.caption(
        "Recommended views focus the queue; all-coverage views let you audit or manually label broader Rapid sample rows."
    )

    if candidates.empty:
        st.info("No grouped stories match the current view.")
        st.stop()

    idx_key = "rapid_spot_sentiment_idx" if review_domain == "Sentiment" else "rapid_spot_tagging_idx"
    st.session_state[idx_key] = min(int(st.session_state.get(idx_key, 0) or 0), len(candidates) - 1)
    idx = int(st.session_state[idx_key])
    row = candidates.iloc[idx]
    current_group_id = int(row["Group ID"])

    head_raw = safe_display_text(row.get("Headline", ""))
    body_raw = safe_display_text(row.get("Example Snippet", row.get("Snippet", "")))
    trans_head = safe_display_text(row.get("Translated Headline", ""))
    trans_body = safe_display_text(row.get("Translated Body", ""))
    headline = trans_head or head_raw
    snippet = trans_body or body_raw
    url = safe_display_text(row.get("Example URL", row.get("URL", "")))
    keywords = build_rapid_highlight_keywords(analysis_payload)
    tolerant_pat_str = build_tolerant_regex_str(keywords)

    with st.sidebar:
        if st.button("Translate", key="rapid_spot_translate"):
            try:
                th = translate_text(head_raw) if head_raw else None
                tb = translate_text(body_raw) if body_raw else None
                unique2, grouped2 = apply_translation_to_group(
                    st.session_state.df_jev_sentiment_unique,
                    st.session_state.df_jev_sentiment_grouped_rows,
                    current_group_id,
                    th,
                    tb,
                )
                st.session_state.df_jev_sentiment_unique = ensure_rapid_human_review_columns(unique2)
                st.session_state.df_jev_sentiment_grouped_rows = grouped2
                st.rerun()
            except Exception as exc:
                st.error(f"Translation failed: {exc}")

    left, right = st.columns([3.8, 1.45], gap="large")
    with left:
        highlighted_head = highlight_with_tolerant_regex(
            escape_markdown(headline),
            tolerant_pat_str,
            keywords,
        )
        highlighted_body = highlight_with_tolerant_regex(
            escape_markdown(snippet),
            tolerant_pat_str,
            keywords,
        )
        st.markdown(f"#### {highlighted_head}", unsafe_allow_html=True)
        if snippet:
            st.markdown(highlighted_body, unsafe_allow_html=True)
        meta_bits = [
            safe_display_text(row.get("Date", "")),
            safe_display_text(row.get("Outlet", "")),
            safe_display_text(row.get("Type", "")),
            f"Group Count: {int(pd.to_numeric(pd.Series([row.get('Group Count', 1)]), errors='coerce').fillna(1).iloc[0]):,}",
        ]
        meta_line = " | ".join([part for part in meta_bits if part])
        if meta_line:
            st.caption(meta_line)
        if url:
            st.markdown(url)

    with right:
        if review_domain == "Sentiment":
            state = safe_display_text(row.get("Rapid Human Sentiment Review State", ""))
            if state:
                st.success(f"Human review: {state}")
            status = safe_display_text(row.get("Rapid Sentiment Resolution Status", ""))
            if status in {"Cross-model disagreement", "Cross-model partial agreement", "First-pass internal conflict"}:
                st.warning(status)
            elif status:
                st.info(status)

            active_jev_column = "Jev Sentiment" if sentiment_scale == HUMAN_SENTIMENT_SCALE_3_WAY else "Jev 5-Way Sentiment"
            active_luna_column = (
                "Rapid Review 3-Way Sentiment"
                if sentiment_scale == HUMAN_SENTIMENT_SCALE_3_WAY
                else "Rapid Review 5-Way Sentiment"
            )
            active_jev = safe_display_text(row.get(active_jev_column, ""))
            active_luna = safe_display_text(row.get(active_luna_column, ""))
            if active_jev:
                st.caption(f"First opinion: {active_jev}")
            if active_luna:
                st.caption(f"Second opinion: {active_luna}")

            st.write("**Assign human sentiment**")
            labels = (
                ["POSITIVE", "NEUTRAL", "NEGATIVE", "NOT RELEVANT"]
                if sentiment_scale == HUMAN_SENTIMENT_SCALE_3_WAY
                else [
                    "VERY POSITIVE",
                    "SOMEWHAT POSITIVE",
                    "NEUTRAL",
                    "SOMEWHAT NEGATIVE",
                    "VERY NEGATIVE",
                    "NOT RELEVANT",
                ]
            )
            for label in labels:
                if rapid_sentiment_button(label, group_id=current_group_id, scale=sentiment_scale):
                    st.session_state.df_jev_sentiment_unique = apply_rapid_sentiment_human_selection(
                        st.session_state.df_jev_sentiment_unique,
                        current_group_id,
                        scale=sentiment_scale,
                        label=label,
                    )
                    st.rerun()

            with st.expander("Sentiment evidence", expanded=False):
                st.write("**First opinion**")
                rapid_evidence_line("3-way", row.get("Jev Sentiment", ""))
                rapid_evidence_line("5-way", row.get("Jev 5-Way Sentiment", ""))
                rapid_evidence_line("Score", format_rapid_numeric(row.get("Jev Sentiment Score")))
                rapid_evidence_line("Relevance probability", format_rapid_numeric(row.get("Jev Relevant Probability")))
                rapid_evidence_line("Mixture", row.get("Jev Mixture Label", ""))

                luna_fields = [
                    safe_display_text(row.get("Rapid Review 3-Way Sentiment", "")),
                    safe_display_text(row.get("Rapid Review 5-Way Sentiment", "")),
                    safe_display_text(row.get("Rapid Review Sentiment Outcome", "")),
                    safe_display_text(row.get("Rapid Sentiment Score Distance", "")),
                ]
                if any(luna_fields):
                    st.write("**Second opinion**")
                    for label, column in [
                        ("3-way", "Rapid Review 3-Way Sentiment"),
                        ("5-way", "Rapid Review 5-Way Sentiment"),
                    ]:
                        rapid_evidence_line(label, row.get(column, ""))
                    score_distance = format_rapid_numeric(row.get("Rapid Sentiment Score Distance"))
                    if score_distance:
                        st.caption(f"Score distance: {score_distance}")

                machine_three = safe_display_text(row.get("Effective Rapid Sentiment 3-Way", ""))
                final_three = safe_display_text(row.get("Final Rapid Sentiment 3-Way", ""))
                machine_five = safe_display_text(row.get("Effective Rapid Sentiment 5-Way", ""))
                final_five = safe_display_text(row.get("Final Rapid Sentiment 5-Way", ""))
                changed = (machine_three and final_three and machine_three != final_three) or (
                    machine_five and final_five and machine_five != final_five
                )
                if changed:
                    st.write("**Final decision**")
                    if machine_three and final_three and machine_three != final_three:
                        st.caption(f"3-way: machine {machine_three} -> final {final_three}")
                    if machine_five and final_five and machine_five != final_five:
                        st.caption(f"5-way: machine {machine_five} -> final {final_five}")

                rationale = safe_display_text(row.get("Rapid Review Sentiment Rationale", ""))
                if rationale:
                    st.write("**Reasoning**")
                    st.caption(rationale)
        else:
            state = rapid_tag_review_state_for_row(row, tag_review_mode)
            if state:
                st.success(f"Human review: {state}")
            status = safe_display_text(row.get("Rapid Tag Resolution Status", ""))
            if status in {"Cross-model disagreement", "Cross-model partial agreement", "First-pass internal conflict"}:
                st.warning(status)
            elif status:
                st.info(status)

            first_active_tag = (
                safe_display_text(row.get("Jev Tags", ""))
                if tag_review_mode == RAPID_TAG_REVIEW_MODE_APPLICABLE
                else safe_display_text(row.get("Jev Best Tag", ""))
            )
            second_active_tag = (
                safe_display_text(row.get("Rapid Review Tags", ""))
                if tag_review_mode == RAPID_TAG_REVIEW_MODE_APPLICABLE
                else safe_display_text(row.get("Rapid Review Best Tag", ""))
            )
            if first_active_tag:
                st.caption(f"First opinion: {first_active_tag}")
            if second_active_tag:
                st.caption(f"Second opinion: {second_active_tag}")

            tag_names = list(build_jev_tag_question_map(tag_definitions))
            tag_names = [item["tag"] for item in tag_names]
            if RESERVED_OTHER_TAG not in tag_names:
                tag_names.append(RESERVED_OTHER_TAG)
            if tag_names:
                st.write("**Assign human tags**")
                selected_assignment = None
                if tag_review_mode == RAPID_TAG_REVIEW_MODE_APPLICABLE:
                    default_tags = normalize_tag_list(
                        safe_display_text(row.get("Rapid Human Applicable Tags Assignment", ""))
                        or safe_display_text(row.get("Effective Rapid Tags", ""))
                    )

                    def enforce_rapid_other_exclusivity(changed_label: str) -> None:
                        changed_key = f"rapid_tag_multi_{current_group_id}_{changed_label}"
                        if not st.session_state.get(changed_key, False):
                            return
                        if changed_label.casefold() == RESERVED_OTHER_TAG.casefold():
                            for other_label in tag_names:
                                if other_label.casefold() != RESERVED_OTHER_TAG.casefold():
                                    st.session_state[f"rapid_tag_multi_{current_group_id}_{other_label}"] = False
                        else:
                            for other_label in tag_names:
                                if other_label.casefold() == RESERVED_OTHER_TAG.casefold():
                                    st.session_state[f"rapid_tag_multi_{current_group_id}_{other_label}"] = False

                    for label in tag_names:
                        key = f"rapid_tag_multi_{current_group_id}_{label}"
                        if key not in st.session_state:
                            st.session_state[key] = label in default_tags
                        st.checkbox(label, key=key, on_change=enforce_rapid_other_exclusivity, args=(label,))
                    selected = [label for label in tag_names if st.session_state.get(f"rapid_tag_multi_{current_group_id}_{label}", False)]
                    if st.button("Save & next", key=f"rapid_assign_tags_{current_group_id}", use_container_width=True):
                        selected_assignment = ", ".join(selected)
                else:
                    for label in tag_names:
                        if st.button(label, key=f"rapid_assign_tag_{current_group_id}_{label}", use_container_width=True):
                            selected_assignment = label
                if selected_assignment:
                    st.session_state.df_jev_sentiment_unique = apply_rapid_tag_human_selection(
                        st.session_state.df_jev_sentiment_unique,
                        current_group_id,
                        assignment=selected_assignment,
                        tagging_mode=tag_review_mode,
                        tag_definitions=tag_definitions,
                    )
                    st.session_state[idx_key] = min(len(candidates) - 1, idx + 1)
                    st.rerun()

            with st.expander("Tagging evidence", expanded=False):
                st.write("**First opinion**")
                if tag_review_mode == RAPID_TAG_REVIEW_MODE_APPLICABLE:
                    rapid_evidence_line("Applicable tags", row.get("Jev Tags", ""))
                    rapid_evidence_line("Best-fit tag", row.get("Jev Best Tag", ""))
                else:
                    rapid_evidence_line("Best-fit tag", row.get("Jev Best Tag", ""))
                    rapid_evidence_line("Applicable tags", row.get("Jev Tags", ""))

                if safe_display_text(row.get("Rapid Review Best Tag", "")) or safe_display_text(row.get("Rapid Review Tags", "")):
                    st.write("**Second opinion**")
                    if tag_review_mode == RAPID_TAG_REVIEW_MODE_APPLICABLE:
                        rapid_evidence_line("Applicable tags", row.get("Rapid Review Tags", ""))
                        rapid_evidence_line("Best-fit tag", row.get("Rapid Review Best Tag", ""))
                    else:
                        rapid_evidence_line("Best-fit tag", row.get("Rapid Review Best Tag", ""))
                        rapid_evidence_line("Applicable tags", row.get("Rapid Review Tags", ""))

                machine_best = safe_display_text(row.get("Effective Rapid Best Tag", ""))
                final_best = safe_display_text(row.get("Final Rapid Best Tag", ""))
                machine_tags = safe_display_text(row.get("Effective Rapid Tags", ""))
                final_tags = safe_display_text(row.get("Final Rapid Tags", ""))
                changed = (machine_best and final_best and machine_best != final_best) or (
                    machine_tags and final_tags and machine_tags != final_tags
                )
                if changed:
                    st.write("**Final decision**")
                    if machine_best and final_best and machine_best != final_best:
                        st.caption(f"Best-fit: machine {machine_best} -> final {final_best}")
                    if machine_tags and final_tags and machine_tags != final_tags:
                        st.caption(f"Applicable tags: machine {machine_tags} -> final {final_tags}")

                rationale = safe_display_text(row.get("Rapid Review Tag Rationale", ""))
                if rationale:
                    st.write("**Reasoning**")
                    st.caption(rationale)

        if state and st.button("Clear human review", key=f"rapid_clear_{review_domain}_{current_group_id}", use_container_width=True):
            st.session_state.df_jev_sentiment_unique = _clear_rapid_human_review(
                st.session_state.df_jev_sentiment_unique,
                current_group_id,
                review_domain=review_domain,
                tag_review_mode=tag_review_mode if review_domain == "Tagging" else None,
            )
            st.rerun()

        st.divider()
        nav1, nav2 = st.columns(2)
        with nav1:
            if st.button("", disabled=(idx <= 0), use_container_width=True, icon=":material/skip_previous:", help="Previous story", key=f"rapid_prev_{review_domain}"):
                st.session_state[idx_key] = max(0, idx - 1)
                st.rerun()
        with nav2:
            if st.button("", disabled=(idx >= len(candidates) - 1), use_container_width=True, icon=":material/skip_next:", help="Next story", key=f"rapid_next_{review_domain}"):
                st.session_state[idx_key] = min(len(candidates) - 1, idx + 1)
                st.rerun()
        st.caption(f"Story {idx + 1} of {len(candidates)} in current view")

    st.stop()

unique_df = ensure_jev_sentiment_columns(
    st.session_state.get("df_jev_sentiment_unique", pd.DataFrame()),
    tag_definitions=tag_definitions,
)
unique_df = ensure_rapid_review_columns(unique_df)
st.session_state.df_jev_sentiment_unique = unique_df

current_analysis_fingerprint = build_analysis_context_fingerprint(analysis_payload)
stored_analysis_fingerprint = str(st.session_state.get("jev_analysis_context_fingerprint", "") or "")
if stored_analysis_fingerprint and stored_analysis_fingerprint != current_analysis_fingerprint:
    status = unique_df["Rapid Review Status"].fillna("").astype(str).str.strip()
    if status.ne("").any():
        st.session_state.df_jev_sentiment_unique = clear_rapid_review_columns(unique_df)
        st.session_state.pop("__last_rapid_review_batch_summary__", None)
        st.session_state.pop("rapid_review_target_batch", None)
        st.session_state.pop("rapid_review_target_source_count", None)
        st.warning("Analysis Context changed, so Rapid second-opinion results were cleared. Run second opinion again after confirming the current Rapid first-pass results are still appropriate.")
    st.session_state.jev_analysis_context_fingerprint = current_analysis_fingerprint
    unique_df = st.session_state.df_jev_sentiment_unique

eligible_count, sample_used, grouped_count, processed_count, error_count, remaining_count = metric_counts()

if st.session_state.jev_sentiment_section == "AI Second Opinion":
    st.caption(
        "Selectively rerun higher-value or higher-risk processed Rapid rows through one combined Luna review call."
    )

    review_candidates = build_rapid_review_candidates(
        st.session_state.df_jev_sentiment_unique,
        tag_definitions=tag_definitions,
    )
    review_status = st.session_state.df_jev_sentiment_unique["Rapid Review Status"].fillna("").astype(str).str.strip()
    reviewed_count = int(review_status.str.upper().eq("COMPLETED").sum())
    review_error_count = int(review_status.str.upper().eq("ERROR").sum())
    remaining_review_count = len(review_candidates)
    recommendation = resolve_rapid_review_recommendation(
        stored_target=int(st.session_state.get("rapid_review_target_batch", 0) or 0),
        stored_source_count=int(st.session_state.get("rapid_review_target_source_count", 0) or 0),
        current_source_count=processed_count,
        completed_count=reviewed_count,
        eligible_count=remaining_review_count,
    )
    st.session_state.rapid_review_target_batch = int(recommendation["target"])
    st.session_state.rapid_review_target_source_count = int(recommendation["source_count"])
    recommended_review_batch = int(recommendation["recommended_batch"])

    review_metrics = st.columns(4)
    review_metrics[0].metric("Rapid-labeled stories", f"{processed_count:,}")
    review_metrics[1].metric("Second-reviewed", f"{reviewed_count:,}")
    review_metrics[2].metric("Remaining available", f"{remaining_review_count:,}")
    review_metrics[3].metric("Review errors", f"{review_error_count:,}")

    last_review = st.session_state.get("__last_rapid_review_batch_summary__")
    if last_review:
        if last_review.get("errors"):
            st.warning(
                f"Last second-opinion batch completed with {last_review.get('successful', 0):,} successful result(s) "
                f"and {len(last_review.get('errors', [])):,} error(s) out of {last_review.get('done', 0):,} stories."
            )
            with st.expander("View last second-opinion batch errors", expanded=False):
                for error in last_review.get("errors", []):
                    st.write(error)
        else:
            st.success(
                f"Completed Rapid second opinion for {last_review.get('done', 0):,} grouped storie(s) in "
                f"{last_review.get('elapsed', 0.0):.1f}s."
            )

    st.caption(f"Model: {RAPID_REVIEW_MODEL}; workers: {RAPID_REVIEW_MAX_WORKERS}")
    openai_key = get_openai_api_key(st.secrets)
    if not openai_key:
        st.warning("Add `key` to Streamlit secrets or set `OPENAI_API_KEY` to run Rapid AI Second Opinion.")

    if remaining_review_count == 0:
        review_batch_size = 0
        st.info("No Rapid rows are currently eligible for second opinion.")
    else:
        default_manual_batch = min(10, remaining_review_count) if recommended_review_batch == 0 else recommended_review_batch
        stored_review_batch_size = int(st.session_state.get("rapid_review_batch_size", default_manual_batch or 1) or 1)
        batch_col, recommendation_col = st.columns([1.3, 1], gap="medium")
        with recommendation_col:
            st.metric("Recommended remaining batch", f"{recommended_review_batch:,}")
            st.caption("Remaining available may be larger; priority rules order the pool, not eligibility.")
        with batch_col:
            review_batch_size = st.number_input(
                "Second-opinion batch size",
                min_value=1,
                max_value=max(1, remaining_review_count),
                value=max(1, min(stored_review_batch_size, remaining_review_count)),
                step=1,
                key="rapid_review_batch_size",
            )
        if recommended_review_batch == 0:
            st.caption("The current recommended Rapid second-opinion coverage has already been reached. You can still run more manually if desired.")

    review_batch_df = review_candidates.head(int(review_batch_size or 0)).copy()
    run_review_clicked = st.button(
        "Run Rapid AI Second Opinion",
        type="primary",
        disabled=not bool(openai_key) or review_batch_df.empty,
        use_container_width=True,
    )

    if run_review_clicked:
        start = time.time()
        progress_bar = st.progress(0)
        progress_status = st.empty()

        def update_review_progress(done: int, total: int) -> None:
            if total <= 0:
                return
            progress_bar.progress(min(done / total, 1.0))
            progress_status.caption(f"Reviewed {done:,} of {total:,} stories · Model: {RAPID_REVIEW_MODEL}")

        update_review_progress(0, len(review_batch_df))
        with st.spinner(f"Running Rapid second opinion on {len(review_batch_df):,} grouped storie(s)..."):
            updated, summary = run_rapid_review_batch(
                st.session_state.df_jev_sentiment_unique,
                review_batch_df,
                analysis_payload,
                openai_key,
                model=RAPID_REVIEW_MODEL,
                tag_definitions=tag_definitions,
                tagging_mode=st.session_state.get("jev_tagging_mode", DEFAULT_JEV_TAGGING_MODE),
                max_workers=RAPID_REVIEW_MAX_WORKERS,
                limit=len(review_batch_df),
                progress_callback=update_review_progress,
            )
        progress_bar.empty()
        progress_status.empty()
        summary["elapsed"] = time.time() - start
        st.session_state.df_jev_sentiment_unique = updated
        st.session_state["__last_rapid_review_batch_summary__"] = summary
        apply_usage_to_session(summary.get("input_tokens", 0), summary.get("output_tokens", 0), RAPID_REVIEW_MODEL)
        st.rerun()

    st.write("**Second-opinion available pool / results**")
    review_display = st.session_state.df_jev_sentiment_unique.copy()
    if not review_candidates.empty:
        for _, candidate_row in review_candidates.iterrows():
            original_index = candidate_row.get("index", candidate_row.name)
            if original_index in review_display.index:
                review_display.loc[original_index, "Rapid Review Priority Tier"] = candidate_row.get("Rapid Review Priority Tier")
                review_display.loc[original_index, "Rapid Review Priority Reason"] = candidate_row.get("Rapid Review Priority Reason")

    visible_review = review_display[
        review_display["Jev Sentiment"].astype("string").fillna("").str.strip().ne("")
        | review_display["Rapid Review Status"].astype("string").fillna("").str.strip().ne("")
    ].copy()
    if visible_review.empty:
        st.info("Run Rapid Labeling first, then return here for selective second opinion.")
    else:
        first_pass_norm = visible_review.apply(normalize_first_pass_sentiment, axis=1)
        visible_review["First-pass Outcome"] = first_pass_norm.map(lambda item: item.get("outcome", ""))
        visible_review["First-pass Comparable Score"] = first_pass_norm.map(lambda item: item.get("score", pd.NA))
        visible_review["First-pass 3-Way From Score"] = first_pass_norm.map(lambda item: item.get("derived_3_way", ""))
        visible_review["First-pass 5-Way From Score"] = first_pass_norm.map(lambda item: item.get("derived_5_way", ""))
        review_columns = [
            "Group ID",
            "Headline",
            "Group Count",
            "Effective Reach",
            "Rapid Review Priority Tier",
            "Rapid Review Priority Reason",
            "First-pass Outcome",
            "First-pass Comparable Score",
            "Rapid Review Sentiment Outcome",
            "Rapid Review Sentiment Score",
            "Rapid Sentiment Score Distance",
            "Jev Best Tag",
            "Rapid Review Best Tag",
            "Rapid Sentiment 3-Way Agreement",
            "Rapid Sentiment 5-Way Agreement",
            "Rapid Tag Best-Fit Agreement",
            "Rapid Tag Set Agreement",
            "Rapid Review Status",
            "Rapid Review Error",
        ]
        existing_review_columns = [column for column in review_columns if column in visible_review.columns]
        st.dataframe(
            rename_jev_columns_to_rapid(visible_review[existing_review_columns]),
            use_container_width=True,
            hide_index=True,
            column_config=probability_column_config(visible_review[existing_review_columns]),
        )

    with st.expander("Rapid second-opinion available-pool details", expanded=False):
        if review_candidates.empty:
            st.info("No eligible candidates to inspect.")
        else:
            st.dataframe(
                rename_jev_columns_to_rapid(
                    review_candidates[
                        [
                            column
                            for column in [
                                "Group ID",
                                "Headline",
                                "Rapid Review Priority Tier",
                                "Rapid Review Priority Reason",
                                "Group Count",
                                "Effective Reach",
                                "Mentions",
                                "Impressions",
                                "Jev Sentiment",
                                "Jev 5-Way Sentiment",
                                "Jev Sentiment Score",
                                "Jev Best Tag",
                                "Jev Tags",
                            ]
                            if column in review_candidates.columns
                        ]
                    ]
                ),
                use_container_width=True,
                hide_index=True,
            )

    st.stop()

st.caption("Each story is labeled once with all structured sentiment and configured tag questions in the same request.")

metrics = st.columns(6)
metrics[0].metric("Eligible mentions", f"{eligible_count:,}")
metrics[1].metric("Sample used", f"{sample_used:,}")
metrics[2].metric("Grouped stories", f"{grouped_count:,}")
metrics[3].metric("Processed", f"{processed_count:,}")
metrics[4].metric("Errors", f"{error_count:,}")
metrics[5].metric("Remaining", f"{remaining_count:,}")

last = st.session_state.get("__last_jev_sentiment_batch_summary__")
if last:
    last_method = last.get("method", "Rapid Labeling")
    if last.get("errors"):
        st.warning(
            f"Last {last_method} batch completed with {last.get('successful', 0):,} successful result(s) and "
            f"{len(last.get('errors', [])):,} error(s) out of {last.get('done', 0):,} stories."
        )
        with st.expander("View last Rapid Labeling batch errors", expanded=False):
            for error in last.get("errors", []):
                st.write(error)
    else:
        st.success(
            f"Completed {last_method} sentiment for {last.get('done', 0):,} grouped storie(s) in "
            f"{last.get('elapsed', 0.0):.1f}s."
        )

config_cols = st.columns(3)
config_cols[0].caption(f"Dataset mode: {format_sample_mode(st.session_state.get('jev_sentiment_sample_mode', 'representative'))}")
tag_caption = f"{len(build_jev_tag_question_map(tag_definitions)):,} tag(s)" if tagging_enabled else "sentiment only"
config_cols[1].caption(f"Rapid Labeling: sentiment + {tag_caption}; workers: {DEFAULT_JEV_MAX_WORKERS}")
config_cols[2].caption(f"Model: {display_model_name(DEFAULT_JEV_MODEL)}")

if tagging_enabled:
    mode_index = (
        list(JEV_TAGGING_MODES).index(current_tagging_mode)
        if current_tagging_mode in JEV_TAGGING_MODES
        else 0
    )
    selected_tagging_mode = st.radio(
        "Active Rapid Tag assignment mode",
        list(JEV_TAGGING_MODES),
        index=mode_index,
        horizontal=True,
        help="Changing this recomputes Rapid Tags from already-stored best-fit and independent tag outputs.",
    )
    if selected_tagging_mode != current_tagging_mode:
        st.session_state.jev_tagging_mode = selected_tagging_mode
        st.session_state.df_jev_sentiment_unique = recompute_jev_official_tag_assignments(
            st.session_state.df_jev_sentiment_unique,
            tag_definitions=tag_definitions,
            tagging_mode=selected_tagging_mode,
        )
        st.rerun()

remaining_df = get_remaining_jev_sentiment_rows(
    st.session_state.df_jev_sentiment_unique,
    tag_definitions=tag_definitions,
)
if remaining_count == 0:
    batch_size = 0
    st.info("No grouped stories remain for Rapid Labeling. Rows with errors are skipped so they do not block later batches.")
else:
    stored_batch_size = int(st.session_state.get("jev_sentiment_batch_size", 200) or 200)
    if stored_batch_size == 10:
        stored_batch_size = 200
    default_batch_size = min(stored_batch_size, remaining_count)
    batch_size = st.number_input(
        "Batch size",
        min_value=1,
        max_value=max(1, remaining_count),
        value=max(1, default_batch_size),
        step=1,
        key="jev_sentiment_batch_size",
    )

batch_df = remaining_df.iloc[: int(batch_size or 0)].copy()
api_key = get_openrouter_api_key(st.secrets)
can_run = bool(api_key) and not batch_df.empty
if not api_key:
    st.warning("Add `openrouter_key` to Streamlit secrets or set `OPENROUTER_API_KEY` to run Rapid Labeling.")

run_col, reset_col = st.columns([3, 1])
with run_col:
    run_clicked = st.button(
        "Run Next Rapid Batch",
        type="primary",
        disabled=not can_run,
        use_container_width=True,
    )
with reset_col:
    if st.button("Reset Rapid Results", use_container_width=True):
        reset_jev_results()
        st.rerun()

if run_clicked:
    start = time.time()
    progress_bar = st.progress(0)
    progress_status = st.empty()

    def update_batch_progress(done: int, total: int) -> None:
        if total <= 0:
            return
        progress_bar.progress(min(done / total, 1.0))
        progress_status.caption(f"Processed {done:,} of {total:,} stories · Model: {display_model_name(DEFAULT_JEV_MODEL)}")

    update_batch_progress(0, len(batch_df))
    with st.spinner(f"Running Rapid Labeling on {len(batch_df):,} grouped storie(s)..."):
        updated, summary = run_jev_sentiment_batch(
            st.session_state.df_jev_sentiment_unique,
            batch_df,
            analysis_payload,
            api_key,
            model=DEFAULT_JEV_MODEL,
            max_workers=DEFAULT_JEV_MAX_WORKERS,
            progress_callback=update_batch_progress,
            tag_definitions=tag_definitions,
            tagging_mode=st.session_state.get("jev_tagging_mode", DEFAULT_JEV_TAGGING_MODE),
        )
    progress_bar.empty()
    progress_status.empty()
    summary["elapsed"] = time.time() - start
    summary["method"] = "Rapid Labeling"
    st.session_state.df_jev_sentiment_unique = updated
    st.session_state["__last_jev_sentiment_batch_summary__"] = summary
    st.rerun()

with st.expander("Rapid request/state/question preview", expanded=False):
    preview_source = batch_df if not batch_df.empty else unique_df.reset_index(drop=False).head(1)
    if preview_source.empty:
        st.info("No story available for preview.")
    else:
        preview_payload = build_jev_combined_sentiment_payload(
            preview_source.iloc[0],
            analysis_payload,
            model=DEFAULT_JEV_MODEL,
            tag_definitions=tag_definitions,
        )
        st.json(preview_payload)

has_rapid_first_pass_results = bool(
    rapid_processed_mask(st.session_state.df_jev_sentiment_unique).any()
)
if not has_rapid_first_pass_results:
    st.stop()

st.divider()
st.write("**Processed Rapid Labeling results**")
st.caption("Each completed row contains all structured sentiment outputs from one shared labeling request.")
display_df = build_jev_results_display(
    st.session_state.df_jev_sentiment_unique,
    tag_definitions=tag_definitions,
)
if display_df.empty:
    st.info("No Rapid Labeling rows to display yet.")
else:
    compact = display_df[
        display_df["Jev Sentiment"].astype("string").fillna("").str.strip().ne("")
        | display_df["Jev Error"].astype("string").fillna("").str.strip().ne("")
    ].copy()
    if compact.empty:
        compact = display_df.head(50).copy()
    rapid_compact = rename_jev_columns_to_rapid(compact)
    st.dataframe(
        percent_display(rapid_compact),
        use_container_width=True,
        hide_index=True,
        column_config=probability_column_config(rapid_compact),
    )

    with st.expander("Machine-effective Rapid labels and provenance", expanded=False):
        effective_sentiment = build_effective_rapid_sentiment_frame(st.session_state.df_jev_sentiment_unique)
        effective_tags = build_effective_rapid_tag_frame(
            st.session_state.df_jev_sentiment_unique,
            tag_definitions=tag_definitions,
        )
        provenance = summarize_rapid_label_provenance(
            st.session_state.df_jev_sentiment_unique,
            tag_definitions=tag_definitions,
        )
        pcols = st.columns(5)
        pcols[0].metric("Rapid-labeled", f"{provenance.get('rapid_labeled_stories', 0):,}")
        pcols[1].metric("Effective sentiment", f"{provenance.get('effective_sentiment_available', 0):,}")
        pcols[2].metric("Effective tags", f"{provenance.get('effective_tags_available', 0):,}")
        pcols[3].metric("AI disagreements", f"{provenance.get('cross_model_disagreement', 0):,}")
        pcols[4].metric("Internal conflicts", f"{provenance.get('first_pass_internal_conflict', 0):,}")

        effective_display = pd.concat(
            [
                st.session_state.df_jev_sentiment_unique[
                    [column for column in ["Group ID", "Headline", "Group Count"] if column in st.session_state.df_jev_sentiment_unique.columns]
                ],
                effective_sentiment,
                effective_tags,
            ],
            axis=1,
        )
        effective_display = effective_display[
            effective_display["Effective Rapid Sentiment 3-Way"].fillna("").astype(str).str.strip().ne("")
            | effective_display["Effective Rapid Best Tag"].fillna("").astype(str).str.strip().ne("")
            | effective_display["Rapid Sentiment Resolution Status"].fillna("").astype(str).str.strip().ne("")
        ].copy()
        if effective_display.empty:
            st.info("No effective Rapid labels are available yet.")
        else:
            st.dataframe(
                effective_display.head(200),
                use_container_width=True,
                hide_index=True,
                column_config=probability_column_config(effective_display),
            )

    with st.expander("Current label distributions", expanded=False):
        distribution_source = ensure_jev_sentiment_columns(
            st.session_state.df_jev_sentiment_unique,
            tag_definitions=tag_definitions,
        )
        sentiment_colors = get_sentiment_color_mapping()
        include_not_relevant = st.toggle(
            "Include Not Relevant in percentages",
            value=False,
            key="jev_distribution_include_not_relevant",
        )

        st.write("**Sentiment**")
        sentiment_col, sentiment_5_col = st.columns(2)
        three_way_order = [
            item for item in JEV_3WAY_SENTIMENT_ORDER if include_not_relevant or item != "NOT RELEVANT"
        ]
        five_way_order = [
            item for item in JEV_5WAY_SENTIMENT_ORDER if include_not_relevant or item != "NOT RELEVANT"
        ]
        three_way_dist = filter_sentiment_distribution(
            build_jev_sentiment_distribution(
                distribution_source,
                column="Jev Sentiment",
                order=JEV_3WAY_SENTIMENT_ORDER,
            ),
            JEV_3WAY_SENTIMENT_ORDER,
            include_not_relevant,
        )
        five_way_dist = filter_sentiment_distribution(
            build_jev_sentiment_distribution(
                distribution_source,
                column="Jev 5-Way Sentiment",
                order=JEV_5WAY_SENTIMENT_ORDER,
            ),
            JEV_5WAY_SENTIMENT_ORDER,
            include_not_relevant,
        )

        with sentiment_col:
            st.caption("3-way sentiment")
            st.altair_chart(
                build_donut_chart(
                    three_way_dist,
                    label_column="Sentiment",
                    count_column="Count",
                    color_domain=three_way_order,
                    color_range=[sentiment_colors[item] for item in three_way_order],
                ),
                use_container_width=True,
            )
            st.dataframe(
                format_distribution_table(three_way_dist, "Sentiment"),
                use_container_width=True,
                hide_index=True,
            )

        with sentiment_5_col:
            st.caption("5-way sentiment")
            st.altair_chart(
                build_donut_chart(
                    five_way_dist,
                    label_column="Sentiment",
                    count_column="Count",
                    color_domain=five_way_order,
                    color_range=[sentiment_colors[item] for item in five_way_order],
                ),
                use_container_width=True,
            )
            st.dataframe(
                format_distribution_table(five_way_dist, "Sentiment"),
                use_container_width=True,
                hide_index=True,
            )

        st.write("**Tagging**")
        tag_col, applicable_col = st.columns(2)
        with tag_col:
            st.caption("Best-fit tags: one mutually exclusive tag per processed story.")
            include_other_in_best_fit = st.toggle(
                "Include Other in percentages",
                value=False,
                key="jev_distribution_best_fit_include_other",
            )
            best_tag_dist = filter_exclusive_tag_distribution(
                build_jev_best_tag_distribution(distribution_source),
                include_other=include_other_in_best_fit,
            )
            if best_tag_dist.empty:
                st.info("No Rapid best-fit tag results yet.")
            else:
                tag_color_domain, tag_color_range = build_jev_tag_color_scale(
                    [item["tag"] for item in build_jev_tag_question_map(tag_definitions)]
                    + best_tag_dist["Tag"].astype(str).tolist(),
                    include_other=include_other_in_best_fit,
                )
                st.altair_chart(
                    build_donut_chart(
                        best_tag_dist,
                        label_column="Tag",
                        count_column="Count",
                        color_domain=tag_color_domain,
                        color_range=tag_color_range,
                        show_legend=True,
                    ),
                    use_container_width=True,
                )
                st.dataframe(
                    format_distribution_table(best_tag_dist, "Tag"),
                    use_container_width=True,
                    hide_index=True,
                )

        with applicable_col:
            st.caption(
                "All applicable tags: a story may contribute to multiple categories. "
                "% uses processed grouped stories as the denominator and may sum above 100%."
            )
            include_other_in_applicable = st.toggle(
                "Include Other",
                value=False,
                key="jev_distribution_applicable_include_other",
            )
            applicable_dist = filter_all_applicable_tag_distribution(
                build_jev_all_applicable_tag_distribution(
                    distribution_source,
                    tag_definitions=tag_definitions,
                ),
                include_other=include_other_in_applicable,
            )
            if applicable_dist.empty:
                st.info("No Rapid all-applicable tag results yet.")
            else:
                st.altair_chart(build_tag_bar_chart(applicable_dist), use_container_width=True)
                applicable_table = format_distribution_table(
                    applicable_dist,
                    "Tag",
                    share_column="Grouped Story Share",
                ).rename(columns={"Share": "% processed grouped stories"})
                st.dataframe(applicable_table, use_container_width=True, hide_index=True)
