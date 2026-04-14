# 8-Tagging.py
from __future__ import annotations

import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

import altair as alt
import pandas as pd
import streamlit as st

from processing.tagging_config import (
    init_tagging_config_state,
    calculate_representative_sample_size,
    prepare_tagging_datasets,
    reset_tagging_config_state,
    get_tagging_source_rows,
    get_available_coverage_flags,
    apply_coverage_flag_exclusions,
    DEFAULT_MAX_FULL_ROWS,
)
from processing.ai_tagging import (
    init_ai_tagging_state,
    build_default_tags_text,
    parse_tag_definitions,
    get_remaining_tagging_rows,
    analyze_story_worker,
    apply_tagging_result_to_unique_df,
    cascade_tags_to_grouped_rows,
    reset_ai_tagging_results,
    DEFAULT_TAGGING_BATCH_SIZE,
    DEFAULT_TAGGING_MAX_WORKERS,
)
from utils.api_meter import (
    apply_usage_to_session,
    estimate_cost_usd,
    get_api_cost_usd,
    init_api_meter,
)

warnings.filterwarnings("ignore")

DEFAULT_TAGGING_MODEL = "gpt-5.4-nano"

st.title("AI Tagging")
st.caption("Prepare a grouped coverage sample, configure tag definitions, and apply AI tags back onto the dataset.")
st.session_state.setdefault("tagging_section", "Setup")

if not st.session_state.get("standard_step", False):
    st.error("Please complete Basic Cleaning before trying this step.")
    st.stop()

init_tagging_config_state(st.session_state)
init_ai_tagging_state(st.session_state)
init_api_meter()

client_name = st.session_state.get("client_name", "")
if not st.session_state.tags_text.strip():
    st.session_state.tags_text = build_default_tags_text(client_name)

_last = st.session_state.get("__last_tagging_batch_summary__")
if _last:
    st.success(f"Completed AI tagging for {_last['done']} grouped storie(s) in {_last['elapsed']:.1f}s.")
    if _last["errors"]:
        with st.expander(f"Completed with {len(_last['errors'])} error(s)", expanded=False):
            for err in _last["errors"]:
                st.write(err)

source_rows = get_tagging_source_rows(st.session_state.df_traditional)
population_size = len(source_rows)


def _format_sample_mode(mode: str) -> str:
    return {
        "reuse_other_sample": "Reused sentiment sample",
        "full": "Full eligible dataset",
        "representative": "Representative sample",
        "custom": "Custom sample",
    }.get(mode, str(mode))


def build_tag_distribution(df_tagging_unique: pd.DataFrame) -> pd.DataFrame:
    tag_col = "AI Tag" if "AI Tag" in df_tagging_unique.columns else ("AI Tags" if "AI Tags" in df_tagging_unique.columns else None)
    if tag_col is None:
        return pd.DataFrame(columns=["Tag", "Count", "Share"])

    tags = (
        df_tagging_unique[tag_col]
        .fillna("")
        .astype(str)
        .str.split(",")
        .explode()
        .astype(str)
        .str.strip()
    )
    tags = tags[tags != ""]

    if tags.empty:
        return pd.DataFrame(columns=["Tag", "Count", "Share"])

    out = tags.value_counts().rename_axis("Tag").reset_index(name="Count")
    out["Share"] = out["Count"] / int(out["Count"].sum())
    return out


def build_tag_distribution_chart(tag_dist: pd.DataFrame) -> alt.Chart:
    working = tag_dist.copy()
    working["Share Label"] = (working["Share"] * 100).map(lambda x: f"{x:.1f}%")

    base = alt.Chart(working).encode(
        y=alt.Y(
            "Tag:N",
            sort="-x",
            axis=alt.Axis(title=None, labelLimit=240, labelPadding=10),
        )
    )

    bars = base.mark_bar(cornerRadiusEnd=3, color="#636E95").encode(
        x=alt.X("Count:Q", axis=alt.Axis(title=None, grid=True, tickMinStep=1)),
        tooltip=[
            "Tag",
            alt.Tooltip("Count:Q", format=","),
            alt.Tooltip("Share:Q", format=".1%", title="Share"),
        ],
    )

    text = base.mark_text(
        align="left",
        baseline="middle",
        dx=6,
        color="#F8FAFC",
        fontWeight=600,
    ).encode(
        x="Count:Q",
        text="Share Label:N",
    )

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


st.markdown(
    """
    <style>
    .tagging-step-note {
        margin: 0.15rem 0 1rem 0;
        color: rgba(250, 250, 250, 0.72);
        font-size: 0.95rem;
    }
    div[data-testid="stButton"] button[kind="secondary"] {
        min-height: 2.8rem;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

step1, step2, step3 = st.columns(3, gap="small")
with step1:
    if st.button(
        "1. Setup",
        key="tagging_nav_setup",
        use_container_width=True,
        type="primary" if st.session_state.tagging_section == "Setup" else "secondary",
    ):
        st.session_state.tagging_section = "Setup"
        st.rerun()
with step2:
    if st.button(
        "2. Run",
        key="tagging_nav_run",
        use_container_width=True,
        type="primary" if st.session_state.tagging_section == "Run" else "secondary",
    ):
        st.session_state.tagging_section = "Run"
        st.rerun()
with step3:
    if st.button(
        "3. Review",
        key="tagging_nav_review",
        use_container_width=True,
        type="primary" if st.session_state.tagging_section == "Review" else "secondary",
    ):
        st.session_state.tagging_section = "Review"
        st.rerun()

st.markdown(
    '<div class="tagging-step-note">Work left to right: prepare the tagging dataset, run AI tagging, then review the outputs and distributions.</div>',
    unsafe_allow_html=True,
)


# =========================
# STEP 1: DATASET PREP + CONFIG
# =========================
if st.session_state.tagging_section == "Setup":
    st.subheader("Step 1: Prepare Tagging Dataset & Configuration")
    st.caption("Sampling happens at the row level first, then the sampled rows are regrouped into unique stories.")

    reusable_sentiment_sample = st.session_state.get("df_sentiment_rows", None)
    has_reusable = isinstance(reusable_sentiment_sample, pd.DataFrame) and not reusable_sentiment_sample.empty

    options = [
        "Use full eligible dataset",
        "Use representative sample",
        "Set custom sample size",
    ]
    if has_reusable:
        options.insert(0, "Reuse sentiment sample")

    default_index = 1 if population_size > DEFAULT_MAX_FULL_ROWS else 0
    if has_reusable:
        default_index = 0

    mode_label = st.radio(
        "Tagging dataset mode",
        options=options,
        index=default_index,
    )

    if mode_label == "Reuse sentiment sample":
        sample_mode = "reuse_other_sample"
    elif mode_label == "Use full eligible dataset":
        sample_mode = "full"
    elif mode_label == "Use representative sample":
        sample_mode = "representative"
    else:
        sample_mode = "custom"

    excluded_flags: list[str] = []
    working_source_rows = source_rows

    if sample_mode != "reuse_other_sample":
        available_flags, default_excluded_flags = get_available_coverage_flags(source_rows)
        stored_excluded_flags = st.session_state.get("tagging_excluded_flags", default_excluded_flags)
        fallback_defaults = [f for f in default_excluded_flags if f in available_flags]
        preselected_flags = [f for f in stored_excluded_flags if f in available_flags] or fallback_defaults

        excluded_flags = st.multiselect(
            "Exclude coverage flags",
            options=available_flags,
            default=preselected_flags,
            help="Exclude selected flagged coverage from tagging sampling.",
        )

        working_source_rows = apply_coverage_flag_exclusions(source_rows, excluded_flags)

    population_size = len(working_source_rows)
    recommended_sample = calculate_representative_sample_size(population_size) if population_size > 0 else 0

    custom_sample_size = None
    if sample_mode == "custom":
        custom_sample_size = st.number_input(
            "Custom sample size",
            min_value=1,
            value=min(400, max(1, population_size)),
            step=1,
        )

    full_override = False
    if sample_mode == "full" and population_size > DEFAULT_MAX_FULL_ROWS:
        st.warning(
            f"Full tagging is limited to {DEFAULT_MAX_FULL_ROWS:,} row-level mentions by default for cost and stability reasons."
        )
        full_override = st.checkbox(
            f"I understand the risk and want to allow full tagging over {DEFAULT_MAX_FULL_ROWS:,} mentions",
            value=False,
        )

    preview_rows = 0
    if sample_mode == "reuse_other_sample":
        preview_rows = len(reusable_sentiment_sample) if has_reusable else 0
    elif sample_mode == "full":
        preview_rows = population_size if (population_size <= DEFAULT_MAX_FULL_ROWS or full_override) else DEFAULT_MAX_FULL_ROWS
    elif sample_mode == "representative":
        preview_rows = recommended_sample
    else:
        preview_rows = int(custom_sample_size or 0)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Eligible mentions", f"{population_size:,}")
    with col2:
        st.metric("Estimated mentions used", f"{preview_rows:,}")

    if sample_mode == "representative":
        st.caption(f"Representative sample size estimate: {recommended_sample:,}")

    if sample_mode == "reuse_other_sample" and has_reusable:
        st.caption(f"Using existing sentiment sample: {len(reusable_sentiment_sample):,} rows")

    st.divider()
    st.write("**Tagging configuration**")

    tags_text = st.text_area(
        "Define tags and criteria",
        height=200,
        value=st.session_state.tags_text,
        help="One per line, in the format: TagName: Criteria",
    )

    tagging_mode = st.radio(
        "Tagging mode",
        ["Single best tag", "Multiple applicable tags"],
        index=0 if st.session_state.get("tagging_mode", "Single best tag") == "Single best tag" else 1,
    )

    prep_clicked = st.button("Prepare Tagging Dataset", type="primary")

    if prep_clicked:
        if sample_mode == "custom":
            if custom_sample_size is None:
                st.error("Please enter a custom sample size.")
                st.stop()
            if int(custom_sample_size) > population_size:
                st.error(f"Custom sample size cannot exceed the eligible dataset size of {population_size:,}.")
                st.stop()

        tag_definitions = parse_tag_definitions(tags_text)
        if len(tag_definitions) == 0:
            st.error("Please define at least one valid tag and its criteria before preparing the dataset.")
            st.stop()

        start = time.time()

        reused_rows = reusable_sentiment_sample if sample_mode == "reuse_other_sample" else None

        results = prepare_tagging_datasets(
            df_traditional=st.session_state.df_traditional,
            sample_mode=sample_mode,
            excluded_flags=excluded_flags if sample_mode != "reuse_other_sample" else [],
            custom_sample_size=custom_sample_size,
            max_full_rows=DEFAULT_MAX_FULL_ROWS,
            full_override=full_override,
            reused_rows=reused_rows,
        )

        st.session_state.df_tagging_rows = results["df_tagging_rows"]
        st.session_state.df_tagging_grouped_rows = results["df_tagging_grouped_rows"]
        st.session_state.df_tagging_unique = results["df_tagging_unique"]
        st.session_state.tagging_sample_mode = sample_mode
        st.session_state.tagging_sample_size = results["sample_size_used"]
        st.session_state.tagging_full_override = full_override
        st.session_state.tagging_excluded_flags = excluded_flags if sample_mode != "reuse_other_sample" else []
        st.session_state.tagging_config_step = True
        st.session_state.tagging_elapsed_time = time.time() - start

        # Lock config here
        st.session_state.tags_text = tags_text
        st.session_state.tag_definitions = tag_definitions
        st.session_state.tagging_mode = tagging_mode
        st.session_state.tagging_model = DEFAULT_TAGGING_MODEL
        st.session_state.tagging_section = "Run"

        st.rerun()

    if st.session_state.tagging_config_step:
        st.info("A tagging dataset is already prepared. You can adjust the setup and prepare again, or move to Run.")
    st.stop()

# =========================
# STEP 2: RUN AI TAGGING
# =========================
if not st.session_state.tagging_config_step:
    if st.session_state.tagging_section in {"Run", "Review"}:
        st.info("Prepare the tagging dataset in Setup before running or reviewing AI tagging.")
    st.stop()

remaining_df = get_remaining_tagging_rows(st.session_state.df_tagging_unique)
remaining_count = len(remaining_df)

if st.session_state.tagging_section == "Run":
    st.subheader("Step 2: Run AI Tagging")
    processed_count = len(st.session_state.df_tagging_unique) - remaining_count

    top_col1, top_col2, top_col3, top_col4, top_col5 = st.columns(5)
    with top_col1:
        st.metric("Eligible mentions", f"{population_size:,}")
    with top_col2:
        st.metric("Sample used", f"{len(st.session_state.df_tagging_rows):,}")
    with top_col3:
        st.metric("Grouped stories", f"{len(st.session_state.df_tagging_unique):,}")
    with top_col4:
        st.metric("Processed stories", f"{processed_count:,}")
    with top_col5:
        st.metric("Remaining stories", f"{remaining_count:,}")

    reset_col1, reset_col2 = st.columns([4, 1])
    with reset_col2:
        if st.button("Reset Tagging Dataset"):
            reset_tagging_config_state(st.session_state)
            st.session_state.tagging_section = "Setup"
            st.rerun()

    config_col1, config_col2 = st.columns(2)
    with config_col1:
        st.caption(f"Dataset mode: {_format_sample_mode(st.session_state.get('tagging_sample_mode', 'representative'))}")
    with config_col2:
        st.caption(f"Tagging mode: {st.session_state.get('tagging_mode', 'Single best tag')}")

    if remaining_count == 0:
        row_limit = 0
        st.info("No grouped stories remain to tag.")
    else:
        default_batch_value = st.session_state.get("tagging_batch_size", DEFAULT_TAGGING_BATCH_SIZE)
        default_batch_value = min(default_batch_value, remaining_count)

        row_limit = st.number_input(
            "Batch size (0 for all remaining rows)",
            min_value=0,
            max_value=remaining_count,
            value=default_batch_value,
            step=1,
            key="tagging_batch_size",
        )

    if row_limit > 0:
        batch_df = remaining_df.iloc[:row_limit].copy()
    else:
        batch_df = remaining_df.copy()

    st.write(f"Selected grouped stories for analysis: {len(batch_df):,}")

    apply_clicked = st.button("Apply Tags", type="primary", disabled=(len(batch_df) == 0))
    reset_results_clicked = st.button("Reset Processed Rows")

    if reset_results_clicked:
        unique, grouped = reset_ai_tagging_results(
            st.session_state.df_tagging_unique,
            st.session_state.df_tagging_grouped_rows,
        )
        st.session_state.df_tagging_unique = unique
        st.session_state.df_tagging_grouped_rows = grouped
        st.session_state.df_tagging_rows = st.session_state.df_tagging_grouped_rows.copy()
        st.success("Reset AI tagging results.")
        st.rerun()

    if apply_clicked:
        tag_definitions = st.session_state.get("tag_definitions", [])
        tagging_mode = st.session_state.get("tagging_mode", "Single best tag")
        model = st.session_state.get("tagging_model", DEFAULT_TAGGING_MODEL)

        if len(tag_definitions) == 0:
            st.error("No tag definitions are saved. Please reset and prepare the tagging dataset again.")
            st.stop()

        progress_bar = st.progress(0)
        total_in = 0
        total_out = 0
        errors = []
        completed = 0
        total = len(batch_df)
        start_time = time.time()

        rows_for_workers = [(idx, row.to_dict()) for idx, row in batch_df.iterrows()]

        with ThreadPoolExecutor(max_workers=DEFAULT_TAGGING_MAX_WORKERS) as executor:
            future_map = {
                executor.submit(
                    analyze_story_worker,
                    row_tuple,
                    tag_definitions,
                    tagging_mode,
                    model,
                    st.secrets["key"],
                ): row_tuple[0]
                for row_tuple in rows_for_workers
            }

            for future in as_completed(future_map):
                completed += 1

                try:
                    idx, result, error_message, in_tok, out_tok = future.result()
                    total_in += int(in_tok or 0)
                    total_out += int(out_tok or 0)

                    original_index = batch_df.loc[idx, "index"]

                    if error_message:
                        errors.append(f"Story {original_index + 1}: {error_message}")
                    else:
                        st.session_state.df_tagging_unique = apply_tagging_result_to_unique_df(
                            st.session_state.df_tagging_unique,
                            original_index=original_index,
                            result=result,
                            tagging_mode=tagging_mode,
                            tag_definitions=tag_definitions,
                        )

                except Exception as e:
                    errors.append(str(e))

                progress_bar.progress(completed / max(1, total))

        st.session_state.df_tagging_grouped_rows = cascade_tags_to_grouped_rows(
            st.session_state.df_tagging_grouped_rows,
            st.session_state.df_tagging_unique,
        )

        st.session_state.df_tagging_rows = st.session_state.df_tagging_grouped_rows.copy()

        apply_usage_to_session(total_in, total_out, model)

        batch_cost = estimate_cost_usd(total_in, total_out, model)
        session_cost = get_api_cost_usd()

        st.session_state["__last_tagging_batch_summary__"] = {
            "done": total,
            "elapsed": time.time() - start_time,
            "in_tok": total_in,
            "out_tok": total_out,
            "batch_cost": batch_cost,
            "session_cost": session_cost,
            "errors": errors,
        }
        st.session_state.tagging_section = "Run"
        st.rerun()

    st.stop()

# =========================
# STEP 3: REVIEW OUTPUTS
# =========================
st.subheader("Step 3: Review Tagging Outputs")
st.caption("Inspect the grouped tagging dataset, current tag distribution, and the saved tag definitions.")

review_col1, review_col2, review_col3 = st.columns(3)
with review_col1:
    st.metric("Grouped stories", f"{len(st.session_state.df_tagging_unique):,}")
with review_col2:
    tagged_df = get_remaining_tagging_rows(st.session_state.df_tagging_unique)
    processed_count = len(st.session_state.df_tagging_unique) - len(tagged_df)
    st.metric("Processed stories", f"{processed_count:,}")
with review_col3:
    st.metric("Remaining stories", f"{len(tagged_df):,}")

review_tab1, review_tab2, review_tab3 = st.tabs(["Dataset", "Distribution", "Tag Guide"])

with review_tab1:
    st.dataframe(
        st.session_state.df_tagging_unique.head(200),
        use_container_width=True,
        hide_index=True,
    )

with review_tab2:
    tag_dist = build_tag_distribution(st.session_state.df_tagging_unique)
    if tag_dist.empty:
        st.caption("No AI tags have been assigned yet.")
    else:
        include_other = st.toggle(
            "Include Other in percentages",
            value=True,
            key="tagging_distribution_include_other",
        )
        filtered_dist = tag_dist.copy()
        if not include_other:
            filtered_dist = filtered_dist[
                filtered_dist["Tag"].fillna("").astype(str).str.strip().str.lower() != "other"
            ].copy()
            total = int(filtered_dist["Count"].sum())
            filtered_dist["Share"] = filtered_dist["Count"] / total if total > 0 else 0.0

        dist_col1, dist_col2 = st.columns([1.35, 1], gap="large")
        with dist_col1:
            st.altair_chart(build_tag_distribution_chart(filtered_dist), use_container_width=True)
        with dist_col2:
            tag_table = filtered_dist.copy()
            tag_table["Share"] = (tag_table["Share"] * 100).map(lambda x: f"{x:.1f}%")
            st.dataframe(tag_table, use_container_width=True, hide_index=True)

with review_tab3:
    st.code(st.session_state.get("tags_text", ""))
