from __future__ import annotations

import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import streamlit as st

from processing.tagging_config import (
    init_tagging_config_state,
    calculate_representative_sample_size,
    prepare_tagging_datasets,
    reset_tagging_config_state,
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

source_rows = st.session_state.df_traditional.copy()
population_size = len(source_rows)
recommended_sample = calculate_representative_sample_size(population_size) if population_size > 0 else 0


def _format_sample_mode(mode: str) -> str:
    return {
        "reuse_other_sample": "Reused sentiment sample",
        "full": "Full eligible dataset",
        "representative": "Representative sample",
        "custom": "Custom sample",
    }.get(mode, str(mode))


# =========================
# STEP 1: DATASET PREP + CONFIG
# =========================
if not st.session_state.tagging_config_step:
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
        st.session_state.tagging_config_step = True
        st.session_state.tagging_elapsed_time = time.time() - start

        # Lock config here
        st.session_state.tags_text = tags_text
        st.session_state.tag_definitions = tag_definitions
        st.session_state.tagging_mode = tagging_mode
        st.session_state.tagging_model = DEFAULT_TAGGING_MODEL

        st.rerun()

    st.stop()

# =========================
# STEP 2: RUN AI TAGGING
# =========================
st.subheader("Step 2: Run AI Tagging")

top_col1, top_col2, top_col3, top_col4 = st.columns(4)
with top_col1:
    st.metric("Eligible mentions", f"{population_size:,}")
with top_col2:
    st.metric("Sample used", f"{len(st.session_state.df_tagging_rows):,}")
with top_col3:
    st.metric("Grouped stories", f"{len(st.session_state.df_tagging_unique):,}")
with top_col4:
    if st.button("Reset Tagging Dataset"):
        reset_tagging_config_state(st.session_state)
        st.rerun()

config_col1, config_col2 = st.columns(2)
with config_col1:
    st.caption(f"Dataset mode: {_format_sample_mode(st.session_state.get('tagging_sample_mode', 'representative'))}")
with config_col2:
    st.caption(f"Tagging mode: {st.session_state.get('tagging_mode', 'Single best tag')}")

with st.expander("Tag definitions", expanded=False):
    st.code(st.session_state.get("tags_text", ""))

remaining_df = get_remaining_tagging_rows(st.session_state.df_tagging_unique)
remaining_count = len(remaining_df)

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

    # keep row-level sample export dataframe in sync too
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

    st.rerun()

with st.expander("Grouped tagging dataset preview", expanded=False):
    st.dataframe(
        st.session_state.df_tagging_unique.head(200),
        use_container_width=True,
        hide_index=True,
    )