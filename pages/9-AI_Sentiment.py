# 9-AI_Sentiment.py
# 9-AI_Sentiment

from __future__ import annotations
import pandas as pd
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

import streamlit as st
from streamlit_tags import st_tags

from processing.sentiment_config import (
    init_sentiment_config_state,
    calculate_representative_sample_size,
    prepare_sentiment_datasets,
    build_sentiment_configuration,
    reset_sentiment_config_state,
    get_sentiment_source_rows,
    get_available_coverage_flags,
    apply_coverage_flag_exclusions,
    DEFAULT_MAX_FULL_ROWS,
)
from processing.ai_sentiment import (
    init_ai_sentiment_state,
    ensure_ai_sentiment_columns,
    get_remaining_sentiment_rows,
    analyze_sentiment_worker,
    apply_sentiment_result_to_unique_df,
    cascade_sentiment_to_grouped_rows,
    reset_ai_sentiment_results,
    DEFAULT_SENTIMENT_BATCH_SIZE,
    DEFAULT_SENTIMENT_MAX_WORKERS,
    DEFAULT_SENTIMENT_MODEL,
)
from utils.api_meter import (
    init_api_meter,
    apply_usage_to_session,
    estimate_cost_usd,
    get_api_cost_usd,
)

warnings.filterwarnings("ignore")

st.title("AI Sentiment")

if not st.session_state.get("standard_step", False):
    st.error("Please complete Basic Cleaning before trying this step.")
    st.stop()

init_sentiment_config_state(st.session_state)
init_ai_sentiment_state(st.session_state)
init_api_meter()

_last = st.session_state.get("__last_sentiment_batch_summary__")
if _last:
    st.success(f"Completed AI sentiment for {_last['done']} grouped storie(s) in {_last['elapsed']:.1f}s.")
    # st.caption(
    #     f"Token usage this batch: input={_last['in_tok']:,} • output={_last['out_tok']:,} • "
    #     f"est. cost=${_last['batch_cost']:.4f} • session total=${_last['session_cost']:.4f}"
    # )
    if _last["errors"]:
        with st.expander(f"Completed with {len(_last['errors'])} error(s)", expanded=False):
            for err in _last["errors"]:
                st.write(err)

meter = st.session_state.api_meter
# usage_col1, usage_col2, usage_col3 = st.columns(3)
# with usage_col1:
#     st.metric("Session input tokens", f"{meter['in_tokens']:,}")
# with usage_col2:
#     st.metric("Session output tokens", f"{meter['out_tokens']:,}")
# with usage_col3:
#     st.metric("Session API cost", f"${meter['cost_usd']:.4f}")

source_rows = get_sentiment_source_rows(st.session_state.df_traditional)
population_size = len(source_rows)

# =========================
# STEP 1: DATASET PREP + CONFIG
# =========================
if not st.session_state.sentiment_config_step:
    st.subheader("Step 1: Prepare Sentiment Dataset & Configuration")
    st.caption("Sampling happens at the row level first, then the sampled rows are regrouped into unique stories.")

    reusable_tagging_sample = st.session_state.get("df_tagging_rows", None)
    has_reusable = isinstance(reusable_tagging_sample, pd.DataFrame) and not reusable_tagging_sample.empty

    options = [
        "Use full eligible dataset",
        "Use representative sample",
        "Set custom sample size",
    ]

    if has_reusable:
        options.insert(0, "Reuse tagging sample")

    mode_label = st.radio(
        "Sentiment dataset mode",
        options=options,
        index=1 if population_size > DEFAULT_MAX_FULL_ROWS else 0,
    )

    if mode_label == "Reuse tagging sample":
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
        stored_excluded_flags = st.session_state.get("sentiment_excluded_flags", default_excluded_flags)
        fallback_defaults = [f for f in default_excluded_flags if f in available_flags]
        preselected_flags = [f for f in stored_excluded_flags if f in available_flags] or fallback_defaults

        excluded_flags = st.multiselect(
            "Exclude coverage flags",
            options=available_flags,
            default=preselected_flags,
            help="Exclude selected flagged coverage from sentiment sampling.",
        )
        working_source_rows = apply_coverage_flag_exclusions(source_rows, excluded_flags)

    population_size = len(working_source_rows)
    recommended_sample = calculate_representative_sample_size(population_size) if population_size > 0 else 0

    # mode_label = st.radio(
    #     "Sentiment dataset mode",
    #     options=[
    #         "Use full eligible dataset",
    #         "Use representative sample",
    #         "Set custom sample size",
    #     ],
    #     index=1 if population_size > DEFAULT_MAX_FULL_ROWS else 0,
    # )
    #
    # if mode_label == "Use full eligible dataset":
    #     sample_mode = "full"
    # elif mode_label == "Use representative sample":
    #     sample_mode = "representative"
    # else:
    #     sample_mode = "custom"

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
            f"Full sentiment analysis is limited to {DEFAULT_MAX_FULL_ROWS:,} row-level mentions by default for cost and stability reasons."
        )
        full_override = st.checkbox(
            f"I understand the risk and want to allow full sentiment analysis over {DEFAULT_MAX_FULL_ROWS:,} mentions",
            value=False,
        )
    preview_rows = 0
    if sample_mode == "reuse_other_sample":
        preview_rows = len(reusable_tagging_sample) if has_reusable else 0
    elif sample_mode == "full":
        preview_rows = population_size if (
                    population_size <= DEFAULT_MAX_FULL_ROWS or full_override) else DEFAULT_MAX_FULL_ROWS
    elif sample_mode == "representative":
        preview_rows = recommended_sample
    else:
        preview_rows = int(custom_sample_size or 0)

    # preview_rows = None
    # if sample_mode == "full":
    #     preview_rows = population_size if (population_size <= DEFAULT_MAX_FULL_ROWS or full_override) else DEFAULT_MAX_FULL_ROWS
    # elif sample_mode == "representative":
    #     preview_rows = recommended_sample
    # else:
    #     preview_rows = int(custom_sample_size or 0)

    top_col1, top_col2 = st.columns(2)
    with top_col1:
        st.metric("Eligible mentions", f"{population_size:,}")
    with top_col2:
        st.metric("Estimated mentions used", f"{preview_rows:,}")

    if sample_mode == "representative":
        st.caption(f"Representative sample size estimate: {recommended_sample:,}")

    # if sample_mode == "reuse_other_sample" and has_reusable:
    #     st.caption(f"Using existing tagging sample: {len(reusable_sentiment_sample):,} rows")
    if sample_mode == "reuse_other_sample" and has_reusable:
        st.caption(f"Using existing tagging sample: {len(reusable_tagging_sample):,} rows")

    # if sample_mode == "reuse_other_sample":
    #     st.caption(f"Using existing tagging sample: {len(reused_rows):,} rows")

    st.divider()
    st.write("**Sentiment configuration**")

    col1, col2 = st.columns([2, 1])

    with col1:
        primary_name = st.text_input(
            "Primary Entity Name",
            value=st.session_state.ui_primary_names[0] if st.session_state.ui_primary_names else "",
        )

    with col2:
        sentiment_type = st.selectbox(
            "Sentiment Type",
            ["3-way", "5-way"],
            index=0 if st.session_state.ui_sentiment_type == "3-way" else 1,
        )
    model = DEFAULT_SENTIMENT_MODEL
    primary_names = [primary_name.strip()] if primary_name.strip() else []


    st.divider()

    alternate_names = st_tags(
        label="**Alternate names**",
        text="Press enter to add more",
        maxtags=10,
        value=st.session_state.ui_alternate_names,
        key="sentiment_alternate_names_tags",
    )
    spokespeople = st_tags(
        label="**Key spokespeople**",
        text="Press enter to add more",
        maxtags=10,
        value=st.session_state.ui_spokespeople,
        key="sentiment_spokespeople_tags",
    )
    products = st_tags(
        label="**Products or sub-brands**",
        text="Press enter to add more",
        maxtags=10,
        value=st.session_state.ui_products,
        key="sentiment_products_tags",
    )

    toning_rationale = st.text_area(
        "**Additional rationale, context, or guidance** (optional):",
        st.session_state.ui_toning_rationale,
        key="sentiment_toning_rationale_text",
    )

    prep_clicked = st.button("Prepare Sentiment Dataset", type="primary")

    if prep_clicked:
        if not primary_names or not str(primary_names[0]).strip():
            st.warning("Add at least one **Primary name** before preparing the sentiment dataset.")
            st.stop()

        start = time.time()
        #
        # results = prepare_sentiment_datasets(
        #     df_traditional=st.session_state.df_traditional,
        #     sample_mode=sample_mode,
        #     custom_sample_size=custom_sample_size,
        #     max_full_rows=DEFAULT_MAX_FULL_ROWS,
        #     full_override=full_override,
        # )

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

        st.session_state.df_sentiment_rows = results["df_sentiment_rows"]
        st.session_state.df_sentiment_grouped_rows = results["df_sentiment_grouped_rows"]
        st.session_state.df_sentiment_unique = results["df_sentiment_unique"]
        st.session_state.sentiment_sample_mode = sample_mode
        st.session_state.sentiment_sample_size = results["sample_size_used"]
        st.session_state.sentiment_full_override = full_override
        st.session_state.sentiment_excluded_flags = excluded_flags if sample_mode != "reuse_other_sample" else []
        st.session_state.sentiment_config_step = True
        st.session_state.sentiment_elapsed_time = time.time() - start

        grouped, unique = ensure_ai_sentiment_columns(
            st.session_state.df_sentiment_grouped_rows,
            st.session_state.df_sentiment_unique,
        )
        st.session_state.df_sentiment_grouped_rows = grouped
        st.session_state.df_sentiment_unique = unique

        build_sentiment_configuration(
            session_state=st.session_state,
            primary_names=primary_names,
            alternate_names=alternate_names,
            spokespeople=spokespeople,
            products=products,
            toning_rationale=toning_rationale,
            sentiment_type=sentiment_type,
            model=model,
        )

        st.rerun()

    st.stop()

# =========================
# STEP 2: RUN AI SENTIMENT
# =========================
st.subheader("Step 2: Run AI Sentiment")

top_col1, top_col2, top_col3, top_col4 = st.columns(4)
with top_col1:
    st.metric("Eligible mentions", f"{population_size:,}")
with top_col2:
    st.metric("Sample used", f"{len(st.session_state.df_sentiment_rows):,}")
with top_col3:
    st.metric("Grouped stories", f"{len(st.session_state.df_sentiment_unique):,}")
with top_col4:
    if st.button("Reset Sentiment Dataset"):
        reset_sentiment_config_state(st.session_state)
        st.rerun()

# config_col1, config_col2, config_col3 = st.columns(3)
# with config_col1:
#     st.metric("Sentiment type", st.session_state.get("sentiment_type", "3-way"))
# with config_col2:
#     st.metric("Model", st.session_state.get("model_choice", "gpt-5.4-nano"))
# with config_col3:
#     st.metric("Prep time", f"{st.session_state.sentiment_elapsed_time:.2f}s")

remaining_df = get_remaining_sentiment_rows(
    st.session_state.df_sentiment_unique,
    st.session_state.df_sentiment_grouped_rows,
)

st.info("Run AI sentiment in batches on grouped stories from the prepared sentiment dataset.")

remaining_count = len(remaining_df)

if remaining_count == 0:
    batch_size = 0
    st.info("No grouped stories remain for AI sentiment.")
else:
    default_batch_value = st.session_state.get("sentiment_batch_size", DEFAULT_SENTIMENT_BATCH_SIZE)
    default_batch_value = min(default_batch_value, remaining_count)

    batch_size = st.number_input(
        "Batch size (0 for all remaining rows)",
        min_value=0,
        max_value=remaining_count,
        value=default_batch_value,
        step=1,
        key="sentiment_batch_size",
    )

if batch_size > 0:
    batch_df = remaining_df.iloc[:batch_size].copy()
else:
    batch_df = remaining_df.copy()

st.write(f"Selected grouped stories for analysis: {len(batch_df):,}")

run_clicked = st.button("Run AI Sentiment", type="primary", disabled=(len(batch_df) == 0))
reset_ai_clicked = st.button("Reset AI Results")

if reset_ai_clicked:
    unique, grouped = reset_ai_sentiment_results(
        st.session_state.df_sentiment_unique,
        st.session_state.df_sentiment_grouped_rows,
    )
    st.session_state.df_sentiment_unique = unique
    st.session_state.df_sentiment_grouped_rows = grouped
    st.success("Reset AI sentiment results.")
    st.rerun()

if run_clicked:
    st.session_state.pop("__last_sentiment_batch_summary__", None)

    progress_bar = st.progress(0.0)
    total_in = 0
    total_out = 0
    errors = []
    completed = 0
    total = len(batch_df)
    start_time = time.time()

    rows_for_workers = [(idx, row.to_dict()) for idx, row in batch_df.iterrows()]

    with ThreadPoolExecutor(max_workers=DEFAULT_SENTIMENT_MAX_WORKERS) as executor:
        future_map = {
            executor.submit(
                analyze_sentiment_worker,
                row_tuple,
                st.session_state.get("pre_prompt", ""),
                st.session_state.get("sentiment_instruction", ""),
                st.session_state.get("post_prompt", ""),
                st.session_state.get("functions", []),
                st.session_state.get("model_choice", "gpt-5.4-nano"),
                st.session_state.get("sentiment_type", "3-way"),
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
                    st.session_state.df_sentiment_unique = apply_sentiment_result_to_unique_df(
                        st.session_state.df_sentiment_unique,
                        original_index=original_index,
                        result=result,
                    )

            except Exception as e:
                errors.append(str(e))

            progress_bar.progress(completed / max(1, total))

    # st.session_state.df_sentiment_grouped_rows = cascade_sentiment_to_grouped_rows(
    #     st.session_state.df_sentiment_grouped_rows,
    #     st.session_state.df_sentiment_unique,
    # )
    st.session_state.df_sentiment_grouped_rows = cascade_sentiment_to_grouped_rows(
        st.session_state.df_sentiment_grouped_rows,
        st.session_state.df_sentiment_unique,
    )

    # keep row-level sample export dataframe in sync too
    st.session_state.df_sentiment_rows = st.session_state.df_sentiment_grouped_rows.copy()

    apply_usage_to_session(total_in, total_out, st.session_state.get("model_choice", "gpt-5.4-nano"))

    batch_cost = estimate_cost_usd(total_in, total_out, st.session_state.get("model_choice", "gpt-5.4-nano"))
    session_cost = get_api_cost_usd()

    st.session_state["__last_sentiment_batch_summary__"] = {
        "done": total,
        "elapsed": time.time() - start_time,
        "in_tok": total_in,
        "out_tok": total_out,
        "batch_cost": batch_cost,
        "session_cost": session_cost,
        "errors": errors,
    }

    st.rerun()

with st.expander("Prompt configuration preview", expanded=False):
    if st.session_state.get("last_saved"):
        st.caption(f"Last saved: {st.session_state.last_saved}")

    if "pre_prompt" in st.session_state:
        st.write("**Pre-prompt:**")
        st.code(st.session_state.pre_prompt)
    if "post_prompt" in st.session_state:
        st.write("**Labeling Clarifications:**")
        st.code(st.session_state.post_prompt)
    if "sentiment_instruction" in st.session_state:
        st.write("**Labeling Instruction:**")
        st.code(st.session_state.sentiment_instruction)

    st.write("**Highlight keywords:**")
    st.write(st.session_state.get("highlight_keyword", []))

with st.expander("Grouped sentiment dataset preview", expanded=False):
    st.dataframe(
        st.session_state.df_sentiment_unique.head(200),
        use_container_width=True,
        hide_index=True,
    )

remaining_now = get_remaining_sentiment_rows(
    st.session_state.df_sentiment_unique,
    st.session_state.df_sentiment_grouped_rows,
)
st.caption(f"Groups remaining (no human label & no AI): {len(remaining_now):,}")
