from __future__ import annotations

from pathlib import Path
import runpy
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

import altair as alt
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
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
    build_sentiment_distribution,
)
from utils.api_meter import (
    init_api_meter,
    apply_usage_to_session,
    estimate_cost_usd,
    get_api_cost_usd,
)

warnings.filterwarnings("ignore")

st.markdown('<div id="sentiment-top-anchor"></div>', unsafe_allow_html=True)
st.title("Sentiment")
st.caption("Prepare the sentiment dataset, run AI sentiment analysis in batches, and complete human review in one workflow.")
st.session_state.setdefault("sentiment_section", "Setup")

if st.session_state.pop("sentiment_scroll_to_top", False):
    components.html(
        """
        <script>
        const anchor = window.parent.document.getElementById("sentiment-top-anchor");
        if (anchor) {
          anchor.scrollIntoView({behavior: "instant", block: "start"});
        } else {
          window.parent.scrollTo({top: 0, behavior: "instant"});
        }
        </script>
        """,
        height=0,
    )

if not st.session_state.get("standard_step", False):
    st.error("Please complete Basic Cleaning before trying this step.")
    st.stop()

init_sentiment_config_state(st.session_state)
init_ai_sentiment_state(st.session_state)
init_api_meter()

_last = st.session_state.get("__last_sentiment_batch_summary__")
if _last and st.session_state.get("sentiment_section") == "Run":
    st.success(f"Completed AI sentiment for {_last['done']} grouped storie(s) in {_last['elapsed']:.1f}s.")
    if _last["errors"]:
        with st.expander(f"Completed with {len(_last['errors'])} error(s)", expanded=False):
            for err in _last["errors"]:
                st.write(err)

source_rows = get_sentiment_source_rows(st.session_state.df_traditional)
population_size = len(source_rows)


def _get_sentiment_order(sentiment_type: str) -> list[str]:
    if str(sentiment_type).strip().lower().startswith("5"):
        return [
            "VERY POSITIVE",
            "SOMEWHAT POSITIVE",
            "NEUTRAL",
            "SOMEWHAT NEGATIVE",
            "VERY NEGATIVE",
            "NOT RELEVANT",
        ]
    return ["POSITIVE", "NEUTRAL", "NEGATIVE", "NOT RELEVANT"]


def _get_sentiment_color_mapping() -> dict[str, str]:
    return {
        "POSITIVE": "#2ecc71",
        "NEUTRAL": "#f1c40f",
        "NEGATIVE": "#e74c3c",
        "VERY POSITIVE": "#0f9d58",
        "SOMEWHAT POSITIVE": "#72cc4a",
        "SOMEWHAT NEGATIVE": "#e67e22",
        "VERY NEGATIVE": "#8e1f1f",
        "NOT RELEVANT": "#6b7280",
    }


def _build_distribution_view(sentiment_dist: pd.DataFrame, sentiment_type: str, include_not_relevant: bool) -> tuple[pd.DataFrame, alt.Chart]:
    order = _get_sentiment_order(sentiment_type)
    working = sentiment_dist.copy()

    if not include_not_relevant:
        working = working[working["Sentiment"] != "NOT RELEVANT"].copy()
        order = [item for item in order if item != "NOT RELEVANT"]

    total = int(working["Count"].sum())
    working["Share"] = working["Count"] / total if total > 0 else 0.0
    working["Sentiment"] = pd.Categorical(working["Sentiment"], categories=order, ordered=True)
    working = working.sort_values("Sentiment").reset_index(drop=True)
    working["Share Label"] = (working["Share"] * 100).map(lambda x: f"{x:.1f}%")

    color_mapping = _get_sentiment_color_mapping()
    color_scale = alt.Scale(
        domain=order,
        range=[color_mapping.get(sentiment, "#9ca3af") for sentiment in order],
    )

    base = alt.Chart(working).encode(
        y=alt.Y(
            "Sentiment:N",
            sort=order,
            axis=alt.Axis(title=None, labelLimit=220, labelPadding=10),
        )
    )

    bars = base.mark_bar(cornerRadiusEnd=3).encode(
        x=alt.X("Count:Q", axis=alt.Axis(title=None, grid=True, tickMinStep=1)),
        color=alt.Color("Sentiment:N", scale=color_scale, legend=None),
        tooltip=[
            "Sentiment",
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

    chart = (
        (bars + text)
        .properties(height=max(180, 48 * len(working)))
        .configure_view(strokeWidth=0)
        .configure_axis(
            gridColor="rgba(148, 163, 184, 0.18)",
            domain=False,
            tickColor="rgba(148, 163, 184, 0.35)",
            labelColor="#E5E7EB",
            titleColor="#E5E7EB",
        )
    )

    return working, chart


def _format_sample_mode(mode: str) -> str:
    return {
        "reuse_other_sample": "Reused tagging sample",
        "full": "Full eligible dataset",
        "representative": "Representative sample",
        "custom": "Custom sample",
    }.get(mode, str(mode))


st.markdown(
    """
    <style>
    .sentiment-step-note {
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

step1, step2, step3, step4, step5 = st.columns(5, gap="small")
with step1:
    if st.button(
        "1. Setup",
        key="sentiment_nav_setup",
        use_container_width=True,
        type="primary" if st.session_state.sentiment_section == "Setup" else "secondary",
    ):
        st.session_state.sentiment_section = "Setup"
        st.session_state.sentiment_scroll_to_top = True
        st.rerun()
with step2:
    if st.button(
        "2. Run",
        key="sentiment_nav_run",
        use_container_width=True,
        type="primary" if st.session_state.sentiment_section == "Run" else "secondary",
    ):
        st.session_state.sentiment_section = "Run"
        st.session_state.sentiment_scroll_to_top = True
        st.rerun()
with step3:
    if st.button(
        "3. AI Pre-Review",
        key="sentiment_nav_pre_review",
        use_container_width=True,
        type="primary" if st.session_state.sentiment_section == "AI Pre-Review" else "secondary",
    ):
        st.session_state.sentiment_section = "AI Pre-Review"
        st.session_state.sentiment_scroll_to_top = True
        st.rerun()
with step4:
    if st.button(
        "4. Spot Checks",
        key="sentiment_nav_review",
        use_container_width=True,
        type="primary" if st.session_state.sentiment_section == "Spot Checks" else "secondary",
    ):
        st.session_state.sentiment_section = "Spot Checks"
        st.session_state.sentiment_scroll_to_top = True
        st.rerun()
with step5:
    if st.button(
        "5. Distribution",
        key="sentiment_nav_distribution",
        use_container_width=True,
        type="primary" if st.session_state.sentiment_section == "Distribution" else "secondary",
    ):
        st.session_state.sentiment_section = "Distribution"
        st.session_state.sentiment_scroll_to_top = True
        st.rerun()

st.markdown(
    '<div class="sentiment-step-note">Work left to right: prepare the sentiment dataset, run AI sentiment in batches, run AI pre-review, complete spot checks, then review the final distribution.</div>',
    unsafe_allow_html=True,
)


if st.session_state.sentiment_section == "Setup":
    st.session_state.sentiment_review_embedded = False
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

    default_index = 1 if population_size > DEFAULT_MAX_FULL_ROWS else 0
    if has_reusable:
        default_index = 0

    mode_label = st.radio(
        "Sentiment dataset mode",
        options=options,
        index=default_index,
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

    if sample_mode == "reuse_other_sample":
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
        st.session_state.sentiment_section = "Run"
        st.session_state.sentiment_scroll_to_top = True
        st.rerun()

    if st.session_state.sentiment_config_step:
        st.info("A sentiment dataset is already prepared. You can adjust the setup and prepare again, or move to Run.")
    st.stop()


if not st.session_state.sentiment_config_step:
    if st.session_state.sentiment_section in {"Run", "Review"}:
        st.info("Prepare the sentiment dataset in Setup before running or reviewing AI sentiment.")
    st.stop()


remaining_df = get_remaining_sentiment_rows(
    st.session_state.df_sentiment_unique,
    st.session_state.df_sentiment_grouped_rows,
)
remaining_count = len(remaining_df)

if st.session_state.sentiment_section == "Run":
    st.session_state.sentiment_review_embedded = False
    st.subheader("Step 2: Run AI Sentiment")
    processed_count = len(st.session_state.df_sentiment_unique) - remaining_count

    top_col1, top_col2, top_col3, top_col4, top_col5 = st.columns(5)
    with top_col1:
        st.metric("Eligible mentions", f"{population_size:,}")
    with top_col2:
        st.metric("Sample used", f"{len(st.session_state.df_sentiment_rows):,}")
    with top_col3:
        st.metric("Grouped stories", f"{len(st.session_state.df_sentiment_unique):,}")
    with top_col4:
        st.metric("Processed stories", f"{processed_count:,}")
    with top_col5:
        st.metric("Remaining stories", f"{remaining_count:,}")

    reset_col1, reset_col2 = st.columns([4, 1])
    with reset_col2:
        if st.button("Reset Sentiment Dataset"):
            reset_sentiment_config_state(st.session_state)
            st.session_state.sentiment_section = "Setup"
            st.session_state.sentiment_scroll_to_top = True
            st.rerun()

    config_col1, config_col2, config_col3 = st.columns(3)
    with config_col1:
        st.caption(f"Dataset mode: {_format_sample_mode(st.session_state.get('sentiment_sample_mode', 'representative'))}")
    with config_col2:
        st.caption(f"Sentiment type: {st.session_state.get('sentiment_type', '3-way')}")
    with config_col3:
        st.caption(f"Model: {st.session_state.get('model_choice', DEFAULT_SENTIMENT_MODEL)}")

    st.info("Run AI sentiment in batches on grouped stories from the prepared sentiment dataset.")

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
                    st.session_state.get("model_choice", DEFAULT_SENTIMENT_MODEL),
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

        st.session_state.df_sentiment_grouped_rows = cascade_sentiment_to_grouped_rows(
            st.session_state.df_sentiment_grouped_rows,
            st.session_state.df_sentiment_unique,
        )
        st.session_state.df_sentiment_rows = st.session_state.df_sentiment_grouped_rows.copy()

        model_choice = st.session_state.get("model_choice", DEFAULT_SENTIMENT_MODEL)
        apply_usage_to_session(total_in, total_out, model_choice)
        batch_cost = estimate_cost_usd(total_in, total_out, model_choice)
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
        st.session_state.sentiment_section = "Run"
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
    with st.expander("Current sentiment distribution", expanded=False):
        sentiment_type = st.session_state.get("sentiment_type")
        sentiment_dist = build_sentiment_distribution(st.session_state.df_sentiment_unique, sentiment_type)
        include_not_relevant_preview = st.toggle(
            "Include Not Relevant in percentages",
            value=True,
            key="sentiment_distribution_include_nr_preview",
        )
        sentiment_table, sentiment_chart = _build_distribution_view(
            sentiment_dist=sentiment_dist,
            sentiment_type=sentiment_type,
            include_not_relevant=include_not_relevant_preview,
        )
        preview_col1, preview_col2 = st.columns([1.35, 1], gap="large")
        with preview_col1:
            st.altair_chart(sentiment_chart, use_container_width=True)
        with preview_col2:
            display_table = sentiment_table[["Sentiment", "Count", "Share Label"]].rename(
                columns={"Share Label": "Share"}
            )
            st.dataframe(display_table, hide_index=True, use_container_width=True)

    st.stop()

if st.session_state.sentiment_section == "AI Pre-Review":
    st.session_state.sentiment_review_embedded = True
    st.session_state.spot_checks_mode = "pre_review"
    legacy_or_current = Path(__file__).resolve().parent / "10-Spot_Checks.py"
    runpy.run_path(str(legacy_or_current), run_name="__main__")
    st.stop()

if st.session_state.sentiment_section == "Spot Checks":
    st.session_state.sentiment_review_embedded = True
    st.session_state.spot_checks_mode = "spot_checks"
    legacy_or_current = Path(__file__).resolve().parent / "10-Spot_Checks.py"
    runpy.run_path(str(legacy_or_current), run_name="__main__")
    st.stop()

st.session_state.sentiment_review_embedded = False
st.session_state.spot_checks_mode = "distribution"
st.subheader("Step 5: Sentiment Distribution")
st.caption("Review the current final sentiment mix after AI sentiment and spot-check assignments.")

sentiment_type = st.session_state.get("sentiment_type")
sentiment_dist = build_sentiment_distribution(st.session_state.df_sentiment_unique, sentiment_type)

dist_col1, dist_col2, dist_col3 = st.columns(3)
with dist_col1:
    st.metric("Grouped stories", f"{len(st.session_state.df_sentiment_unique):,}")
with dist_col2:
    remaining_now = get_remaining_sentiment_rows(
        st.session_state.df_sentiment_unique,
        st.session_state.df_sentiment_grouped_rows,
    )
    st.metric("Remaining unlabeled", f"{len(remaining_now):,}")
with dist_col3:
    assigned_count = int(
        st.session_state.df_sentiment_unique.get("Assigned Sentiment", pd.Series(dtype="object")).fillna("").astype(str).str.strip().ne("").sum()
    )
    st.metric("Human-assigned", f"{assigned_count:,}")

include_not_relevant_final = st.toggle(
    "Include Not Relevant in percentages",
    value=True,
    key="sentiment_distribution_include_nr_final",
)

sentiment_table, sentiment_chart = _build_distribution_view(
    sentiment_dist=sentiment_dist,
    sentiment_type=sentiment_type,
    include_not_relevant=include_not_relevant_final,
)

dist_view1, dist_view2 = st.columns([1.35, 1], gap="large")
with dist_view1:
    st.altair_chart(sentiment_chart, use_container_width=True)
with dist_view2:
    display_table = sentiment_table[["Sentiment", "Count", "Share Label"]].rename(
        columns={"Share Label": "Share"}
    )
    st.dataframe(display_table, hide_index=True, use_container_width=True)
