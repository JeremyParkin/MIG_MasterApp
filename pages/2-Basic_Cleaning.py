# 2-Basic_Cleaning.py

from __future__ import annotations

import gc
import time
import warnings
import pandas as pd
import streamlit as st

from processing.standard_cleaning import run_standard_cleaning
from processing.coverage_flags import add_coverage_flags
from processing.effective_reach import (
    apply_effective_reach_traditional,
    apply_effective_reach_social,
)

from processing.story_grouping import (
    cluster_by_media_type,
    build_unique_story_table,
    mark_prime_examples,
)

from utils.formatting import format_number, NUMERIC_FORMAT_DICT
from utils.io import normalize_uploaded_dataframe

warnings.filterwarnings("ignore")

STAGED_BASIC_CLEANING_THRESHOLD = 10_000

st.title("Basic Cleaning")
st.caption("Standardize the raw export, remove duplicates, calculate effective reach, and group similar coverage into unique stories.")


def ensure_basic_cleaning_state() -> None:
    """Initialize any session keys this page depends on."""
    df_keys = [
        "df_social",
        "df_dupes",
        "df_ai_grouped",
        "df_ai_unique",
    ]
    for key in df_keys:
        if key not in st.session_state:
            st.session_state[key] = pd.DataFrame()

    if "standard_step" not in st.session_state:
        st.session_state.standard_step = False
    if "basic_cleaning_stage" not in st.session_state:
        st.session_state.basic_cleaning_stage = 0
    if "basic_cleaning_elapsed" not in st.session_state:
        st.session_state.basic_cleaning_elapsed = None
    if "basic_cleaning_part_durations" not in st.session_state:
        st.session_state.basic_cleaning_part_durations = {}
    if "basic_cleaning_merge_online" not in st.session_state:
        st.session_state.basic_cleaning_merge_online = True
    if "basic_cleaning_drop_dupes" not in st.session_state:
        st.session_state.basic_cleaning_drop_dupes = True
    if "basic_cleaning_started_at" not in st.session_state:
        st.session_state.basic_cleaning_started_at = None


def build_post_upload_baseline_df() -> pd.DataFrame:
    raw_df = st.session_state.get("df_untouched")
    if not isinstance(raw_df, pd.DataFrame) or raw_df.empty:
        return pd.DataFrame()

    baseline_df = normalize_uploaded_dataframe(raw_df)

    category_columns = [
        "Sentiment",
        "Continent",
        "Country",
        "Prov/State",
        "City",
        "Language",
    ]
    for column in category_columns:
        if column in baseline_df.columns:
            baseline_df[column] = baseline_df[column].astype("category")

    return baseline_df


def reset_to_post_upload_baseline() -> None:
    """Reset all downstream workflow state back to the completed upload baseline."""
    baseline_df = build_post_upload_baseline_df()
    preserved_values = {
        "df_untouched": st.session_state.get("df_untouched", pd.DataFrame()),
        "df_traditional": baseline_df,
        "uploaded_filename": st.session_state.get("uploaded_filename"),
        "original_ave_col": st.session_state.get("original_ave_col"),
        "ave_col": st.session_state.get("ave_col", "AVE"),
        "export_name": st.session_state.get("export_name", ""),
        "client_name": st.session_state.get("client_name", ""),
        "client": st.session_state.get("client", ""),
        "period": st.session_state.get("period", ""),
        "upload_step": True,
    }

    for key in list(st.session_state.keys()):
        if key not in preserved_values:
            del st.session_state[key]

    for key, value in preserved_values.items():
        st.session_state[key] = value

    ensure_basic_cleaning_state()
    gc.collect()


def get_pre_standard_source_df() -> pd.DataFrame:
    baseline_df = build_post_upload_baseline_df()
    if not baseline_df.empty:
        return baseline_df

    fallback_df = st.session_state.get("df_traditional")
    if isinstance(fallback_df, pd.DataFrame):
        return fallback_df.copy()

    return pd.DataFrame()


def run_basic_cleaning_stage_1() -> None:
    start = time.time()
    source_df = get_pre_standard_source_df()
    cleaning_results = run_standard_cleaning(
        df=source_df,
        merge_online=st.session_state.basic_cleaning_merge_online,
        drop_dupes=st.session_state.basic_cleaning_drop_dupes,
        add_flags=False,
    )
    st.session_state.df_traditional = cleaning_results["df_traditional"]
    st.session_state.df_social = cleaning_results["df_social"]
    st.session_state.df_dupes = cleaning_results["df_dupes"]
    st.session_state.df_ai_grouped = pd.DataFrame()
    st.session_state.df_ai_unique = pd.DataFrame()
    duration = time.time() - start
    st.session_state.basic_cleaning_elapsed = (st.session_state.basic_cleaning_elapsed or 0.0) + duration
    st.session_state.basic_cleaning_part_durations[1] = duration
    st.session_state.basic_cleaning_stage = 1
    del source_df, cleaning_results
    gc.collect()


def run_basic_cleaning_stage_2() -> None:
    start = time.time()
    st.session_state.df_traditional = apply_effective_reach_traditional(st.session_state.df_traditional)
    st.session_state.df_social = apply_effective_reach_social(st.session_state.df_social)
    st.session_state.df_traditional = add_coverage_flags(st.session_state.df_traditional)
    duration = time.time() - start
    st.session_state.basic_cleaning_elapsed = (st.session_state.basic_cleaning_elapsed or 0.0) + duration
    st.session_state.basic_cleaning_part_durations[2] = duration
    st.session_state.basic_cleaning_stage = 2
    gc.collect()


def run_basic_cleaning_stage_3() -> None:
    start = time.time()
    df_ai_grouped = cluster_by_media_type(
        df=st.session_state.df_traditional,
        similarity_threshold=0.935,
        max_batch_size=1800,
    )
    df_ai_grouped = mark_prime_examples(df_ai_grouped)
    df_ai_unique = build_unique_story_table(df_ai_grouped)

    st.session_state.df_traditional = df_ai_grouped
    st.session_state.df_ai_grouped = df_ai_grouped
    st.session_state.df_ai_unique = df_ai_unique
    duration = time.time() - start
    st.session_state.basic_cleaning_elapsed = (st.session_state.basic_cleaning_elapsed or 0.0) + duration
    st.session_state.basic_cleaning_part_durations[3] = duration
    st.session_state.standard_step = True
    st.session_state.basic_cleaning_stage = 3
    gc.collect()


def render_stage_progress() -> None:
    current_stage = int(st.session_state.get("basic_cleaning_stage", 0))
    part_durations = st.session_state.get("basic_cleaning_part_durations", {})
    progress = min(max(current_stage / 3, 0.0), 1.0)
    st.progress(progress)

    if current_stage == 1:
        duration = part_durations.get(1)
        duration_text = f" in {duration:.1f} seconds" if duration is not None else ""
        st.info(f"Part 1 of 3 complete{duration_text}. Click Continue to keep going.")
    elif current_stage == 2:
        duration = part_durations.get(2)
        duration_text = f" in {duration:.1f} seconds" if duration is not None else ""
        st.info(f"Part 2 of 3 complete{duration_text}. Click Continue to finish.")


def should_use_staged_basic_cleaning() -> bool:
    current_stage = int(st.session_state.get("basic_cleaning_stage", 0))
    if current_stage > 0:
        return True

    source_df = get_pre_standard_source_df()
    return len(source_df) >= STAGED_BASIC_CLEANING_THRESHOLD


def render_preview_dataframe(df: pd.DataFrame, preview_rows: int = 1000) -> None:
    """Render a dataframe preview safely."""
    if df is None or df.empty:
        st.info("No data available.")
        return

    preview_df = df.head(preview_rows).copy()
    cell_count = preview_df.shape[0] * preview_df.shape[1]

    st.caption(f"Showing first {min(len(df), preview_rows):,} rows of {len(df):,}.")

    if cell_count <= 262144:
        st.dataframe(
            preview_df.style.format(NUMERIC_FORMAT_DICT, na_rep=" "),
            use_container_width=True,
        )
    else:
        st.dataframe(
            preview_df.fillna(""),
            use_container_width=True,
        )



def build_type_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """Build a compact type breakdown table."""
    if df is None or df.empty or "Type" not in df.columns:
        return pd.DataFrame()

    agg_dict = {}

    if "Mentions" in df.columns:
        agg_dict["Mentions"] = ("Mentions", "sum")
    if "Impressions" in df.columns:
        agg_dict["Impressions"] = ("Impressions", "sum")
    if "Effective Reach" in df.columns:
        agg_dict["Effective Reach"] = ("Effective Reach", "sum")

    if not agg_dict:
        return pd.DataFrame()

    breakdown = (
        df.groupby("Type", dropna=False)
        .agg(**agg_dict)
        .reset_index()
    )

    sort_candidates = [c for c in ["Effective Reach", "Impressions", "Mentions"] if c in breakdown.columns]
    if sort_candidates:
        breakdown = breakdown.sort_values(by=sort_candidates, ascending=False)

    return breakdown


def render_dataset_expander(
    df: pd.DataFrame,
    title: str,
    preview_rows: int = 1000,
    unique_mentions: int | None = None,
    show_total_engagements: bool = False,
) -> None:
    """Render compact metrics + type breakdown + preview."""
    if df is None or len(df) == 0:
        return

    with st.expander(title):
        col1h, col2h = st.columns(2)
        with col1h:
            st.subheader("Basic Metrics")
        with col2h:
            st.subheader("Media Type")

        col1, col1b, col2 = st.columns([1, 1, 2])

        with col1:
            mentions_total = (
                pd.to_numeric(df["Mentions"], errors="coerce").fillna(0).sum()
                if "Mentions" in df.columns
                else len(df)
            )
            st.metric("Total Mentions", f"{int(mentions_total):,}")

            if show_total_engagements and "Engagements" in df.columns:
                engagements_total = pd.to_numeric(df["Engagements"], errors="coerce").fillna(0).sum()
                st.metric(
                    "Total Engagements",
                    format_number(engagements_total),
                    help=f"{int(engagements_total):,}",
                )

            if unique_mentions is not None:
                st.metric("Unique Mentions", f"{int(unique_mentions):,}")



        with col1b:


            if "Impressions" in df.columns:
                impressions_total = pd.to_numeric(df["Impressions"], errors="coerce").fillna(0).sum()
                st.metric(
                    "Impressions",
                    format_number(impressions_total),
                    help=f"{int(impressions_total):,}",
                )

            if "Effective Reach" in df.columns:
                er_total = pd.to_numeric(df["Effective Reach"], errors="coerce").fillna(0).sum()
                st.metric(
                    "Effective Reach",
                    format_number(er_total),
                    help=f"{int(er_total):,}",
                )

        with col2:
            breakdown = build_type_breakdown(df)

            if not breakdown.empty:
                st.dataframe(
                    breakdown.style.format(NUMERIC_FORMAT_DICT, na_rep=""),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info("No type breakdown available.")

        st.subheader("Data Preview")
        render_preview_dataframe(df, preview_rows=preview_rows)


ensure_basic_cleaning_state()

if not st.session_state.get("upload_step", False):
    st.error("Please upload a CSV/XLSX before trying this step.")
    st.stop()

source_df_for_page = get_pre_standard_source_df()
if source_df_for_page.empty:
    st.error("Uploaded data is missing or empty. Please return to Getting Started.")
    st.stop()


if st.session_state.standard_step:
    top_col1, top_col2 = st.columns([3, 1])

    with top_col1:
        st.success("Basic cleaning completed.")
        if "basic_cleaning_elapsed" in st.session_state:
            st.caption(f"Completed in {st.session_state.basic_cleaning_elapsed:.2f} seconds.")

    with top_col2:
        if st.button("Reset Basic Cleaning", type="secondary"):
            reset_to_post_upload_baseline()
            st.rerun()

    original_rows = len(st.session_state.get("df_untouched", pd.DataFrame()))
    traditional_rows = len(st.session_state.get("df_traditional", pd.DataFrame()))
    social_rows = len(st.session_state.get("df_social", pd.DataFrame()))
    duplicate_rows = len(st.session_state.get("df_dupes", pd.DataFrame()))
    reconciled_total = traditional_rows + social_rows + duplicate_rows
    reconciliation_matches = reconciled_total == original_rows

    st.subheader("Row Reconciliation")
    rec1, rec2, rec3, rec4, rec5 = st.columns(5)
    with rec1:
        st.metric("Original Rows", f"{original_rows:,}")
    with rec2:
        st.metric("Traditional", f"{traditional_rows:,}")
    with rec3:
        st.metric("Social", f"{social_rows:,}")
    with rec4:
        st.metric("Deleted Duplicates", f"{duplicate_rows:,}")
    with rec5:
        st.metric("Reconciled Total", f"{reconciled_total:,}")

    if reconciliation_matches:
        st.info("Row reconciliation passed: Original Rows match Traditional + Social + Deleted Duplicates.")
    else:
        delta = reconciled_total - original_rows
        delta_text = f"{delta:+,}"
        st.error(
            "Row reconciliation failed: Original Rows do not match Traditional + Social + Deleted Duplicates "
            f"({reconciled_total:,} vs {original_rows:,}, delta {delta_text})."
        )

    traditional_unique_mentions = None
    if not st.session_state.df_ai_unique.empty and "Group Count" in st.session_state.df_ai_unique.columns:
        traditional_unique_mentions = len(st.session_state.df_ai_unique)

    render_dataset_expander(
        st.session_state.df_traditional,
        "Traditional",
        preview_rows=1000,
        unique_mentions=traditional_unique_mentions,
        show_total_engagements=False,
    )

    render_dataset_expander(
        st.session_state.df_social,
        "Social",
        preview_rows=200,
        unique_mentions=None,
        show_total_engagements=True,
    )

    render_dataset_expander(
        st.session_state.df_dupes,
        "Deleted Duplicates",
        preview_rows=200,
        unique_mentions=None,
        show_total_engagements=False,
    )
else:
    current_stage = int(st.session_state.get("basic_cleaning_stage", 0))
    staged_mode = should_use_staged_basic_cleaning()
    source_rows = len(source_df_for_page)

    if staged_mode:
        st.info(
            f"Large dataset detected ({source_rows:,} rows). Basic Cleaning will run in 3 parts to improve stability."
        )

    if current_stage == 0:
        with st.form("basic_cleaning_form"):
            st.subheader("Cleaning Options")

            merge_online = st.checkbox(
                "Merge 'blogs' and 'press releases' into 'Online'",
                value=st.session_state.get("basic_cleaning_merge_online", True),
                help="Combine ONLINE NEWS, PRESS RELEASE, and BLOGS into ONLINE.",
            )

            drop_dupes = st.checkbox(
                "Drop duplicates",
                value=st.session_state.get("basic_cleaning_drop_dupes", True),
                help="After media types are normalized, remove non-broadcast duplicates by URL and by Type + Outlet + Headline, run separate broadcast duplicate checks using outlet, media type, date, time proximity, and snippet similarity, and apply conservative exact-match social dedupe within each network.",
            )

            submitted = st.form_submit_button("Run Basic Cleaning", type="primary")

        if submitted:
            st.session_state.basic_cleaning_merge_online = merge_online
            st.session_state.basic_cleaning_drop_dupes = drop_dupes
            st.session_state.basic_cleaning_started_at = time.time()
            st.session_state.basic_cleaning_elapsed = 0.0
            st.session_state.basic_cleaning_part_durations = {}
            if staged_mode:
                with st.spinner("Running part 1 of 3..."):
                    run_basic_cleaning_stage_1()
            else:
                with st.spinner("Running basic cleaning..."):
                    run_basic_cleaning_stage_1()
                    run_basic_cleaning_stage_2()
                    run_basic_cleaning_stage_3()
            st.rerun()
    elif staged_mode:
        render_stage_progress()

        action_cols = st.columns([0.34, 0.2, 0.46], gap="small")
        with action_cols[0]:
            continue_label = "Continue" if current_stage == 1 else "Finish Basic Cleaning"
            if st.button(continue_label, type="primary", use_container_width=True):
                spinner_text = "Running part 2 of 3..." if current_stage == 1 else "Running part 3 of 3..."
                with st.spinner(spinner_text):
                    if current_stage == 1:
                        run_basic_cleaning_stage_2()
                    elif current_stage == 2:
                        run_basic_cleaning_stage_3()
                st.rerun()
        with action_cols[1]:
            if st.button("Start Over", use_container_width=True):
                reset_to_post_upload_baseline()
                st.rerun()
