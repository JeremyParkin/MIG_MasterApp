# 2-Basic_Cleaning.py

from __future__ import annotations

import time
import warnings
import pandas as pd
import streamlit as st

from processing.standard_cleaning import run_standard_cleaning
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

warnings.filterwarnings("ignore")

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


def reset_basic_cleaning() -> None:
    """Reset this step while preserving uploaded source data."""
    if "df_traditional_pre_standard" in st.session_state and isinstance(
        st.session_state.df_traditional_pre_standard, pd.DataFrame
    ):
        st.session_state.df_traditional = st.session_state.df_traditional_pre_standard.copy()

    st.session_state.df_social = pd.DataFrame()
    st.session_state.df_dupes = pd.DataFrame()
    st.session_state.df_ai_grouped = pd.DataFrame()
    st.session_state.df_ai_unique = pd.DataFrame()
    st.session_state.standard_step = False


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

if "df_traditional_pre_standard" not in st.session_state or st.session_state.df_traditional_pre_standard.empty:
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
            reset_basic_cleaning()
            st.rerun()

    traditional_unique_mentions = None
    if not st.session_state.df_ai_unique.empty and "Group Count" in st.session_state.df_ai_unique.columns:
        traditional_unique_mentions = len(st.session_state.df_ai_unique)

    render_dataset_expander(
        st.session_state.df_traditional,
        "Traditional",
        preview_rows=1000,
        unique_mentions=traditional_unique_mentions,
    )

    render_dataset_expander(
        st.session_state.df_social,
        "Social",
        preview_rows=200,
        unique_mentions=None,
    )

    render_dataset_expander(
        st.session_state.df_dupes,
        "Deleted Duplicates",
        preview_rows=200,
        unique_mentions=None,
    )
else:
    with st.form("basic_cleaning_form"):
        st.subheader("Cleaning Options")

        merge_online = st.checkbox(
            "Merge 'blogs' and 'press releases' into 'Online'",
            value=True,
            help="Combine ONLINE NEWS, PRESS RELEASE, and BLOGS into ONLINE.",
        )

        drop_dupes = st.checkbox(
            "Drop duplicates",
            value=True,
            help="After media types are normalized, remove non-broadcast duplicates by URL and by Type + Outlet + Headline, then run separate broadcast duplicate checks using outlet, media type, date, time proximity, and snippet similarity.",
        )

        submitted = st.form_submit_button("Run Basic Cleaning", type="primary")

    if submitted:
        with st.spinner("Running cleaning, Effective Reach, and story grouping..."):
            start_time = time.time()
            source_df = st.session_state.df_traditional_pre_standard.copy()
            cleaning_results = run_standard_cleaning(
                df=source_df,
                merge_online=merge_online,
                drop_dupes=drop_dupes,
                add_flags=True,
            )
            df_traditional = apply_effective_reach_traditional(cleaning_results["df_traditional"])
            df_social = apply_effective_reach_social(cleaning_results["df_social"])
            df_dupes = cleaning_results["df_dupes"]

            df_ai_grouped = cluster_by_media_type(
                df=df_traditional.copy(),
                similarity_threshold=0.935,
                max_batch_size=1800,
            )
            df_ai_grouped = mark_prime_examples(df_ai_grouped)
            df_ai_unique = build_unique_story_table(df_ai_grouped)

            # Canonical grouped traditional dataset
            df_traditional = df_ai_grouped.copy()

            st.session_state.df_traditional = df_traditional
            st.session_state.df_social = df_social
            st.session_state.df_dupes = df_dupes
            st.session_state.df_ai_grouped = df_ai_grouped
            st.session_state.df_ai_unique = df_ai_unique
            st.session_state.standard_step = True

            elapsed = time.time() - start_time
            st.session_state.basic_cleaning_elapsed = elapsed

            st.rerun()
