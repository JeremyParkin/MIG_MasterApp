# 2-Basic_Cleaning.py

from __future__ import annotations

import time
import warnings
import pandas as pd
import streamlit as st

from processing.standard_cleaning import run_standard_cleaning
from processing.standard_cleaning import SOURCE_ROW_COL
from processing.effective_reach import (
    apply_effective_reach_traditional,
    apply_effective_reach_social,
)

from processing.story_grouping import (
    cluster_by_media_type_with_timings,
    cluster_by_media_type,
    build_unique_story_table,
    mark_prime_examples_with_timings,
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
    if "basic_cleaning_timings" not in st.session_state:
        st.session_state.basic_cleaning_timings = pd.DataFrame()
    if "basic_cleaning_validation" not in st.session_state:
        st.session_state.basic_cleaning_validation = {}


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
    st.session_state.basic_cleaning_timings = pd.DataFrame()
    st.session_state.basic_cleaning_validation = {}


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

    timings_df = st.session_state.get("basic_cleaning_timings", pd.DataFrame())
    if isinstance(timings_df, pd.DataFrame) and not timings_df.empty:
        with st.expander("Temporary performance timings", expanded=False):
            display_timings = timings_df.copy()
            display_timings["Seconds"] = pd.to_numeric(display_timings["Seconds"], errors="coerce").fillna(0.0)
            display_timings["Share"] = (
                display_timings["Seconds"] / display_timings["Seconds"].sum()
            ).fillna(0.0).map(lambda x: f"{x:.1%}")
            display_timings["Seconds"] = display_timings["Seconds"].map(lambda x: f"{x:.2f}")
            st.dataframe(display_timings, use_container_width=True, hide_index=True)

    validation = st.session_state.get("basic_cleaning_validation", {})
    if validation:
        with st.expander("Temporary optimization validation", expanded=False):
            rows = []
            dedupe_validation = validation.get("dedupe", {})
            if dedupe_validation.get("Enabled"):
                rows.append({
                    "Check": "Broadcast dedupe matches legacy",
                    "Result": "Yes" if dedupe_validation.get("Broadcast Matches Legacy") else "No",
                    "Legacy Seconds": dedupe_validation.get("Broadcast Legacy Seconds", ""),
                })
            prime_validation = validation.get("prime_examples", {})
            if prime_validation.get("Enabled"):
                rows.append({
                    "Check": "Prime example selection matches legacy",
                    "Result": "Yes" if prime_validation.get("Matches Legacy") else "No",
                    "Legacy Seconds": prime_validation.get("Legacy Seconds", ""),
                })
            cluster_validation = validation.get("clustering", {})
            if cluster_validation.get("Enabled"):
                rows.append({
                    "Check": "Story clustering matches legacy",
                    "Result": "Yes" if cluster_validation.get("Matches Legacy") else "No",
                    "Legacy Seconds": cluster_validation.get("Legacy Seconds", ""),
                })
            broadcast_cluster = validation.get("broadcast_cluster_experiment", {})
            if broadcast_cluster.get("Enabled"):
                rows.append({
                    "Check": "Broadcast duplicate pairs share later story cluster",
                    "Result": f"{broadcast_cluster.get('Match Rate', 0.0):.1%}",
                    "Legacy Seconds": "",
                })
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            else:
                st.caption("No validation checks were run on the last cleaning pass.")

    traditional_unique_mentions = None
    if not st.session_state.df_ai_unique.empty and "Group Count" in st.session_state.df_ai_unique.columns:
        traditional_unique_mentions = len(st.session_state.df_ai_unique)

    render_dataset_expander(
        st.session_state.df_traditional,
        "Traditional / Online / Broadcast",
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

    # st.write(
    #     st.session_state.df_ai_unique[["Group ID", "Prime Example", "Headline", "Outlet", "Group Count"]].head(20)
    # )
    #
    # prime_check = (
    #     st.session_state.df_traditional.groupby("Group ID")["Prime Example"].sum().value_counts()
    # )
    # st.write(prime_check)
    #
    # st.write(
    #     st.session_state.df_ai_unique[["Group ID", "Prime Example", "Headline", "Outlet", "Group Count"]].head(20)
    # )


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
            help="Remove likely duplicates after cleaning and media-type splitting.",
        )

        validate_optimizations = st.checkbox(
            "Temporarily validate optimized dedupe/grouping against legacy logic",
            value=False,
            help="Slower, but compares the new broadcast dedupe and prime example logic against the legacy implementations.",
        )

        submitted = st.form_submit_button("Run Basic Cleaning", type="primary")

    if submitted:
        with st.spinner("Running cleaning, Effective Reach, and story grouping..."):
            start_time = time.time()
            source_df = st.session_state.df_traditional_pre_standard.copy()
            performance_rows: list[dict[str, float | str]] = []

            def add_perf_row(step_name: str, started_at: float) -> None:
                performance_rows.append({
                    "Step": step_name,
                    "Seconds": round(time.perf_counter() - started_at, 4),
                })

            overall_perf_start = time.perf_counter()
            perf_start = time.perf_counter()
            cleaning_results = run_standard_cleaning(
                df=source_df,
                merge_online=merge_online,
                drop_dupes=drop_dupes,
                add_flags=True,
                validate_optimizations=validate_optimizations,
            )
            add_perf_row("Standard cleaning total", perf_start)
            for row in cleaning_results.get("timings", []):
                performance_rows.append({
                    "Step": f"  - {row.get('Step', '')}",
                    "Seconds": row.get("Seconds", 0.0),
                })

            perf_start = time.perf_counter()
            df_traditional = apply_effective_reach_traditional(cleaning_results["df_traditional"])
            add_perf_row("Traditional effective reach", perf_start)

            perf_start = time.perf_counter()
            df_social = apply_effective_reach_social(cleaning_results["df_social"])
            add_perf_row("Social effective reach", perf_start)
            df_dupes = cleaning_results["df_dupes"]

            perf_start = time.perf_counter()
            df_ai_grouped, clustering_timings, clustering_validation = cluster_by_media_type_with_timings(
                df=df_traditional.copy(),
                similarity_threshold=0.935,
                max_batch_size=1800,
                validate=validate_optimizations,
            )
            add_perf_row("Cluster by media type", perf_start)
            for row in clustering_timings:
                performance_rows.append({
                    "Step": f"  - {row.get('Step', '')}",
                    "Seconds": row.get("Seconds", 0.0),
                })

            perf_start = time.perf_counter()
            df_ai_grouped, prime_timings, prime_validation = mark_prime_examples_with_timings(
                df_ai_grouped,
                validate=validate_optimizations,
            )
            add_perf_row("Mark prime examples", perf_start)
            for row in prime_timings:
                performance_rows.append({
                    "Step": f"  - {row.get('Step', '')}",
                    "Seconds": row.get("Seconds", 0.0),
                })

            perf_start = time.perf_counter()
            df_ai_unique = build_unique_story_table(df_ai_grouped)
            add_perf_row("Build unique story table", perf_start)

            broadcast_cluster_validation = {"Enabled": validate_optimizations}
            if validate_optimizations:
                experiment = cleaning_results.get("validation", {}).get("dedupe", {})
                pair_list = experiment.get("Broadcast Duplicate Pairs", [])
                pre_dedupe_broadcast = experiment.get("Pre-Dedupe Broadcast Rows")
                if isinstance(pre_dedupe_broadcast, pd.DataFrame) and not pre_dedupe_broadcast.empty and pair_list:
                    experiment_clustered = cluster_by_media_type(
                        pre_dedupe_broadcast.copy(),
                        similarity_threshold=0.935,
                        max_batch_size=1800,
                    )
                    source_to_group = dict(
                        zip(
                            experiment_clustered.get(SOURCE_ROW_COL, pd.Series(dtype="int64")),
                            experiment_clustered.get("Group ID", pd.Series(dtype="int64")),
                        )
                    )
                    comparable = [
                        (dup_id, keep_id)
                        for dup_id, keep_id in pair_list
                        if dup_id in source_to_group and keep_id in source_to_group
                    ]
                    if comparable:
                        matches = sum(1 for dup_id, keep_id in comparable if source_to_group[dup_id] == source_to_group[keep_id])
                        broadcast_cluster_validation["Compared Pairs"] = len(comparable)
                        broadcast_cluster_validation["Matches"] = matches
                        broadcast_cluster_validation["Match Rate"] = matches / len(comparable)
                    else:
                        broadcast_cluster_validation["Compared Pairs"] = 0
                        broadcast_cluster_validation["Matches"] = 0
                        broadcast_cluster_validation["Match Rate"] = 0.0

            # Canonical grouped traditional dataset
            df_traditional = df_ai_grouped.copy()

            for frame in [df_traditional, df_social, df_dupes, df_ai_grouped, df_ai_unique]:
                if isinstance(frame, pd.DataFrame) and SOURCE_ROW_COL in frame.columns:
                    frame.drop(columns=[SOURCE_ROW_COL], inplace=True, errors="ignore")

            st.session_state.df_traditional = df_traditional
            st.session_state.df_social = df_social
            st.session_state.df_dupes = df_dupes
            st.session_state.df_ai_grouped = df_ai_grouped
            st.session_state.df_ai_unique = df_ai_unique
            st.session_state.standard_step = True

            elapsed = time.time() - start_time
            st.session_state.basic_cleaning_elapsed = elapsed
            add_perf_row("Overall basic cleaning", overall_perf_start)
            st.session_state.basic_cleaning_timings = pd.DataFrame(performance_rows)
            st.session_state.basic_cleaning_validation = {
                "dedupe": cleaning_results.get("validation", {}).get("dedupe", {}),
                "clustering": clustering_validation,
                "prime_examples": prime_validation,
                "broadcast_cluster_experiment": broadcast_cluster_validation,
            }

            st.rerun()
