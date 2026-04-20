from __future__ import annotations

import html
import importlib
import re
from urllib.parse import urlparse
import warnings

import pandas as pd
import streamlit as st

from processing.analysis_context import (
    apply_session_coverage_flag_policy,
    build_analysis_context_text,
    build_dataset_scope_cache_key,
    get_outlet_insight_coverage_flag_exclusions,
    init_analysis_context_state,
)
import processing.outlet_insights as outlet_insights
from ui.insight_blocks import build_linked_example_blocks_html
from utils.formatting import NUMERIC_FORMAT_DICT

warnings.filterwarnings("ignore")

outlet_insights = importlib.reload(outlet_insights)
DEFAULT_OUTLET_SUMMARY_MODEL = outlet_insights.DEFAULT_OUTLET_SUMMARY_MODEL
apply_outlet_rollup_map = outlet_insights.apply_outlet_rollup_map
build_outlet_headline_table = outlet_insights.build_outlet_headline_table
build_outlet_metrics = outlet_insights.build_outlet_metrics
build_outlet_rollup_preview = outlet_insights.build_outlet_rollup_preview
build_outlet_cleanup_clusters = outlet_insights.build_outlet_cleanup_clusters
build_rollup_suggestions = outlet_insights.build_rollup_suggestions
build_outlet_top_authors = outlet_insights.build_outlet_top_authors
build_outlet_variant_candidates = outlet_insights.build_outlet_variant_candidates
generate_outlet_summary = outlet_insights.generate_outlet_summary
init_outlet_workflow_state = outlet_insights.init_outlet_workflow_state
remove_outlet_rollup_map = outlet_insights.remove_outlet_rollup_map

METRIC_FIELD_MAP = {
    "Mentions": "Mention_Total",
    "Impressions": "Impressions",
    "Effective Reach": "Effective_Reach",
}


def format_compact_integer(num: float | int) -> str:
    try:
        n = float(num)
    except Exception:
        return str(num)

    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.0f}B"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.0f}M"
    if n >= 1_000:
        return f"{n / 1_000:.0f}K"
    return str(int(n)) if n.is_integer() else str(round(n))


def extract_domain(value: object) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    if not re.match(r"^[a-z]+://", raw, flags=re.IGNORECASE):
        raw = f"https://{raw}"
    try:
        host = (urlparse(raw).netloc or "").lower()
    except Exception:
        return ""
    if host.startswith("www."):
        host = host[4:]
    if "tveyes" in host or "tvey" in host:
        return ""
    return host


def render_outlets_page() -> None:
    st.title("Outlets")
    st.caption("Clean up outlet names, curate the final outlet list, and generate outlet insights.")

    if not st.session_state.get("standard_step", False):
        st.error("Please complete Basic Cleaning before trying this step.")
        st.stop()

    if len(st.session_state.get("df_traditional", [])) == 0:
        st.error("Traditional / online / broadcast coverage is missing. Please complete Basic Cleaning again.")
        st.stop()

    init_outlet_workflow_state(st.session_state)
    init_analysis_context_state(st.session_state)
    st.session_state.setdefault("outlet_selection_checked_outlets", [])
    st.session_state.setdefault("outlet_selection_editor_version", 0)

    def rebuild_outlet_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        df_traditional = st.session_state.df_traditional.copy()
        filtered_df = apply_session_coverage_flag_policy(
            df_traditional,
            st.session_state,
            get_outlet_insight_coverage_flag_exclusions(st.session_state),
        )
        outlet_map = st.session_state.get("outlet_rollup_map", {})
        cache_key = (
            "outlet_data_cache",
            len(filtered_df),
            tuple(filtered_df.columns.tolist()),
            tuple(sorted((str(k), str(v)) for k, v in outlet_map.items())),
            tuple(get_outlet_insight_coverage_flag_exclusions(st.session_state)),
            build_dataset_scope_cache_key(st.session_state),
        )
        cached = st.session_state.get("outlet_data_cache")
        if cached and cached.get("key") == cache_key:
            return cached["metrics_df"], cached["story_df"], cached["variants_df"]

        metrics_df, story_df = build_outlet_metrics(filtered_df, outlet_rollup_map=outlet_map)
        variants_df = build_outlet_variant_candidates(filtered_df, outlet_rollup_map=outlet_map)
        st.session_state.outlet_data_cache = {
            "key": cache_key,
            "metrics_df": metrics_df,
            "story_df": story_df,
            "variants_df": variants_df,
        }
        return metrics_df, story_df, variants_df

    def rebuild_outlet_cleanup_data() -> tuple[pd.DataFrame, list[dict[str, object]], pd.DataFrame]:
        df_traditional = st.session_state.df_traditional.copy()
        outlet_map = st.session_state.get("outlet_rollup_map", {})
        cache_key = (
            "outlet_cleanup_cache",
            5,
            len(df_traditional),
            tuple(df_traditional.columns.tolist()),
            tuple(sorted((str(k), str(v)) for k, v in outlet_map.items())),
        )
        cached = st.session_state.get("outlet_cleanup_cache")
        if cached and cached.get("key") == cache_key:
            return cached["rollup_preview"], cached["cleanup_clusters"], cached["current_entities"]

        rollup_preview = build_outlet_rollup_preview(df_traditional, outlet_map)
        cleanup_clusters = build_outlet_cleanup_clusters(df_traditional, outlet_map)
        if rollup_preview.empty:
            current_entities = pd.DataFrame(
                columns=["Outlet", "Country", "Domain", "Media Type", "Source_Count", "Mentions", "Impressions", "Effective Reach"]
            )
        else:
            current_entities = (
                rollup_preview.groupby("Canonical Outlet", as_index=False)
                .agg(
                    Source_Count=("Outlet", "nunique"),
                    Mentions=("Mentions", "sum"),
                    Impressions=("Impressions", "sum"),
                    Effective_Reach=("Effective_Reach", "sum"),
                )
                .rename(columns={"Canonical Outlet": "Outlet", "Effective_Reach": "Effective Reach"})
            )

            domain_working = df_traditional.copy()
            domain_working["Outlet"] = domain_working.get("Outlet", "").fillna("").astype(str).str.strip()
            domain_working["URL"] = domain_working.get("URL", "").fillna("").astype(str).str.strip()
            domain_working["Canonical Outlet"] = domain_working["Outlet"].map(
                lambda name: outlet_map.get(str(name).strip(), str(name).strip())
            )
            domain_working["_domain"] = domain_working["URL"].apply(extract_domain)
            domain_working = domain_working[domain_working["_domain"] != ""].copy()
            if not domain_working.empty:
                top_domain = (
                    domain_working.sort_values(
                        ["Mentions", "Impressions", "Date"],
                        ascending=[False, False, False],
                        na_position="last",
                    )
                    .drop_duplicates(subset=["Canonical Outlet"], keep="first")
                )[["Canonical Outlet", "_domain"]].rename(
                    columns={"Canonical Outlet": "Outlet", "_domain": "Domain"}
                )
                current_entities = current_entities.merge(top_domain, on="Outlet", how="left")
            else:
                current_entities["Domain"] = ""

            type_working = df_traditional.copy()
            type_working["Outlet"] = type_working.get("Outlet", "").fillna("").astype(str).str.strip()
            type_working["Type"] = type_working.get("Type", "").fillna("").astype(str).str.strip()
            type_working["Canonical Outlet"] = type_working["Outlet"].map(
                lambda name: outlet_map.get(str(name).strip(), str(name).strip())
            )
            type_rollup = (
                type_working[type_working["Type"] != ""]
                .groupby(["Canonical Outlet", "Type"], as_index=False)["Mentions"]
                .sum()
                .sort_values(["Canonical Outlet", "Mentions", "Type"], ascending=[True, False, True])
            )
            if not type_rollup.empty:
                top_type = (
                    type_rollup.drop_duplicates(subset=["Canonical Outlet"], keep="first")
                    .rename(columns={"Canonical Outlet": "Outlet", "Type": "Media Type"})
                )[["Outlet", "Media Type"]]
                current_entities = current_entities.merge(top_type, on="Outlet", how="left")
            else:
                current_entities["Media Type"] = ""

            country_working = df_traditional.copy()
            country_working["Outlet"] = country_working.get("Outlet", "").fillna("").astype(str).str.strip()
            country_working["Country"] = country_working.get("Country", "").fillna("").astype(str).str.strip()
            country_working["Canonical Outlet"] = country_working["Outlet"].map(
                lambda name: outlet_map.get(str(name).strip(), str(name).strip())
            )
            country_rollup = (
                country_working[country_working["Country"] != ""]
                .groupby(["Canonical Outlet", "Country"], as_index=False)["Mentions"]
                .sum()
                .sort_values(["Canonical Outlet", "Mentions", "Country"], ascending=[True, False, True])
            )
            if not country_rollup.empty:
                top_country = (
                    country_rollup.drop_duplicates(subset=["Canonical Outlet"], keep="first")
                    .rename(columns={"Canonical Outlet": "Outlet"})
                )[["Outlet", "Country"]]
                current_entities = current_entities.merge(top_country, on="Outlet", how="left")
            else:
                current_entities["Country"] = ""

            current_entities["Country"] = current_entities["Country"].fillna("")
            current_entities["Domain"] = current_entities["Domain"].fillna("")
            current_entities["Media Type"] = current_entities["Media Type"].fillna("")
            current_entities = current_entities[[
                "Outlet",
                "Country",
                "Domain",
                "Media Type",
                "Source_Count",
                "Mentions",
                "Impressions",
                "Effective Reach",
            ]].sort_values(["Outlet"], ascending=[True]).reset_index(drop=True)

        st.session_state.outlet_cleanup_cache = {
            "key": cache_key,
            "rollup_preview": rollup_preview,
            "cleanup_clusters": cleanup_clusters,
            "current_entities": current_entities,
        }
        return rollup_preview, cleanup_clusters, current_entities

    def build_story_examples_html(
        df: pd.DataFrame,
        show_author: bool = True,
        show_date: bool = False,
        show_media_type: bool = True,
        show_mentions: bool = True,
        show_impressions: bool = True,
        show_effective_reach: bool = True,
    ) -> str:
        if df.empty:
            return ""

        items = []
        for _, row in df.iterrows():
            items.append(
                {
                    "headline": row.get("Headline", ""),
                    "url": row.get("Representative URL", ""),
                    "outlet": row.get("Author", ""),
                    "date": row.get("Date", ""),
                    "example_type": row.get("Type", ""),
                    "mentions": int(pd.to_numeric(pd.Series([row.get("Story Mentions", 0)]), errors="coerce").fillna(0).iloc[0]),
                    "impressions": int(pd.to_numeric(pd.Series([row.get("Story Impressions", 0)]), errors="coerce").fillna(0).iloc[0]),
                    "effective_reach": int(pd.to_numeric(pd.Series([row.get("Story Effective Reach", 0)]), errors="coerce").fillna(0).iloc[0]),
                }
            )

        return build_linked_example_blocks_html(
            items,
            show_outlet=show_author,
            show_date=show_date,
            show_media_type=show_media_type,
            show_mentions=show_mentions,
            show_impressions=show_impressions,
            show_effective_reach=show_effective_reach,
        )

    def build_report_html(
        shortlist_df: pd.DataFrame,
        show_author: bool,
        show_date: bool,
        show_media_type: bool,
        show_mentions: bool,
        show_impressions: bool,
        show_effective_reach: bool,
        show_headline_examples: bool,
    ) -> str:
        blocks = []
        for _, row in shortlist_df.iterrows():
            outlet = str(row.get("Outlet", "") or "").strip()
            top_types = str(row.get("Top_Types", "") or "").strip()
            themes = " ".join(str(row.get("Coverage Themes", "") or "").split())

            header = f"<strong>{html.escape(outlet)}</strong>"
            if top_types:
                header += f' <span style="opacity:0.82;">|</span> <span style="font-style:italic; opacity:0.92;">{html.escape(top_types)}</span>'

            examples_html = ""
            if show_headline_examples and outlet:
                story_df = build_outlet_headline_table(get_story_rows(), outlet, limit=5)
                example_items = build_story_examples_html(
                    story_df,
                    show_author=show_author,
                    show_date=show_date,
                    show_media_type=show_media_type,
                    show_mentions=show_mentions,
                    show_impressions=show_impressions,
                    show_effective_reach=show_effective_reach,
                )
                if example_items:
                    examples_html = (
                        '<div style="margin-top:0.72rem;">'
                        f"{example_items}"
                        "</div>"
                    )

            block = (
                '<div style="margin-bottom:1.35rem;">'
                f'<div style="font-size:1.08rem; font-weight:700; margin-bottom:0.2rem;">{header}</div>'
                f'<div style="line-height:1.55; margin-bottom:0.28rem;">{html.escape(themes)}</div>'
                f"{examples_html}"
                "</div>"
            )
            blocks.append(block)

        return f'<div style="display:block;">{"".join(blocks)}</div>' if blocks else ""

    outlet_runtime_cache: dict[str, pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]] = {}

    def get_outlet_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        cached = outlet_runtime_cache.get("outlet_data")
        if cached is None:
            cached = rebuild_outlet_data()
            outlet_runtime_cache["outlet_data"] = cached
        return cached  # type: ignore[return-value]

    def get_metrics_df() -> pd.DataFrame:
        return get_outlet_data()[0]

    def get_story_rows() -> pd.DataFrame:
        return get_outlet_data()[1]

    def get_variants_df() -> pd.DataFrame:
        return get_outlet_data()[2]

    st.markdown(
        """
        <style>
        .outlets-step-note {
            margin: 0.15rem 0 1rem 0;
            color: rgba(250, 250, 250, 0.72);
            font-size: 0.95rem;
        }
        div[data-testid="stButton"] button {
            min-height: 2.8rem;
            font-weight: 600;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    step1, step2, step3 = st.columns(3, gap="small")
    with step1:
        if st.button(
            "1. Cleanup",
            key="outlets_step_cleanup",
            use_container_width=True,
            type="primary" if st.session_state.outlets_section == "Cleanup" else "secondary",
        ):
            st.session_state.outlets_section = "Cleanup"
            st.rerun()
    with step2:
        if st.button(
            "2. Selection",
            key="outlets_step_selection",
            use_container_width=True,
            type="primary" if st.session_state.outlets_section == "Selection" else "secondary",
        ):
            st.session_state.outlets_section = "Selection"
            st.rerun()
    with step3:
        if st.button(
            "3. Insights",
            key="outlets_step_insights",
            use_container_width=True,
            type="primary" if st.session_state.outlets_section == "Insights" else "secondary",
        ):
            st.session_state.outlets_section = "Insights"
            st.rerun()

    st.markdown(
        '<div class="outlets-step-note">Work left to right: clean outlet names, save the final outlet list, then review chart/table and narrative outputs.</div>',
        unsafe_allow_html=True,
    )

    def get_ranked_outlet_metrics() -> pd.DataFrame:
        rank_by = st.session_state.get("outlets_rank_by", "Mentions")
        metric_col = METRIC_FIELD_MAP.get(rank_by, "Mention_Total")
        sort_cols = [metric_col, "Impressions", "Mention_Total", "Unique_Mentions"]
        metrics_df = get_metrics_df()
        return metrics_df.sort_values(sort_cols, ascending=False).reset_index(drop=True)

    def render_cleanup_section() -> None:
        def render_cleanup_section_legacy() -> None:
            st.session_state.outlets_section = "Cleanup"
            st.subheader("Outlet Cleanup")
            st.info("Cleanup does not overwrite raw outlet names in your cleaned data. It creates a reporting rollup map used by this Outlets workflow and downstream exports.")

            rank_by = st.radio(
                "Rank outlets by",
                ["Mentions", "Impressions", "Effective Reach"],
                horizontal=True,
                key="outlets_rank_by_cleanup_legacy",
            )
            st.session_state.outlets_rank_by = rank_by
            ranked = get_ranked_outlet_metrics()
            raw_rollup_preview = build_outlet_rollup_preview(st.session_state.df_traditional.copy(), st.session_state.get("outlet_rollup_map", {}))
            raw_outlet_options = raw_rollup_preview["Outlet"].tolist()
            suggestion_df = build_rollup_suggestions(st.session_state.df_traditional.copy(), st.session_state.get("outlet_rollup_map", {}))

            stats1, stats2, stats3 = st.columns(3)
            with stats1:
                st.metric("Canonical outlets", f"{len(ranked):,}")
            with stats2:
                st.metric("Mapped source outlets", f"{int(raw_rollup_preview['Rollup Applied'].sum()):,}" if not raw_rollup_preview.empty else "0")
            with stats3:
                st.metric("Rollup rules applied", f"{len(st.session_state.get('outlet_rollup_map', {})):,}")

            st.write("**Suggested rollups**")
            if suggestion_df.empty:
                st.info("No obvious network rollups detected from the current outlet list.")
            else:
                for idx, row in suggestion_df.head(8).iterrows():
                    canonical_name = str(row.get("Suggested Rollup", "") or "").strip()
                    source_outlets = [part.strip() for part in str(row.get("Source Outlets", "") or "").split("|") if part.strip()]
                    source_count = int(row.get("Source Outlet Count", 0) or 0)
                    with st.container(border=True):
                        top_row1, top_row2 = st.columns([3, 1], gap="small")
                        with top_row1:
                            st.write(f"**{canonical_name}**")
                            st.caption(
                                f"Showing {min(len(source_outlets), 8):,} of {source_count:,} source outlets. "
                                f"This creates a reporting rollup only; it does not overwrite raw outlet names."
                            )
                            st.caption(str(row.get("Source Outlets", "")))
                        with top_row2:
                            st.metric("Source outlets", f"{source_count:,}")
                        metric1, metric2, metric3 = st.columns(3)
                        with metric1:
                            st.metric("Mentions", f"{int(row.get('Mentions', 0)):,}")
                        with metric2:
                            st.metric("Impressions", f"{int(row.get('Impressions', 0)):,}")
                        with metric3:
                            st.metric("Effective Reach", f"{int(row.get('Effective Reach', 0)):,}")
                        if st.button(f"Apply rollup to {canonical_name}", key=f"apply_outlet_rollup_{idx}", type="primary"):
                            apply_outlet_rollup_map(st.session_state, source_outlets, canonical_name)
                            st.rerun()

            st.divider()
            st.write("**Bulk rule**")
            st.caption("Use this when many outlets share a pattern and should roll up to the same reporting record.")
            with st.form("outlet_cleanup_bulk_rule_form"):
                rule_col1, rule_col2, rule_col3 = st.columns([1, 1.25, 1.25], gap="small")
                with rule_col1:
                    rule_mode = st.selectbox("Match mode", ["Contains", "Starts with", "Regex"], key="outlet_cleanup_rule_mode")
                with rule_col2:
                    rule_pattern = st.text_input("Pattern", key="outlet_cleanup_rule_pattern")
                with rule_col3:
                    rule_target = st.text_input("Canonical outlet name", key="outlet_cleanup_rule_target")
                apply_bulk_rule = st.form_submit_button("Apply bulk rule", type="primary")

            matches: list[str] = []
            if raw_outlet_options and st.session_state.get("outlet_cleanup_rule_pattern", "").strip():
                pattern = st.session_state.get("outlet_cleanup_rule_pattern", "").strip()
                mode = st.session_state.get("outlet_cleanup_rule_mode", "Contains")
                for outlet in raw_outlet_options:
                    outlet_text = str(outlet)
                    if mode == "Contains" and pattern.lower() in outlet_text.lower():
                        matches.append(outlet)
                    elif mode == "Starts with" and outlet_text.lower().startswith(pattern.lower()):
                        matches.append(outlet)
                    elif mode == "Regex":
                        try:
                            if re.search(pattern, outlet_text, flags=re.IGNORECASE):
                                matches.append(outlet)
                        except Exception:
                            matches = []
                            break

            if st.session_state.get("outlet_cleanup_rule_pattern", "").strip():
                st.caption(f"Bulk rule preview: {len(matches):,} source outlet(s) match.")
                if matches:
                    st.caption(" | ".join(matches[:12]))

            if apply_bulk_rule:
                target = st.session_state.get("outlet_cleanup_rule_target", "").strip()
                if not target or not matches:
                    st.warning("Enter a canonical outlet name and a pattern that matches at least one source outlet.")
                else:
                    apply_outlet_rollup_map(st.session_state, matches, target)
                    st.rerun()

            st.divider()
            st.write("**Manual exceptions**")
            st.caption("Use this for one-off source outlets that weren’t covered by a suggested rollup or bulk rule.")
            with st.form("outlet_cleanup_manual_form"):
                manual_col1, manual_col2 = st.columns([1.4, 1], gap="small")
                with manual_col1:
                    manual_sources = st.multiselect(
                        "Source outlets to roll up",
                        options=raw_outlet_options,
                        key="outlet_cleanup_manual_selection",
                    )
                with manual_col2:
                    manual_target = st.text_input(
                        "Canonical outlet name",
                        key="outlet_cleanup_manual_target",
                    )
                apply_manual = st.form_submit_button("Apply manual rollup", type="primary")

            if apply_manual:
                if not manual_sources or not manual_target.strip():
                    st.warning("Choose one or more source outlets and enter a canonical outlet name.")
                else:
                    apply_outlet_rollup_map(st.session_state, manual_sources, manual_target.strip())
                    st.rerun()

            st.divider()
            preview_col1, preview_col2 = st.columns([1.15, 0.85], gap="large")
            with preview_col1:
                st.write("**Rollup preview**")
                st.caption("This shows how source outlet names will roll up inside the Outlets workflow and export outputs.")
                preview_df = raw_rollup_preview[["Outlet", "Canonical Outlet", "Rollup Applied", "Mentions", "Impressions", "Effective_Reach"]].head(30).copy()
                preview_df = preview_df.rename(columns={"Effective_Reach": "Effective Reach"})
                st.dataframe(preview_df.style.format(NUMERIC_FORMAT_DICT, na_rep=" "), use_container_width=True, hide_index=True)
            with preview_col2:
                with st.expander("Possible naming variants", expanded=False):
                    st.caption("Optional reference only. These are simple normalized-name clusters that might suggest additional cleanup opportunities.")
                    variants_df = get_variants_df()
                    if variants_df.empty:
                        st.info("No obvious duplicate outlet variants were detected from simple name normalization.")
                    else:
                        st.dataframe(variants_df.style.format(NUMERIC_FORMAT_DICT, na_rep=" "), use_container_width=True, hide_index=True)

            current_mappings = st.session_state.get("outlet_rollup_map", {})
            if current_mappings:
                st.divider()
                with st.expander("Current rollup mappings", expanded=False):
                    mapping_df = pd.DataFrame(
                        [{"Source Outlet": source, "Canonical Outlet": target} for source, target in current_mappings.items()]
                    ).sort_values(["Canonical Outlet", "Source Outlet"]).reset_index(drop=True)
                    mapping_df["Remove"] = False
                    mapping_editor = st.data_editor(
                        mapping_df,
                        use_container_width=True,
                        hide_index=True,
                        key="outlet_mapping_editor",
                        column_config={"Remove": st.column_config.CheckboxColumn("Remove", width="small")},
                    )
                    remove_rows = mapping_editor[mapping_editor["Remove"]].index.tolist()
                    if remove_rows:
                        remove_sources = mapping_df.iloc[remove_rows]["Source Outlet"].tolist()
                        remove_outlet_rollup_map(st.session_state, remove_sources)
                        st.rerun()

        st.session_state.outlets_section = "Cleanup"
        raw_rollup_preview, cleanup_clusters, current_entities = rebuild_outlet_cleanup_data()
        cleanup_clusters = sorted(
            cleanup_clusters,
            key=lambda row: (
                row.get("candidate_count", 0),
                row.get("mentions", 0),
                row.get("impressions", 0),
                row.get("effective_reach", 0),
            ),
            reverse=True,
        )

        stats1, stats2, stats3 = st.columns(3)
        with stats1:
            st.metric("Reporting outlets", f"{len(current_entities):,}")
        with stats2:
            st.metric("Source outlets mapped", f"{int(raw_rollup_preview['Rollup Applied'].sum()):,}" if not raw_rollup_preview.empty else "0")
        with stats3:
            st.metric("Mappings saved", f"{len(st.session_state.get('outlet_rollup_map', {})):,}")

        st.subheader("Suggested merge review")
        if not cleanup_clusters:
            st.success("No unresolved outlet merge clusters remain.")
        else:
            cluster_index = int(st.session_state.get("outlet_cleanup_cluster_index", 0) or 0)
            cluster_index = max(0, min(cluster_index, len(cleanup_clusters) - 1))
            st.session_state.outlet_cleanup_cluster_index = cluster_index
            cluster = cleanup_clusters[cluster_index]

            anchor_col, cluster_nav1, cluster_nav2, cluster_nav3, cluster_nav4, cluster_nav5 = st.columns(
                [2.1, 0.8, 0.8, 0.8, 0.8, 1.45], gap="small"
            )
            with anchor_col:
                st.markdown(
                    f'<div style="font-size:2rem; font-weight:800; color:#d4a72c; line-height:1.1; margin:0.1rem 0 0.2rem 0;">'
                    f'{html.escape(str(cluster["suggested_master"]))}</div>',
                    unsafe_allow_html=True,
                )

            with cluster_nav1:
                if st.button("", key="outlet_cleanup_first", use_container_width=True, disabled=cluster_index <= 0, icon=":material/first_page:", help="First cluster"):
                    st.session_state.outlet_cleanup_cluster_index = 0
                    st.rerun()
            with cluster_nav2:
                if st.button("", key="outlet_cleanup_prev", use_container_width=True, disabled=cluster_index <= 0, icon=":material/skip_previous:", help="Previous cluster"):
                    st.session_state.outlet_cleanup_cluster_index = max(cluster_index - 1, 0)
                    st.rerun()
            with cluster_nav3:
                if st.button("", key="outlet_cleanup_next", use_container_width=True, disabled=cluster_index >= len(cleanup_clusters) - 1, icon=":material/skip_next:", help="Next cluster"):
                    st.session_state.outlet_cleanup_cluster_index = min(cluster_index + 1, len(cleanup_clusters) - 1)
                    st.rerun()
            with cluster_nav4:
                if st.button("", key="outlet_cleanup_last", use_container_width=True, disabled=cluster_index >= len(cleanup_clusters) - 1, icon=":material/last_page:", help="Last cluster"):
                    st.session_state.outlet_cleanup_cluster_index = len(cleanup_clusters) - 1
                    st.rerun()
            with cluster_nav5:
                st.caption(f"Reviewing cluster {cluster_index + 1} of {len(cleanup_clusters)}")

            selected_candidates_map = dict(st.session_state.get("outlet_cleanup_selected_candidates", {}))
            default_selected = selected_candidates_map.get(
                cluster["cluster_id"],
                [row["Outlet"] for row in cluster["candidates"]],
            )

            candidate_df = pd.DataFrame(cluster["candidates"]).copy()
            for col in ["Media Type", "Country", "Domain", "Mentions", "Impressions", "Effective Reach"]:
                if col not in candidate_df.columns:
                    candidate_df[col] = "" if col in {"Media Type", "Country", "Domain"} else 0
            candidate_sort_cols = ["Mentions", "Impressions", "Effective Reach", "Outlet"]
            candidate_df = candidate_df.sort_values(candidate_sort_cols, ascending=[False, False, False, True]).reset_index(drop=True)
            candidate_df["Merge"] = candidate_df["Outlet"].isin(default_selected)

            with st.form(key=f"outlet_cleanup_cluster_form_{cluster['cluster_id']}"):
                candidate_editor = st.data_editor(
                    candidate_df[["Outlet", "Media Type", "Country", "Domain", "Mentions", "Impressions", "Effective Reach", "Merge"]],
                    use_container_width=True,
                    hide_index=True,
                    key=f"outlet_cleanup_cluster_editor_{cluster['cluster_id']}",
                    column_config={
                        "Media Type": st.column_config.Column("Media Type", width="small"),
                        "Country": st.column_config.Column("Country", width="small"),
                        "Domain": st.column_config.Column("Domain", width="medium"),
                        "Mentions": st.column_config.NumberColumn("Mentions", width="small", format="%d"),
                        "Impressions": st.column_config.NumberColumn("Impressions", width="small", format="%,d"),
                        "Effective Reach": st.column_config.NumberColumn("Effective Reach", width="small", format="%,d"),
                        "Merge": st.column_config.CheckboxColumn("Merge", width="small"),
                    },
                )
                selected_sources = candidate_editor.loc[candidate_editor["Merge"], "Outlet"].astype(str).tolist()
                master_col1, master_col2 = st.columns([1, 1], gap="small")
                master_options = list(dict.fromkeys([cluster["suggested_master"]] + selected_sources)) if selected_sources else [cluster["suggested_master"]]
                with master_col1:
                    master_selection = st.selectbox(
                        "Master name",
                        options=master_options,
                        index=0,
                        key=f"outlet_cleanup_master_choice_{cluster['cluster_id']}",
                    )
                with master_col2:
                    master_override = st.text_input(
                        "Or write your own master",
                        value="",
                        key=f"outlet_cleanup_master_override_{cluster['cluster_id']}",
                    )

                action_col1, action_col2 = st.columns([1, 3], gap="small")
                with action_col1:
                    apply_merge = st.form_submit_button("Apply merge", type="primary", use_container_width=True)
                with action_col2:
                    st.caption("Check the outlets that belong together, then choose the master name from the checked set or write your own.")

            st.session_state.outlet_cleanup_selected_candidates[cluster["cluster_id"]] = selected_sources
            if apply_merge:
                canonical_target = str(master_override or master_selection or "").strip()
                if len(selected_sources) < 2:
                    st.warning("Check at least two outlets before applying a merge.")
                elif not canonical_target:
                    st.warning("Choose or enter a master outlet name.")
                else:
                    apply_outlet_rollup_map(st.session_state, selected_sources, canonical_target)
                    selected_candidates_map.pop(cluster["cluster_id"], None)
                    st.session_state.outlet_cleanup_selected_candidates = selected_candidates_map
                    st.session_state.outlet_cleanup_cluster_index = min(cluster_index + 1, max(len(cleanup_clusters) - 1, 0))
                    st.rerun()

        st.divider()
        with st.expander("Advanced merging", expanded=False):
            st.caption("Filter the current outlet/reporting-name list, review the matches, uncheck where needed, then assign a shared reporting name.")
            if st.session_state.pop("outlet_cleanup_advanced_filter_reset_pending", False):
                st.session_state.outlet_cleanup_advanced_filter = ""
            if st.session_state.pop("outlet_cleanup_advanced_override_reset_pending", False):
                st.session_state.outlet_cleanup_advanced_override = ""
            filter_col1, filter_col2 = st.columns([5, 1], gap="small")
            with filter_col1:
                advanced_filter = st.text_input(
                    "Filter outlet names",
                    key="outlet_cleanup_advanced_filter",
                    placeholder="Type part of an outlet name...",
                ).strip()
            previous_filter = str(st.session_state.get("outlet_cleanup_advanced_filter_last_seen", "") or "")
            current_override = str(st.session_state.get("outlet_cleanup_advanced_override", "") or "")
            if advanced_filter != previous_filter:
                if advanced_filter and (not current_override.strip() or current_override.strip() == previous_filter.strip()):
                    st.session_state.outlet_cleanup_advanced_override = advanced_filter
                st.session_state.outlet_cleanup_advanced_filter_last_seen = advanced_filter
            with filter_col2:
                st.write("")
                if st.button("Clear filter", key="outlet_cleanup_advanced_clear_filter", use_container_width=True):
                    st.session_state.outlet_cleanup_advanced_filter_reset_pending = True
                    st.rerun()

            canonical_to_sources = (
                raw_rollup_preview.groupby("Canonical Outlet")["Outlet"]
                .agg(lambda s: list(dict.fromkeys(str(x) for x in s)))
                .to_dict()
            )

            if advanced_filter:
                filtered_entities = current_entities[
                    current_entities["Outlet"].astype(str).str.contains(advanced_filter, case=False, na=False)
                ].copy()
            else:
                filtered_entities = current_entities.copy()

            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
            with metric_col1:
                st.metric("Outlets", f"{len(filtered_entities):,}")
            with metric_col2:
                st.metric("Mentions", f"{int(filtered_entities['Mentions'].sum()) if not filtered_entities.empty else 0:,}")
            with metric_col3:
                st.metric("Impressions", f"{int(filtered_entities['Impressions'].sum()) if not filtered_entities.empty else 0:,}")
            with metric_col4:
                st.metric("Effective Reach", f"{int(filtered_entities['Effective Reach'].sum()) if not filtered_entities.empty else 0:,}")

            if filtered_entities.empty:
                st.info("No current outlet/reporting names match that filter.")
            else:
                for col in ["Country", "Domain", "Media Type", "Mentions", "Impressions", "Effective Reach"]:
                    if col not in filtered_entities.columns:
                        filtered_entities[col] = "" if col in {"Country", "Domain", "Media Type"} else 0
                filtered_entities["Merge"] = bool(advanced_filter)
                advanced_slice_key = abs(hash(tuple(filtered_entities["Outlet"].astype(str).tolist())))

                with st.form("outlet_cleanup_advanced_form"):
                    advanced_editor = st.data_editor(
                        filtered_entities[["Outlet", "Country", "Domain", "Media Type", "Mentions", "Impressions", "Effective Reach", "Merge"]],
                        use_container_width=True,
                        hide_index=True,
                        key=f"outlet_cleanup_advanced_editor_{advanced_slice_key}",
                        column_config={
                            "Country": st.column_config.Column("Country", width="small"),
                            "Domain": st.column_config.Column("Domain", width="medium"),
                            "Media Type": st.column_config.Column("Media Type", width="small"),
                            "Mentions": st.column_config.NumberColumn("Mentions", width="small", format="%d"),
                            "Impressions": st.column_config.NumberColumn("Impressions", width="small", format="%,d"),
                            "Effective Reach": st.column_config.NumberColumn("Effective Reach", width="small", format="%,d"),
                            "Merge": st.column_config.CheckboxColumn("Merge", width="small"),
                        },
                    )
                    selected_entities = advanced_editor.loc[advanced_editor["Merge"], "Outlet"].astype(str).tolist()
                    adv_col1, adv_col2 = st.columns([1, 1], gap="small")
                    with adv_col1:
                        suggested_reporting_name = selected_entities[0] if selected_entities else ""
                        reporting_name_options = selected_entities if selected_entities else ([suggested_reporting_name] if suggested_reporting_name else [""])
                        advanced_master = st.selectbox(
                            "Reporting name",
                            options=reporting_name_options,
                            index=0,
                            key=f"outlet_cleanup_advanced_master_{advanced_slice_key}",
                        )
                    with adv_col2:
                        advanced_override = st.text_input(
                            "Or write your own reporting name",
                            key="outlet_cleanup_advanced_override",
                        )

                    apply_advanced = st.form_submit_button("Apply filtered merge", type="primary")

                if apply_advanced:
                    reporting_name = str(advanced_override or advanced_master or "").strip()
                    if len(selected_entities) < 2:
                        st.warning("Check at least two outlets/reporting names before applying a merge.")
                    elif not reporting_name:
                        st.warning("Enter a reporting name.")
                    else:
                        source_outlets_to_merge: list[str] = []
                        for entity in selected_entities:
                            matching_sources = canonical_to_sources.get(entity, [])
                            source_outlets_to_merge.extend(matching_sources if matching_sources else [entity])
                        apply_outlet_rollup_map(
                            st.session_state,
                            list(dict.fromkeys(source_outlets_to_merge)),
                            reporting_name,
                        )
                        st.session_state.outlet_cleanup_advanced_filter_reset_pending = True
                        st.session_state.outlet_cleanup_advanced_override_reset_pending = True
                        st.session_state.outlet_cleanup_cluster_index = 0
                        st.rerun()

        current_mappings = st.session_state.get("outlet_rollup_map", {})
        if current_mappings:
            st.divider()
            with st.expander("Reporting rollups", expanded=False):
                grouped_rollups = (
                    pd.DataFrame(
                        [{"Source Outlet": source, "Reporting Name": target} for source, target in current_mappings.items()]
                    )
                    .groupby("Reporting Name", as_index=False)
                    .agg(
                        Source_Count=("Source Outlet", "count"),
                        Source_Outlets=("Source Outlet", lambda s: " | ".join(sorted(str(x) for x in s.head(8)))),
                    )
                    .sort_values(["Source_Count", "Reporting Name"], ascending=[False, True])
                    .reset_index(drop=True)
                )
                st.dataframe(
                    grouped_rollups,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Source_Count": st.column_config.NumberColumn("Source Outlets", width="small", format="%d"),
                        "Source_Outlets": st.column_config.Column("Examples", width="large"),
                    },
                )

            with st.expander("Current rollup mappings", expanded=False):
                mapping_df = pd.DataFrame(
                    [{"Source Outlet": source, "Canonical Outlet": target} for source, target in current_mappings.items()]
                ).sort_values(["Canonical Outlet", "Source Outlet"]).reset_index(drop=True)
                mapping_df["Remove"] = False
                mapping_editor = st.data_editor(
                    mapping_df,
                    use_container_width=True,
                    hide_index=True,
                    key="outlet_mapping_editor",
                    column_config={"Remove": st.column_config.CheckboxColumn("Remove", width="small")},
                )
                remove_rows = mapping_editor[mapping_editor["Remove"]].index.tolist()
                if remove_rows:
                    remove_sources = mapping_df.iloc[remove_rows]["Source Outlet"].tolist()
                    remove_outlet_rollup_map(st.session_state, remove_sources)
                    st.rerun()
                if st.button("Clear all mappings", key="outlet_clear_all_mappings"):
                    remove_outlet_rollup_map(st.session_state, mapping_df["Source Outlet"].tolist())
                    st.session_state.outlet_cleanup_cluster_index = 0
                    st.rerun()
        elif not get_variants_df().empty:
            st.divider()
            with st.expander("Possible naming variants", expanded=False):
                st.caption("Optional reference only. These are simple normalized-name clusters that might suggest additional cleanup opportunities.")
                st.dataframe(get_variants_df().style.format(NUMERIC_FORMAT_DICT, na_rep=" "), use_container_width=True, hide_index=True)

    def render_selection_section() -> None:
        st.session_state.outlets_section = "Selection"
        rank_by = st.radio(
            "Ranking metric",
            ["Mentions", "Impressions", "Effective Reach"],
            horizontal=True,
            key="outlets_rank_by",
        )
        ranked = get_ranked_outlet_metrics()
        valid_outlets = ranked["Outlet"].tolist()
        if not valid_outlets:
            st.info("No outlets available.")
            return

        active_outlet = str(st.session_state.get("outlet_insights_active_outlet", "") or "")
        pending_active_outlet = str(st.session_state.pop("outlet_insights_pending_active_outlet", "") or "")
        if pending_active_outlet:
            active_outlet = pending_active_outlet
        if active_outlet not in valid_outlets:
            active_outlet = valid_outlets[0]

        st.session_state.outlet_insights_active_outlet = active_outlet

        if st.session_state.get("outlet_insights_active_outlet", "") not in valid_outlets:
            st.session_state.outlet_insights_active_outlet = valid_outlets[0]

        current_selected = [
            outlet for outlet in st.session_state.get("outlet_insights_selected_outlets", [])
            if outlet in valid_outlets
        ]
        st.session_state.outlet_insights_selected_outlets = current_selected

        suggested_outlets = ranked.head(10)["Outlet"].tolist()
        candidate_df = ranked[~ranked["Outlet"].isin(current_selected)].copy().head(50)
        candidate_df["Keep"] = False

        left_col, right_col = st.columns([0.9, 1.25], gap="large")
        with left_col:
            st.subheader("Outlet Inspector")
            inspect_outlet = st.session_state["outlet_insights_active_outlet"]
            inspect_index = valid_outlets.index(inspect_outlet) if inspect_outlet in valid_outlets else 0
            st.selectbox(
                "Inspect outlet",
                options=valid_outlets,
                key="outlet_insights_active_outlet",
            )
            inspect_outlet = st.session_state["outlet_insights_active_outlet"]
            inspect_index = valid_outlets.index(inspect_outlet) if inspect_outlet in valid_outlets else 0
            nav_col1, nav_col2, nav_col3 = st.columns([1, 1, 1.3], gap="small")
            with nav_col1:
                if st.button("", key="outlet_inspector_prev", use_container_width=True, disabled=inspect_index <= 0, icon=":material/skip_previous:", help="Previous outlet"):
                    st.session_state.outlet_insights_pending_active_outlet = valid_outlets[inspect_index - 1]
                    st.rerun()
            with nav_col2:
                if st.button("", key="outlet_inspector_next", use_container_width=True, disabled=inspect_index >= len(valid_outlets) - 1, icon=":material/skip_next:", help="Next outlet"):
                    st.session_state.outlet_insights_pending_active_outlet = valid_outlets[inspect_index + 1]
                    st.rerun()
            with nav_col3:
                save_label = "Already saved" if inspect_outlet in current_selected else "Save this outlet"
                if st.button(
                    save_label,
                    key="outlet_save_inspected",
                    type="primary" if inspect_outlet not in current_selected else "secondary",
                    use_container_width=True,
                    disabled=inspect_outlet in current_selected,
                ):
                    selected = list(dict.fromkeys(current_selected + [inspect_outlet]))
                    st.session_state.outlets_section = "Selection"
                    st.session_state.outlet_insights_selected_outlets = selected
                    st.session_state.outlet_selection_checked_outlets = selected
                    st.session_state.outlet_insights_summaries = {
                        k: v for k, v in st.session_state.get("outlet_insights_summaries", {}).items()
                        if k in selected
                    }
                    current_index = valid_outlets.index(inspect_outlet) if inspect_outlet in valid_outlets else 0
                    next_index = min(current_index + 1, len(valid_outlets) - 1)
                    st.session_state.outlet_insights_pending_active_outlet = valid_outlets[next_index]
                    st.rerun()
            st.caption(f"{inspect_index + 1} of {len(valid_outlets)} by {st.session_state.outlets_rank_by}")
            inspect_row = ranked.loc[ranked["Outlet"] == inspect_outlet].iloc[0]
            story_df = build_outlet_headline_table(get_story_rows(), inspect_outlet, limit=5)
            top_authors_df = build_outlet_top_authors(
                st.session_state.df_traditional,
                inspect_outlet,
                limit=5,
                outlet_rollup_map=st.session_state.get("outlet_rollup_map", {}),
            )
            st.markdown(
                (
                    '<div style="font-size:0.92rem; color:#9aa0aa; margin:0.15rem 0 0.65rem 0;">'
                    f"Top types: {html.escape(str(inspect_row.get('Top_Types', '') or 'Unknown'))} | "
                    f"Mentions: {int(inspect_row.get('Mention_Total', 0)):,} | "
                    f"Impressions: {int(inspect_row.get('Impressions', 0)):,}"
                    "</div>"
                ),
                unsafe_allow_html=True,
            )
            examples_html = build_story_examples_html(story_df)
            if examples_html:
                st.markdown(examples_html, unsafe_allow_html=True)
            if not top_authors_df.empty:
                st.write("**Top authors in this outlet**")
                st.dataframe(
                    top_authors_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Mentions": st.column_config.NumberColumn("Mentions", width="small", format="%d"),
                        "Impressions": st.column_config.NumberColumn("Impressions", width="small", format="%,d"),
                    },
                )

        with right_col:
            st.subheader("Candidate Outlets")
            st.caption('Check the "Keep" box for outlets you want on the final shortlist, then click "Save Selected".')
            checked_outlets = [
                outlet for outlet in st.session_state.get("outlet_selection_checked_outlets", [])
                if outlet in set(candidate_df["Outlet"].astype(str).tolist())
            ]
            candidate_df = candidate_df.copy()
            candidate_df["Keep"] = candidate_df["Outlet"].isin(checked_outlets)
            candidate_editor = st.data_editor(
                candidate_df[["Outlet", "Mention_Total", "Impressions", "Effective_Reach", "Keep"]],
                use_container_width=True,
                hide_index=True,
                key=f"outlet_candidate_editor_{st.session_state.get('outlet_selection_editor_version', 0)}",
                column_config={
                    "Mention_Total": st.column_config.NumberColumn("Mentions", width="small", format="%d"),
                    "Impressions": st.column_config.NumberColumn("Impressions", width="small", format="%,d"),
                    "Effective_Reach": st.column_config.NumberColumn("Effective Reach", width="small", format="%,d"),
                    "Keep": st.column_config.CheckboxColumn("Keep", width="small"),
                },
            )

            action1, action2, action3 = st.columns([1, 1, 2], gap="small")
            with action1:
                if st.button("Check Top 10", key="outlet_use_suggestion"):
                    st.session_state.outlets_section = "Selection"
                    st.session_state.outlet_selection_checked_outlets = suggested_outlets
                    st.session_state.outlet_selection_editor_version += 1
                    st.rerun()
            with action2:
                if st.button("Clear Selected", key="outlet_clear_selected"):
                    st.session_state.outlets_section = "Selection"
                    st.session_state.outlet_insights_selected_outlets = []
                    st.session_state.outlet_selection_checked_outlets = []
                    st.session_state.outlet_selection_editor_version += 1
                    st.session_state.outlet_insights_summaries = {}
                    st.rerun()
            with action3:
                if st.button("Save Selected", key="outlet_save_selected", type="primary"):
                    st.session_state.outlets_section = "Selection"
                    newly_selected = candidate_editor.loc[candidate_editor["Keep"], "Outlet"].tolist()
                    selected = list(dict.fromkeys(current_selected + newly_selected))
                    st.session_state.outlet_insights_selected_outlets = selected
                    st.session_state.outlet_selection_checked_outlets = selected
                    st.session_state.outlet_insights_summaries = {
                        k: v for k, v in st.session_state.get("outlet_insights_summaries", {}).items()
                        if k in selected
                    }
                    st.rerun()
            st.caption(f"Selected {len(current_selected)} outlet(s). Target: 10.")

        selected_outlets = st.session_state.get("outlet_insights_selected_outlets", [])
        if selected_outlets:
            st.divider()
            st.write("**Current shortlist**")
            shortlist_df = ranked[ranked["Outlet"].isin(selected_outlets)].copy()
            shortlist_df["Coverage Themes"] = shortlist_df["Outlet"].map(
                lambda outlet: st.session_state.get("outlet_insights_summaries", {}).get(outlet, "")
            )
            shortlist_view = shortlist_df[[
                "Outlet",
                "Top_Types",
                "Mention_Total",
                "Impressions",
                "Effective_Reach",
            ]].copy()
            shortlist_view["Delete"] = False
            shortlist_editor = st.data_editor(
                shortlist_view,
                use_container_width=True,
                hide_index=True,
                key="outlet_shortlist_editor",
                column_config={
                    "Top_Types": st.column_config.Column("Media Types", width="medium"),
                    "Mention_Total": st.column_config.NumberColumn("Mentions", width="small", format="%d"),
                    "Impressions": st.column_config.NumberColumn("Impressions", width="small", format="%,d"),
                    "Effective_Reach": st.column_config.NumberColumn("Effective Reach", width="small", format="%,d"),
                    "Delete": st.column_config.CheckboxColumn("Delete", width="small"),
                },
            )
            rows_to_delete = shortlist_editor[shortlist_editor["Delete"]].index.tolist()
            if rows_to_delete:
                st.session_state.outlets_section = "Selection"
                remove_outlets = shortlist_df.iloc[rows_to_delete]["Outlet"].tolist()
                remaining = [o for o in selected_outlets if o not in remove_outlets]
                st.session_state.outlet_insights_selected_outlets = remaining
                st.session_state.outlet_insights_summaries = {
                    k: v for k, v in st.session_state.get("outlet_insights_summaries", {}).items()
                    if k in remaining
                }
                st.rerun()

    def render_insights_section() -> None:
        st.session_state.outlets_section = "Insights"
        selected_outlets = st.session_state.get("outlet_insights_selected_outlets", [])
        if not selected_outlets:
            st.info("Save outlets in Selection before reviewing insights.")
            return

        metric_label = st.radio(
            "Ranking metric",
            ["Mentions", "Impressions", "Effective Reach"],
            horizontal=True,
            key="outlets_rank_by",
        )

        ranked = get_ranked_outlet_metrics()
        shortlist_df = ranked[ranked["Outlet"].isin(selected_outlets)].copy()
        shortlist_df["Coverage Themes"] = shortlist_df["Outlet"].map(
            lambda outlet: st.session_state.get("outlet_insights_summaries", {}).get(outlet, "")
        )

        chart_table = shortlist_df[[
            "Outlet",
            "Top_Types",
            "Mention_Total",
            "Unique_Mentions",
            "Impressions",
            "Effective_Reach",
        ]].copy().rename(
            columns={
                "Top_Types": "Media Types",
                "Mention_Total": "Mentions",
                "Unique_Mentions": "Unique Mentions",
                "Effective_Reach": "Effective Reach",
            }
        )
        chart_table = chart_table.sort_values(metric_label, ascending=False).reset_index(drop=True)

        chart_tab, table_tab = st.tabs(["Chart", "Table"])
        with chart_tab:
            try:
                alt = importlib.import_module("altair")
            except Exception:
                alt = None

            if alt is None:
                st.info("Outlet chart unavailable in this environment.")
            else:
                plot_df = chart_table.copy()
                plot_df["Metric Label"] = plot_df[metric_label].apply(format_compact_integer)
                chart_height = max(280, len(plot_df) * 36)
                max_metric = float(plot_df[metric_label].max()) if not plot_df.empty else 0.0
                padded_max = max_metric * 1.18 if max_metric > 0 else 1.0
                compact_axis_expr = """
                    datum.value >= 1e9 ? format(datum.value / 1e9, '.0f') + 'B' :
                    datum.value >= 1e6 ? format(datum.value / 1e6, '.0f') + 'M' :
                    datum.value >= 1e3 ? format(datum.value / 1e3, '.0f') + 'K' :
                    format(datum.value, ',')
                """

                bars = alt.Chart(plot_df).mark_bar(color="#37415f", cornerRadiusEnd=2).encode(
                    y=alt.Y("Outlet:N", sort=plot_df["Outlet"].tolist(), axis=alt.Axis(title=None, labelLimit=300)),
                    x=alt.X(
                        f"{metric_label}:Q",
                        title=None,
                        axis=alt.Axis(labelExpr=compact_axis_expr, grid=True),
                        scale=alt.Scale(domain=[0, padded_max], nice=False),
                    ),
                    tooltip=[
                        alt.Tooltip("Outlet:N", title="Outlet"),
                        alt.Tooltip("Media Types:N", title="Media Types"),
                        alt.Tooltip(f"{metric_label}:Q", title=metric_label, format=","),
                    ],
                )
                labels = alt.Chart(plot_df).mark_text(
                    align="left",
                    baseline="middle",
                    dx=6,
                    color="#f3f4f6",
                    fontSize=11,
                ).encode(
                    y=alt.Y("Outlet:N", sort=plot_df["Outlet"].tolist()),
                    x=alt.X(f"{metric_label}:Q"),
                    text=alt.Text("Metric Label:N"),
                )

                chart = (
                    (bars + labels)
                    .properties(height=chart_height, title=f"Top Outlets by {metric_label.lower()}")
                    .configure_view(strokeWidth=0)
                )
                st.altair_chart(chart, use_container_width=True)
        with table_tab:
            st.dataframe(
                chart_table,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Mentions": st.column_config.NumberColumn("Mentions", width="small", format="%d"),
                    "Unique Mentions": st.column_config.NumberColumn("Unique Mentions", width="small", format="%d"),
                    "Impressions": st.column_config.NumberColumn("Impressions", width="small", format="%,d"),
                    "Effective Reach": st.column_config.NumberColumn("Effective Reach", width="small", format="%,d"),
                },
            )

        st.divider()
        st.subheader("Report Copy")
        generate_col1, generate_col2 = st.columns([1.2, 3], gap="medium")
        with generate_col1:
            if st.button("Generate outlet coverage themes", type="primary", key="generate_outlet_summaries"):
                summaries = dict(st.session_state.get("outlet_insights_summaries", {}))
                client_name = str(st.session_state.get("client_name", "")).strip()
                analysis_context = build_analysis_context_text(st.session_state)
                with st.spinner("Generating outlet theme summaries..."):
                    for outlet_name in selected_outlets:
                        outlet_row = shortlist_df.loc[shortlist_df["Outlet"] == outlet_name].iloc[0]
                        headline_df = build_outlet_headline_table(get_story_rows(), outlet_name, limit=6)
                        author_df = build_outlet_top_authors(
                            st.session_state.df_traditional,
                            outlet_name,
                            limit=5,
                            outlet_rollup_map=st.session_state.get("outlet_rollup_map", {}),
                        )
                        try:
                            summary_text, _, _ = generate_outlet_summary(
                                outlet_name=outlet_name,
                                client_name=client_name,
                                outlet_row=outlet_row,
                                headline_df=headline_df,
                                top_authors_df=author_df,
                                api_key=st.secrets["key"],
                                model=DEFAULT_OUTLET_SUMMARY_MODEL,
                                analysis_context=analysis_context,
                            )
                            summaries[outlet_name] = summary_text
                        except Exception as e:
                            summaries[outlet_name] = f"Could not generate summary: {e}"
                st.session_state.outlet_insights_summaries = summaries
                st.rerun()
        with generate_col2:
            st.caption("Uses shortlisted outlets plus representative grouped stories and top contributing authors to generate concise, report-ready coverage themes.")

        field_options = ["Author", "Date", "Media type", "Mentions", "Impressions", "Effective reach", "Examples"]
        if "outlets_report_selected_fields" not in st.session_state:
            st.session_state.outlets_report_selected_fields = field_options.copy()
        if "outlets_report_previous_fields" not in st.session_state:
            st.session_state.outlets_report_previous_fields = field_options.copy()

        child_fields = {"Author", "Date", "Media type", "Mentions", "Impressions", "Effective reach"}

        def _normalize_outlet_report_fields() -> None:
            current_fields = st.session_state.get("outlets_report_selected_fields", []) or []
            previous_fields = st.session_state.get("outlets_report_previous_fields", []) or []
            current_set = set(current_fields)
            previous_set = set(previous_fields)

            if "Examples" not in current_set and current_set & child_fields:
                if "Examples" in previous_set:
                    current_set -= child_fields
                else:
                    current_set.add("Examples")

            normalized_fields = [field for field in field_options if field in current_set]
            st.session_state.outlets_report_selected_fields = normalized_fields
            st.session_state.outlets_report_previous_fields = normalized_fields.copy()

        preset_col, fields_col = st.columns([0.18, 0.82], gap="small")
        with preset_col:
            bulk_col1, bulk_col2 = st.columns(2, gap="small")
            with bulk_col1:
                if st.button("All", key="outlets_report_select_all", use_container_width=True):
                    st.session_state.outlets_report_selected_fields = field_options.copy()
                    st.session_state.outlets_report_previous_fields = field_options.copy()
                    st.rerun()
            with bulk_col2:
                if st.button("None", key="outlets_report_select_none", use_container_width=True):
                    st.session_state.outlets_report_selected_fields = []
                    st.session_state.outlets_report_previous_fields = []
                    st.rerun()

        with fields_col:
            st.pills(
                "Fields",
                options=field_options,
                selection_mode="multi",
                default=st.session_state.get("outlets_report_selected_fields", field_options),
                key="outlets_report_selected_fields",
                on_change=_normalize_outlet_report_fields,
                label_visibility="collapsed",
            )

        selected_fields = st.session_state.get("outlets_report_selected_fields", []) or []
        st.session_state.outlets_report_previous_fields = list(selected_fields)
        selected_field_set = set(selected_fields)
        show_example_author = "Author" in selected_field_set
        show_example_date = "Date" in selected_field_set
        show_example_type = "Media type" in selected_field_set
        show_example_mentions = "Mentions" in selected_field_set
        show_example_impressions = "Impressions" in selected_field_set
        show_example_effective_reach = "Effective reach" in selected_field_set
        show_headline_examples = "Examples" in selected_field_set

        report_html = build_report_html(
            shortlist_df,
            show_author=show_example_author,
            show_date=show_example_date,
            show_media_type=show_example_type,
            show_mentions=show_example_mentions,
            show_impressions=show_example_impressions,
            show_effective_reach=show_example_effective_reach,
            show_headline_examples=show_headline_examples,
        )
        if report_html:
            st.markdown(report_html, unsafe_allow_html=True)
        else:
            st.info("Generate outlet coverage themes to build the narrative block.")

    section = st.session_state.get("outlets_section", "Cleanup")
    if section == "Selection":
        render_selection_section()
    elif section == "Insights":
        render_insights_section()
    else:
        render_cleanup_section()
