from __future__ import annotations

import html
import importlib
import re
import warnings

import pandas as pd
import streamlit as st

from processing.analysis_context import build_analysis_context_text, init_analysis_context_state
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
        outlet_map = st.session_state.get("outlet_rollup_map", {})
        cache_key = (
            "outlet_data_cache",
            len(df_traditional),
            tuple(df_traditional.columns.tolist()),
            tuple(sorted((str(k), str(v)) for k, v in outlet_map.items())),
        )
        cached = st.session_state.get("outlet_data_cache")
        if cached and cached.get("key") == cache_key:
            return cached["metrics_df"], cached["story_df"], cached["variants_df"]

        metrics_df, story_df = build_outlet_metrics(df_traditional, outlet_rollup_map=outlet_map)
        variants_df = build_outlet_variant_candidates(df_traditional, outlet_rollup_map=outlet_map)
        st.session_state.outlet_data_cache = {
            "key": cache_key,
            "metrics_df": metrics_df,
            "story_df": story_df,
            "variants_df": variants_df,
        }
        return metrics_df, story_df, variants_df

    def build_story_examples_html(
        df: pd.DataFrame,
        show_author: bool = True,
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
                    "example_type": row.get("Type", ""),
                    "mentions": int(pd.to_numeric(pd.Series([row.get("Story Mentions", 0)]), errors="coerce").fillna(0).iloc[0]),
                    "impressions": int(pd.to_numeric(pd.Series([row.get("Story Impressions", 0)]), errors="coerce").fillna(0).iloc[0]),
                    "effective_reach": int(pd.to_numeric(pd.Series([row.get("Story Effective Reach", 0)]), errors="coerce").fillna(0).iloc[0]),
                }
            )

        return build_linked_example_blocks_html(
            items,
            show_outlet=show_author,
            show_media_type=show_media_type,
            show_mentions=show_mentions,
            show_impressions=show_impressions,
            show_effective_reach=show_effective_reach,
        )

    def build_report_html(
        shortlist_df: pd.DataFrame,
        show_mentions: bool,
        show_unique_mentions: bool,
        show_impressions: bool,
        show_effective_reach: bool,
        show_headline_examples: bool,
    ) -> str:
        blocks = []
        for _, row in shortlist_df.iterrows():
            outlet = str(row.get("Outlet", "") or "").strip()
            top_types = str(row.get("Top_Types", "") or "").strip()
            themes = " ".join(str(row.get("Coverage Themes", "") or "").split())
            mentions = int(row.get("Mention_Total", 0) or 0)
            unique_mentions = int(row.get("Unique_Mentions", 0) or 0)
            impressions = int(row.get("Impressions", 0) or 0)
            effective_reach = int(row.get("Effective_Reach", 0) or 0)

            header = f"<strong>{html.escape(outlet)}</strong>"
            if top_types:
                header += f' <span style="opacity:0.82;">|</span> <span style="font-style:italic; opacity:0.92;">{html.escape(top_types)}</span>'

            metric_parts = []
            if show_mentions:
                metric_parts.append(f"Mentions: {mentions:,}")
            if show_unique_mentions:
                metric_parts.append(f"Unique Mentions: {unique_mentions:,}")
            if show_impressions:
                metric_parts.append(f"Impressions: {impressions:,}")
            if show_effective_reach:
                metric_parts.append(f"Effective Reach: {effective_reach:,}")
            metrics = " | ".join(metric_parts)
            metrics_html = (
                f'<div style="font-size:0.84rem; opacity:0.72; letter-spacing:0.01em;">{html.escape(metrics)}</div>'
                if metrics else ""
            )

            examples_html = ""
            if show_headline_examples and outlet:
                story_df = build_outlet_headline_table(story_rows, outlet, limit=5)
                example_items = build_story_examples_html(
                    story_df,
                    show_author=True,
                    show_media_type=True,
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
                f"{metrics_html}"
                f"{examples_html}"
                "</div>"
            )
            blocks.append(block)

        return f'<div style="display:block;">{"".join(blocks)}</div>' if blocks else ""

    metrics_df, story_rows, variants_df = rebuild_outlet_data()
    if metrics_df.empty:
        st.info("No outlet-level data available.")
        st.stop()

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
        return metrics_df.sort_values(sort_cols, ascending=False).reset_index(drop=True)

    def render_cleanup_section() -> None:
        st.session_state.outlets_section = "Cleanup"
        st.subheader("Outlet Cleanup")
        st.info("Cleanup does not overwrite raw outlet names in your cleaned data. It creates a reporting rollup map used by this Outlets workflow and downstream exports.")

        rank_by = st.radio(
            "Rank outlets by",
            ["Mentions", "Impressions", "Effective Reach"],
            horizontal=True,
            key="outlets_rank_by_cleanup",
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

    def render_selection_section() -> None:
        st.session_state.outlets_section = "Selection"
        st.session_state.outlets_rank_by = st.radio(
            "Rank outlets by",
            ["Mentions", "Impressions", "Effective Reach"],
            horizontal=True,
            key="outlets_rank_by_selection",
        )
        ranked = get_ranked_outlet_metrics()
        valid_outlets = ranked["Outlet"].tolist()
        if not valid_outlets:
            st.info("No outlets available.")
            return

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
            st.selectbox(
                "Inspect outlet",
                options=valid_outlets,
                key="outlet_insights_active_outlet",
            )
            inspect_outlet = st.session_state["outlet_insights_active_outlet"]
            inspect_row = ranked.loc[ranked["Outlet"] == inspect_outlet].iloc[0]
            story_df = build_outlet_headline_table(story_rows, inspect_outlet, limit=5)
            top_authors_df = build_outlet_top_authors(
                st.session_state.df_traditional,
                inspect_outlet,
                limit=5,
                outlet_rollup_map=st.session_state.get("outlet_rollup_map", {}),
            )
            st.caption(
                f"Top types: {inspect_row.get('Top_Types', '') or 'Unknown'} | "
                f"Mentions: {int(inspect_row.get('Mention_Total', 0)):,} | "
                f"Impressions: {int(inspect_row.get('Impressions', 0)):,}"
            )
            examples_html = build_story_examples_html(story_df)
            if examples_html:
                st.markdown(examples_html, unsafe_allow_html=True)
            st.write("**Top authors in this outlet**")
            if top_authors_df.empty:
                st.info("No attributed authors available for this outlet.")
            else:
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
                if st.button("Check Top Suggestion", key="outlet_use_suggestion"):
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
            shortlist_df["SortOrder"] = shortlist_df["Outlet"].map({name: idx for idx, name in enumerate(selected_outlets)})
            shortlist_df = shortlist_df.sort_values("SortOrder").drop(columns=["SortOrder"])
            shortlist_df["Coverage Themes"] = shortlist_df["Outlet"].map(
                lambda outlet: st.session_state.get("outlet_insights_summaries", {}).get(outlet, "")
            )
            shortlist_view = shortlist_df[[
                "Outlet",
                "Top_Types",
                "Mention_Total",
                "Unique_Mentions",
                "Impressions",
                "Effective_Reach",
                "Coverage Themes",
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
                    "Unique_Mentions": st.column_config.NumberColumn("Unique Mentions", width="small", format="%d"),
                    "Impressions": st.column_config.NumberColumn("Impressions", width="small", format="%,d"),
                    "Effective_Reach": st.column_config.NumberColumn("Effective Reach", width="small", format="%,d"),
                    "Coverage Themes": st.column_config.Column("Coverage Themes", width="large"),
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

        ranked = get_ranked_outlet_metrics()
        shortlist_df = ranked[ranked["Outlet"].isin(selected_outlets)].copy()
        shortlist_df["SortOrder"] = shortlist_df["Outlet"].map({name: idx for idx, name in enumerate(selected_outlets)})
        shortlist_df = shortlist_df.sort_values("SortOrder").drop(columns=["SortOrder"])
        shortlist_df["Coverage Themes"] = shortlist_df["Outlet"].map(
            lambda outlet: st.session_state.get("outlet_insights_summaries", {}).get(outlet, "")
        )

        metric_label = st.radio(
            "Chart metric",
            ["Mentions", "Impressions", "Effective Reach"],
            horizontal=True,
            key="outlets_insights_chart_metric",
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
                        headline_df = build_outlet_headline_table(story_rows, outlet_name, limit=6)
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

        metrics_col1, metrics_col2, metrics_col3, metrics_col4, metrics_col5 = st.columns(5, gap="small")
        with metrics_col1:
            show_mentions = st.checkbox("Show mentions", value=True, key="outlets_report_show_mentions")
        with metrics_col2:
            show_unique_mentions = st.checkbox("Show unique mentions", value=True, key="outlets_report_show_unique_mentions")
        with metrics_col3:
            show_impressions = st.checkbox("Show impressions", value=True, key="outlets_report_show_impressions")
        with metrics_col4:
            show_effective_reach = st.checkbox("Show effective reach", value=True, key="outlets_report_show_effective_reach")
        with metrics_col5:
            show_headline_examples = st.checkbox("Examples", value=True, key="outlets_report_show_examples")

        report_html = build_report_html(
            shortlist_df,
            show_mentions=show_mentions,
            show_unique_mentions=show_unique_mentions,
            show_impressions=show_impressions,
            show_effective_reach=show_effective_reach,
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
