from __future__ import annotations

import html
import importlib
import warnings

import pandas as pd
import streamlit as st

from processing.outlet_insights import (
    DEFAULT_OUTLET_SUMMARY_MODEL,
    apply_outlet_name_cleanup,
    build_outlet_headline_table,
    build_outlet_metrics,
    build_outlet_top_authors,
    build_outlet_variant_candidates,
    generate_outlet_summary,
    init_outlet_workflow_state,
)
from utils.formatting import NUMERIC_FORMAT_DICT

warnings.filterwarnings("ignore")

METRIC_FIELD_MAP = {
    "Mentions": "Mention_Total",
    "Unique Mentions": "Unique_Mentions",
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


st.title("Outlets")
st.caption("Clean up outlet names, curate the final outlet list, and review output-ready insights in one place.")

if not st.session_state.get("standard_step", False):
    st.error("Please complete Basic Cleaning before trying this step.")
    st.stop()

if len(st.session_state.get("df_traditional", [])) == 0:
    st.error("Traditional / online / broadcast coverage is missing. Please complete Basic Cleaning again.")
    st.stop()

init_outlet_workflow_state(st.session_state)


def rebuild_outlet_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics_df, story_df = build_outlet_metrics(st.session_state.df_traditional.copy())
    variants_df = build_outlet_variant_candidates(st.session_state.df_traditional.copy())
    return metrics_df, story_df, variants_df


def render_story_examples(df: pd.DataFrame) -> None:
    if df.empty:
        st.info("No example stories available for this outlet.")
        return

    html_blocks: list[str] = []
    for _, row in df.iterrows():
        headline = str(row.get("Headline", "") or "").strip()
        url = str(row.get("Representative URL", "") or "").strip()
        author = str(row.get("Author", "") or "").strip()
        date_val = row.get("Date")
        mentions = int(pd.to_numeric(pd.Series([row.get("Story Mentions", 0)]), errors="coerce").fillna(0).iloc[0])
        impressions = int(pd.to_numeric(pd.Series([row.get("Story Impressions", 0)]), errors="coerce").fillna(0).iloc[0])

        headline_line = f'<a href="{url}" target="_blank">{html.escape(headline)}</a>' if url else html.escape(headline)
        meta_parts = []
        if author:
            meta_parts.append(f"<em>{html.escape(author)}</em>")
        if pd.notna(date_val):
            meta_parts.append(pd.to_datetime(date_val).strftime("%B %d, %Y"))
        meta_line = " - ".join(meta_parts)
        stats_line = f"Mentions: {mentions:,} | Impressions: {impressions:,}"

        html_blocks.append(
            f"""
            <div style="margin-bottom:0.55rem;">
                <div>{headline_line}</div>
                <div style="font-size:0.78rem; opacity:0.72; line-height:1.2;">{meta_line}</div>
                <div style="font-size:0.78rem; opacity:0.72; line-height:1.2;">{html.escape(stats_line)}</div>
            </div>
            """
        )

    st.markdown("".join(html_blocks), unsafe_allow_html=True)


def build_report_html(shortlist_df: pd.DataFrame) -> str:
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

        metrics = (
            f"Mentions: {mentions:,} | "
            f"Unique Mentions: {unique_mentions:,} | "
            f"Impressions: {impressions:,} | "
            f"Effective Reach: {effective_reach:,}"
        )

        block = (
            '<div style="margin-bottom:1.2rem;">'
            f'<div style="font-size:1.08rem; font-weight:700; margin-bottom:0.2rem;">{header}</div>'
            f'<div style="line-height:1.55; margin-bottom:0.28rem;">{html.escape(themes)}</div>'
            f'<div style="font-size:0.84rem; opacity:0.72; letter-spacing:0.01em;">{html.escape(metrics)}</div>'
            '</div>'
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
    if st.button("1. Cleanup", key="outlets_step_cleanup", use_container_width=True):
        st.session_state.outlets_section = "Cleanup"
with step2:
    if st.button("2. Selection", key="outlets_step_selection", use_container_width=True):
        st.session_state.outlets_section = "Selection"
with step3:
    if st.button("3. Insights", key="outlets_step_insights", use_container_width=True):
        st.session_state.outlets_section = "Insights"

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

    rank_by = st.radio(
        "Rank outlets by",
        ["Mentions", "Unique Mentions", "Impressions", "Effective Reach"],
        horizontal=True,
        key="outlets_rank_by_cleanup",
    )
    st.session_state.outlets_rank_by = rank_by
    ranked = get_ranked_outlet_metrics()
    outlet_options = ranked["Outlet"].tolist()

    pending_source = str(st.session_state.pop("outlet_cleanup_pending_source", "") or "").strip()
    if pending_source and pending_source in outlet_options:
        st.session_state.outlet_cleanup_source = pending_source
        st.session_state.outlet_cleanup_target = pending_source
        st.session_state.outlet_cleanup_merge_target = ""

    current_source = st.session_state.get("outlet_cleanup_source", "")
    if current_source not in outlet_options:
        st.session_state.outlet_cleanup_source = outlet_options[0]
        st.session_state.outlet_cleanup_target = outlet_options[0]

    selected_outlet = st.selectbox(
        "Review outlet name",
        options=outlet_options,
        key="outlet_cleanup_source",
    )

    if st.session_state.get("outlet_cleanup_target", "") == "":
        st.session_state.outlet_cleanup_target = selected_outlet

    selected_row = ranked.loc[ranked["Outlet"] == selected_outlet].iloc[0]
    top_examples = build_outlet_headline_table(story_rows, selected_outlet, limit=3)

    stats1, stats2, stats3, stats4 = st.columns(4)
    with stats1:
        st.metric("Mentions", f"{int(selected_row['Mention_Total']):,}")
    with stats2:
        st.metric("Unique Mentions", f"{int(selected_row['Unique_Mentions']):,}")
    with stats3:
        st.metric("Impressions", f"{int(selected_row['Impressions']):,}")
    with stats4:
        st.metric("Effective Reach", f"{int(selected_row['Effective_Reach']):,}")

    with st.form("outlet_cleanup_form"):
        form_col1, form_col2 = st.columns(2, gap="large")
        with form_col1:
            merge_target = st.selectbox(
                "Merge into existing outlet (optional)",
                options=[""] + [o for o in outlet_options if o != selected_outlet],
                key="outlet_cleanup_merge_target",
            )
        with form_col2:
            new_name = st.text_input(
                "Or write a canonical outlet name",
                value=st.session_state.get("outlet_cleanup_target", selected_outlet),
                key="outlet_cleanup_target",
            )
        submitted = st.form_submit_button("Apply Outlet Cleanup", type="primary")

    if submitted:
        final_name = merge_target.strip() if merge_target.strip() else new_name.strip()
        if not final_name:
            st.warning("Please choose or enter a target outlet name.")
        elif final_name == selected_outlet:
            st.info("No change to apply.")
        else:
            apply_outlet_name_cleanup(st.session_state, selected_outlet, final_name)
            st.session_state.outlet_cleanup_pending_source = final_name
            st.rerun()

    preview_col1, preview_col2 = st.columns([1.15, 0.85], gap="large")
    with preview_col1:
        st.write("**Top outlets**")
        preview_df = ranked[["Outlet", "Mention_Total", "Unique_Mentions", "Impressions", "Effective_Reach"]].head(20).copy()
        preview_df = preview_df.rename(
            columns={
                "Mention_Total": "Mentions",
                "Unique_Mentions": "Unique Mentions",
                "Effective_Reach": "Effective Reach",
            }
        )
        st.dataframe(preview_df.style.format(NUMERIC_FORMAT_DICT, na_rep=" "), use_container_width=True, hide_index=True)
    with preview_col2:
        st.write("**Selected outlet examples**")
        render_story_examples(top_examples)

    st.divider()
    st.write("**Potential outlet variants**")
    if variants_df.empty:
        st.info("No obvious duplicate outlet variants were detected from simple name normalization.")
    else:
        st.dataframe(variants_df.style.format(NUMERIC_FORMAT_DICT, na_rep=" "), use_container_width=True, hide_index=True)


def render_selection_section() -> None:
    st.session_state.outlets_section = "Selection"
    st.session_state.outlets_rank_by = st.radio(
        "Rank outlets by",
        ["Mentions", "Unique Mentions", "Impressions", "Effective Reach"],
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
        top_authors_df = build_outlet_top_authors(st.session_state.df_traditional, inspect_outlet, limit=5)
        st.caption(
            f"Top types: {inspect_row.get('Top_Types', '') or 'Unknown'} | "
            f"Mentions: {int(inspect_row.get('Mention_Total', 0)):,} | "
            f"Unique Mentions: {int(inspect_row.get('Unique_Mentions', 0)):,} | "
            f"Impressions: {int(inspect_row.get('Impressions', 0)):,}"
        )
        render_story_examples(story_df)
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
        candidate_editor = st.data_editor(
            candidate_df[["Outlet", "Mention_Total", "Unique_Mentions", "Impressions", "Effective_Reach", "Keep"]],
            use_container_width=True,
            hide_index=True,
            key="outlet_candidate_editor",
            column_config={
                "Mention_Total": st.column_config.NumberColumn("Mentions", width="small", format="%d"),
                "Unique_Mentions": st.column_config.NumberColumn("Unique Mentions", width="small", format="%d"),
                "Impressions": st.column_config.NumberColumn("Impressions", width="small", format="%,d"),
                "Effective_Reach": st.column_config.NumberColumn("Effective Reach", width="small", format="%,d"),
                "Keep": st.column_config.CheckboxColumn("Keep", width="small"),
            },
        )

        action1, action2, action3 = st.columns([1, 1, 2], gap="small")
        with action1:
            if st.button("Use Top Suggestion", key="outlet_use_suggestion"):
                st.session_state.outlets_section = "Selection"
                st.session_state.outlet_insights_selected_outlets = suggested_outlets
                st.session_state.outlet_insights_summaries = {
                    k: v for k, v in st.session_state.get("outlet_insights_summaries", {}).items()
                    if k in suggested_outlets
                }
                st.rerun()
        with action2:
            if st.button("Clear Selected", key="outlet_clear_selected"):
                st.session_state.outlets_section = "Selection"
                st.session_state.outlet_insights_selected_outlets = []
                st.session_state.outlet_insights_summaries = {}
                st.rerun()
        with action3:
            if st.button("Save Selected", key="outlet_save_selected", type="primary"):
                st.session_state.outlets_section = "Selection"
                newly_selected = candidate_editor.loc[candidate_editor["Keep"], "Outlet"].tolist()
                selected = list(dict.fromkeys(current_selected + newly_selected))
                st.session_state.outlet_insights_selected_outlets = selected
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

        if st.button("Generate outlet coverage themes", type="primary", key="generate_outlet_summaries"):
            st.session_state.outlets_section = "Insights"
            summaries = dict(st.session_state.get("outlet_insights_summaries", {}))
            client_name = str(st.session_state.get("client_name", "")).strip()
            with st.spinner("Generating outlet theme summaries..."):
                for outlet_name in selected_outlets:
                    outlet_row = shortlist_df.loc[shortlist_df["Outlet"] == outlet_name].iloc[0]
                    headline_df = build_outlet_headline_table(story_rows, outlet_name, limit=6)
                    author_df = build_outlet_top_authors(st.session_state.df_traditional, outlet_name, limit=5)
                    try:
                        summary_text, _, _ = generate_outlet_summary(
                            outlet_name=outlet_name,
                            client_name=client_name,
                            outlet_row=outlet_row,
                            headline_df=headline_df,
                            top_authors_df=author_df,
                            api_key=st.secrets["key"],
                            model=DEFAULT_OUTLET_SUMMARY_MODEL,
                        )
                        summaries[outlet_name] = summary_text
                    except Exception as e:
                        summaries[outlet_name] = f"Could not generate summary: {e}"
            st.session_state.outlet_insights_summaries = summaries
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
        ["Mentions", "Unique Mentions", "Impressions", "Effective Reach"],
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

    table_tab, chart_tab = st.tabs(["Table", "Chart"])
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

    st.divider()
    st.subheader("Report Copy")
    report_html = build_report_html(shortlist_df)
    if report_html:
        st.markdown(report_html, unsafe_allow_html=True)
    else:
        st.info("Generate outlet coverage themes in Selection to build the narrative block.")


section = st.session_state.get("outlets_section", "Cleanup")
if section == "Selection":
    render_selection_section()
elif section == "Insights":
    render_insights_section()
else:
    render_cleanup_section()
