from __future__ import annotations

import importlib

import pandas as pd
import streamlit as st

from processing.analysis_context import (
    apply_session_coverage_flag_policy,
    build_analysis_context_text,
    build_dataset_scope_cache_key,
    format_qualitative_exclusion_caption,
    get_qualitative_coverage_flag_exclusions,
)
from processing.regions import (
    METRIC_FIELD_MAP,
    build_region_rankings,
    build_regions_health_summary,
    build_region_story_group_examples,
    build_regions_source_df,
    filter_regions_df,
    generate_regions_insight_output,
    init_regions_state,
)


def _format_integer(value: object) -> str:
    try:
        return f"{int(float(value or 0)):,}"
    except Exception:
        return ""


def _format_region_table_display(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame()

    display_df = df.copy()
    for col in ["Mentions", "Impressions", "Effective Reach", "Outlet Count"]:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(_format_integer)
    return display_df


def _build_regions_signature(session_state, filtered_df: pd.DataFrame) -> tuple:
    return (
        session_state.get("regions_metric", "Mentions"),
        tuple(session_state.get("regions_analysis_levels", [])),
        tuple(session_state.get("regions_exclude_coverage_flags", [])),
        tuple(session_state.get("regions_include_countries", [])),
        tuple(session_state.get("regions_exclude_countries", [])),
        int(len(filtered_df)),
        int(filtered_df["Mentions"].sum()) if not filtered_df.empty else 0,
        int(filtered_df["Impressions"].sum()) if not filtered_df.empty else 0,
        int(filtered_df["Effective Reach"].sum()) if not filtered_df.empty else 0,
    )


def _render_region_chart(level_df: pd.DataFrame, metric_label: str, title: str, region_label: str) -> None:
    if level_df is None or level_df.empty:
        st.info(f"No {title.lower()} data is available for the current filters.")
        return

    try:
        alt = importlib.import_module("altair")
    except Exception:
        alt = None

    if alt is None:
        st.info("Region chart unavailable in this environment.")
        return

    plot_df = level_df.head(10).copy()
    plot_df["Metric Label"] = plot_df[metric_label].apply(_format_integer)
    chart_height = max(260, len(plot_df) * 42)
    max_metric = float(plot_df[metric_label].max()) if not plot_df.empty else 0.0
    padded_max = max_metric * 1.18 if max_metric > 0 else 1.0
    compact_axis_expr = """
        datum.value >= 1e9 ? format(datum.value / 1e9, '.0f') + 'B' :
        datum.value >= 1e6 ? format(datum.value / 1e6, '.0f') + 'M' :
        datum.value >= 1e3 ? format(datum.value / 1e3, '.0f') + 'K' :
        format(datum.value, ',')
    """

    bars = alt.Chart(plot_df).mark_bar(color="#37415f", cornerRadiusEnd=2).encode(
        y=alt.Y("Region:N", sort=plot_df["Region"].tolist(), title=None, axis=alt.Axis(labelLimit=300)),
        x=alt.X(
            f"{metric_label}:Q",
            title=None,
            axis=alt.Axis(labelExpr=compact_axis_expr, grid=True),
            scale=alt.Scale(domain=[0, padded_max], nice=False),
        ),
        tooltip=[
            alt.Tooltip("Region:N", title=region_label),
            alt.Tooltip(f"{metric_label}:Q", title=metric_label, format=","),
            alt.Tooltip("Outlet Count:Q", title="Outlets", format=","),
            alt.Tooltip("Example Outlets:N", title="Example Outlets"),
        ],
    )
    labels = alt.Chart(plot_df).mark_text(
        align="left",
        baseline="middle",
        dx=6,
        color="#f3f4f6",
        fontSize=11,
    ).encode(
        y=alt.Y("Region:N", sort=plot_df["Region"].tolist()),
        x=alt.X(f"{metric_label}:Q"),
        text=alt.Text("Metric Label:N"),
    )

    chart = (
        (bars + labels)
        .properties(height=chart_height, title=title)
        .configure_view(strokeWidth=0)
    )
    st.altair_chart(chart, use_container_width=True)


def _tail_heading_for_level(label: str) -> str:
    return {
        "Countries": "Other countries",
        "States / Provinces": "Other regions",
        "Cities": "Other cities",
    }.get(label, "Other regions")


def render_regions_page() -> None:
    st.title("Regions")
    st.caption("Preview regional distribution, choose the geography levels that matter, and generate report-ready regional observations.")

    if not st.session_state.get("standard_step", False):
        st.error("Please complete Basic Cleaning before trying this step.")
        st.stop()

    if len(st.session_state.get("df_traditional", [])) == 0:
        st.error("Traditional / online / broadcast coverage is missing. Please complete Basic Cleaning again.")
        st.stop()

    init_regions_state(st.session_state)

    def get_regions_source_df() -> pd.DataFrame:
        outlet_map = st.session_state.get("outlet_rollup_map", {})
        df_traditional = st.session_state.get("df_traditional", pd.DataFrame()).copy()
        qualitative_flags = tuple(get_qualitative_coverage_flag_exclusions(st.session_state))
        cache_key = (
            "regions_source_cache",
            3,
            len(df_traditional),
            tuple(df_traditional.columns.tolist()),
            tuple(sorted((str(k), str(v)) for k, v in outlet_map.items())),
            qualitative_flags,
            build_dataset_scope_cache_key(st.session_state),
        )
        cached = st.session_state.get("regions_source_cache")
        if cached and cached.get("key") == cache_key:
            cached_df = cached["df"].copy()
            if "Group ID" not in cached_df.columns:
                cached_df["Group ID"] = (
                    cached_df.get("Headline", pd.Series(index=cached_df.index, dtype="object"))
                    .fillna("")
                    .astype(str)
                    .str.strip()
                )
                cached_df["Group ID"] = cached_df["Group ID"].replace("", pd.NA)
                cached_df["Group ID"] = cached_df["Group ID"].fillna(
                    pd.Series(index=cached_df.index, data=[f"ROW::{idx}" for idx in cached_df.index])
                ).astype(str)
                st.session_state.regions_source_cache = {"key": cache_key, "df": cached_df}
            return cached_df

        prepared = build_regions_source_df(df_traditional, outlet_rollup_map=outlet_map)
        prepared = apply_session_coverage_flag_policy(
            prepared,
            st.session_state,
            list(qualitative_flags),
        )
        st.session_state.regions_source_cache = {"key": cache_key, "df": prepared}
        return prepared

    def get_filtered_regions_df() -> pd.DataFrame:
        source_df = get_regions_source_df()
        return filter_regions_df(
            source_df,
            exclude_coverage_flags=[],
            include_countries=st.session_state.get("regions_include_countries", []),
            exclude_countries=st.session_state.get("regions_exclude_countries", []),
        )

    def get_rankings() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        filtered_df = get_filtered_regions_df()
        metric_label = st.session_state.get("regions_metric", "Mentions")
        country_df = build_region_rankings(filtered_df, "Country", metric_label=metric_label)
        state_df = build_region_rankings(filtered_df, "State / Province", metric_label=metric_label)
        city_df = build_region_rankings(filtered_df, "City", metric_label=metric_label)
        return country_df, state_df, city_df

    current_step = st.session_state.get("regions_step", "Setup")
    if current_step not in {"Setup", "Insights"}:
        current_step = "Insights" if st.session_state.get("regions_prepared", False) else "Setup"
        st.session_state.regions_step = current_step

    st.markdown(
        """
        <style>
        .regions-step-note {
            margin: 0.15rem 0 1rem 0;
            color: rgba(250, 250, 250, 0.72);
            font-size: 0.95rem;
        }
        .regions-subtle {
            color: rgba(250, 250, 250, 0.72);
            font-size: 0.95rem;
        }
        .regions-copy-blurb {
            margin: 0 0 0.9rem 0;
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

    step1, step2 = st.columns(2, gap="small")
    with step1:
        if st.button(
            "1. Setup",
            key="regions_step_setup",
            use_container_width=True,
            type="primary" if current_step == "Setup" else "secondary",
        ):
            st.session_state.regions_step = "Setup"
            st.rerun()
    with step2:
        if st.button(
            "2. Insights",
            key="regions_step_insights",
            use_container_width=True,
            disabled=not st.session_state.get("regions_prepared", False),
            type="primary" if current_step == "Insights" else "secondary",
        ):
            st.session_state.regions_step = "Insights"
            st.rerun()

    st.markdown(
        '<div class="regions-step-note">Work left to right: review the geography first, decide which levels matter, then generate regional observations that explain what is driving the leading places.</div>',
        unsafe_allow_html=True,
    )

    source_df = get_regions_source_df()
    filtered_df = get_filtered_regions_df()
    health = build_regions_health_summary(filtered_df)
    country_df, state_df, city_df = get_rankings()
    metric_label = st.session_state.get("regions_metric", "Mentions")
    metric_col = METRIC_FIELD_MAP[metric_label]

    if current_step == "Setup":
        st.subheader("Setup")
        metric_cols = st.columns(6, gap="small")
        with metric_cols[0]:
            st.metric("Rows", f"{health['rows']:,}")
        with metric_cols[1]:
            st.metric("Usable geo", f"{health['usable_rows']:,}")
        with metric_cols[2]:
            st.metric("Missing geo", f"{health['missing_rows']:,}")
        with metric_cols[3]:
            st.metric("Countries", f"{health['countries']:,}")
        with metric_cols[4]:
            st.metric("States / provinces", f"{health['states']:,}")
        with metric_cols[5]:
            st.metric("Cities", f"{health['cities']:,}")

        st.session_state.regions_metric = st.radio(
            "Rank regions by",
            ["Mentions", "Impressions", "Effective Reach"],
            index=["Mentions", "Impressions", "Effective Reach"].index(
                st.session_state.get("regions_metric", "Mentions")
            ),
            horizontal=True,
            key="regions_metric_radio",
        )
        metric_col = METRIC_FIELD_MAP[st.session_state.regions_metric]
        country_df, state_df, city_df = get_rankings()

        preview_col1, preview_col2, preview_col3 = st.columns(3, gap="medium")
        preview_specs = [
            (preview_col1, "Countries", country_df.head(10), "Country"),
            (preview_col2, "States / provinces", state_df.head(10), "State / Province"),
            (preview_col3, "Cities", city_df.head(10), "City"),
        ]
        for container, heading, table_df, region_label in preview_specs:
            with container:
                st.write(f"**{heading}**")
                if table_df.empty:
                    st.info(f"No usable {heading.lower()} data.")
                else:
                    display_df = _format_region_table_display(table_df)
                    display_cols = ["Region", metric_col, "Outlet Count"]
                    st.dataframe(
                        display_df[display_cols],
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "Region": st.column_config.Column(region_label, width="medium"),
                            metric_col: st.column_config.TextColumn(metric_col, width="medium"),
                            "Outlet Count": st.column_config.TextColumn("Outlets", width="small"),
                        },
                    )

        st.write("**Settings**")
        settings_col1, settings_col2 = st.columns(2, gap="medium")
        available_countries = sorted([value for value in source_df["Country"].dropna().astype(str).str.strip().unique().tolist() if value])
        qualitative_flags = get_qualitative_coverage_flag_exclusions(st.session_state)
        st.session_state.regions_exclude_coverage_flags = list(qualitative_flags)
        with settings_col1:
            st.session_state.regions_analysis_levels = st.multiselect(
                "Analysis of",
                options=["Countries", "States / Provinces", "Cities"],
                default=st.session_state.get("regions_analysis_levels", ["Countries", "States / Provinces", "Cities"]),
                key="regions_analysis_levels_multiselect",
            )
            if qualitative_flags:
                st.caption(format_qualitative_exclusion_caption(qualitative_flags))
        with settings_col2:
            st.session_state.regions_include_countries = st.multiselect(
                "Include countries",
                options=available_countries,
                default=st.session_state.get("regions_include_countries", []),
                key="regions_include_countries_multiselect",
            )
            st.session_state.regions_exclude_countries = st.multiselect(
                "Exclude countries",
                options=available_countries,
                default=st.session_state.get("regions_exclude_countries", []),
                key="regions_exclude_countries_multiselect",
            )

        action1, action2 = st.columns([1, 1], gap="medium")
        with action1:
            if st.button("Prepare Regions View", type="primary", use_container_width=True):
                st.session_state.regions_prepared = True
                st.session_state.regions_step = "Insights"
                st.session_state.regions_generated_output = {}
                st.session_state.regions_generated_signature = None
                st.rerun()
        with action2:
            if st.button("Clear Settings", use_container_width=True):
                st.session_state.regions_metric = "Mentions"
                st.session_state.regions_analysis_levels = ["Countries", "States / Provinces", "Cities"]
                st.session_state.regions_exclude_coverage_flags = list(qualitative_flags)
                st.session_state.regions_include_countries = []
                st.session_state.regions_exclude_countries = []
                st.session_state.regions_prepared = False
                st.session_state.regions_generated_output = {}
                st.session_state.regions_generated_signature = None
                st.rerun()

        st.markdown(
            '<div class="regions-subtle" style="margin-top:0.65rem;">This first pass always uses rows with usable geography. The key decision here is usually which geography levels are worth analyzing in the final output.</div>',
            unsafe_allow_html=True,
        )
        return

    if not st.session_state.get("regions_prepared", False):
        st.info("Prepare the Regions view first.")
        return

    st.subheader("Insights")
    insight_stats = st.columns(4, gap="small")
    with insight_stats[0]:
        st.metric("Rows in view", f"{len(filtered_df):,}")
    with insight_stats[1]:
        st.metric("Countries in view", f"{len(country_df):,}")
    with insight_stats[2]:
        st.metric("States / provinces in view", f"{len(state_df):,}")
    with insight_stats[3]:
        st.metric("Cities in view", f"{len(city_df):,}")

    selected_levels = st.session_state.get("regions_analysis_levels", ["Countries", "States / Provinces", "Cities"])
    if not selected_levels:
        st.info("Choose at least one geography level in Setup before generating insights.")
        return

    current_signature = _build_regions_signature(st.session_state, filtered_df)
    generated_store = st.session_state.get("regions_generated_output", {})

    level_specs: list[tuple[str, pd.DataFrame, str, str]] = []
    if "Countries" in selected_levels:
        level_specs.append(("Countries", country_df, "Country", "Country"))
    if "States / Provinces" in selected_levels:
        level_specs.append(("States / Provinces", state_df, "State / Province", "State / Province"))
    if "Cities" in selected_levels:
        level_specs.append(("Cities", city_df, "City", "City"))

    for label, level_df, region_label, level_key in level_specs:
        st.divider()
        header_col, button_col = st.columns([3, 1.2], gap="medium")
        with header_col:
            st.subheader(label)
        with button_col:
            button_label = {
                "Countries": "Generate country observations",
                "States / Provinces": "Generate state / province observations",
                "Cities": "Generate city observations",
            }.get(label, f"Generate {label.lower()} observations")
            if st.button(button_label, key=f"generate_regions_{label}", type="primary", use_container_width=True):
                client_name = str(st.session_state.get("client_name", "") or "").strip()
                analysis_context = build_analysis_context_text(st.session_state)
                with st.spinner(f"Generating {label.lower()} observations..."):
                    try:
                        output, _, _ = generate_regions_insight_output(
                            client_name=client_name,
                            analysis_context=analysis_context,
                            metric_label=metric_label,
                            filtered_df=filtered_df,
                            level_label=label,
                            level_key=level_key,
                            ranking_df=level_df,
                            api_key=st.secrets["key"],
                        )
                    except Exception as exc:
                        st.error(f"Could not generate {label.lower()} observations: {exc}")
                    else:
                        updated_store = dict(st.session_state.get("regions_generated_output", {}))
                        updated_store[label] = {"signature": current_signature, "content": output.get(label, {})}
                        st.session_state.regions_generated_output = updated_store
                        generated_store = updated_store

        if level_df.empty:
            st.info(f"No {label.lower()} data is available for the current filters.")
            continue

        chart_tab, table_tab, copy_tab = st.tabs(["Chart", "Table", "Report Copy"])

        with chart_tab:
            _render_region_chart(level_df, metric_label, f"Top {label} by {metric_label.lower()}", region_label)

        with table_tab:
            display_df = _format_region_table_display(level_df)
            st.dataframe(
                display_df.head(15)[["Region", "Mentions", "Impressions", "Effective Reach", "Outlet Count", "Example Outlets"]],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Region": st.column_config.Column(region_label, width="medium"),
                    "Mentions": st.column_config.TextColumn("Mentions", width="small"),
                    "Impressions": st.column_config.TextColumn("Impressions", width="medium"),
                    "Effective Reach": st.column_config.TextColumn("Effective Reach", width="medium"),
                    "Outlet Count": st.column_config.TextColumn("Outlets", width="small"),
                    "Example Outlets": st.column_config.Column("Example Outlets", width="large"),
                },
            )

        with copy_tab:
            level_entry = generated_store.get(label, {})
            level_is_current = False
            level_copy: dict[str, object] = {}
            if isinstance(level_entry, dict) and "content" in level_entry:
                level_is_current = level_entry.get("signature") == current_signature
                level_copy = level_entry.get("content", {}) or {}
            elif isinstance(level_entry, dict):
                level_copy = level_entry
            if level_copy and not level_is_current:
                st.info("Setup has changed since these observations were generated. Regenerate this section to refresh the copy.")
            if not level_copy:
                st.info("Generate this section to create report-copy observations for this geography level.")
            else:
                overall_observation = str(level_copy.get("overall_observation", "") or "").strip()
                if overall_observation:
                    st.write("**Overall observation**")
                    st.markdown(f'<div class="regions-copy-blurb">{overall_observation}</div>', unsafe_allow_html=True)

                top_profiles = [item for item in level_copy.get("top_region_profiles", []) if str(item.get("blurb", "")).strip()]
                if top_profiles:
                    st.write("**Top region profiles**")
                    for item in top_profiles:
                        region_name = str(item.get("region", "") or "").strip()
                        blurb = str(item.get("blurb", "") or "").strip()
                        if not region_name or not blurb:
                            continue
                        st.markdown(f"**{region_name}**")
                        st.markdown(f'<div class="regions-copy-blurb">{blurb}</div>', unsafe_allow_html=True)
                        with st.expander(f"Supporting grouped stories for {region_name}", expanded=False):
                            st.dataframe(
                                _format_region_table_display(
                                    build_region_story_group_examples(
                                        filtered_df,
                                        level_key,
                                        region_name,
                                        metric_label=metric_label,
                                        limit=10,
                                    )
                                ),
                                use_container_width=True,
                                hide_index=True,
                            )

                tail_observation = str(level_copy.get("tail_observation", "") or "").strip()
                if tail_observation:
                    st.write(f"**{_tail_heading_for_level(label)}**")
                    st.markdown(f'<div class="regions-copy-blurb">{tail_observation}</div>', unsafe_allow_html=True)
