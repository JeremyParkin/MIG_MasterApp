from __future__ import annotations

def render_top_stories_selection() -> None:
    # 6-Top_Stories.py
    
    import importlib
    import warnings
    
    import pandas as pd
    import streamlit as st
    
    from processing.top_stories import (
        normalize_top_stories_df,
        apply_filters,
        build_grouped_story_candidates,
        save_selected_rows,
        remove_saved_candidates_from_display,
        reset_generated_candidates,
    )
    
    warnings.filterwarnings("ignore")
    
    title_col, chart_col = st.columns([2, 3], gap="medium")
    
    with title_col:
        st.subheader("Step 1: Top Story Selection")
        st.caption("Filter the grouped coverage set, inspect candidates, and save the final top stories.")
    
    if not st.session_state.get("standard_step", False):
        st.error("Please complete Basic Cleaning before trying this step.")
        st.stop()
    
    if "df_ai_grouped" not in st.session_state or st.session_state.df_ai_grouped.empty:
        st.error("Grouped story data is missing. Please complete Basic Cleaning again.")
        st.stop()
    
    # Session state init
    df_vars = ["filtered_df", "df_grouped", "selected_df", "selected_rows", "top_stories", "added_df"]
    for var in df_vars:
        if var not in st.session_state:
            st.session_state[var] = pd.DataFrame()
    
    if "top_stories_generated" not in st.session_state:
        st.session_state.top_stories_generated = False
    
    if not st.session_state.added_df.empty:
        st.session_state.added_df = normalize_top_stories_df(st.session_state.added_df)
    
    source_df = normalize_top_stories_df(st.session_state.df_ai_grouped.copy())
    
    with chart_col:
        trend_df = (
            st.session_state.filtered_df.copy()
            if st.session_state.top_stories_generated and not st.session_state.filtered_df.empty
            else source_df.copy()
        )
    
        if not trend_df.empty and "Date" in trend_df.columns:
            trend_df["Date"] = pd.to_datetime(trend_df["Date"], errors="coerce")
            trend_df = trend_df.dropna(subset=["Date"]).copy()
    
            if not trend_df.empty:
                summary_stats = (
                    trend_df.groupby(pd.Grouper(key="Date", freq="D"))
                    .agg({"Mentions": "sum", "Impressions": "sum"})
                    .reset_index()
                    .sort_values("Date")
                )
    
                if not summary_stats.empty:
                    show_time = False
                    date_span = summary_stats["Date"].max() - summary_stats["Date"].min()
                    show_time = date_span <= pd.Timedelta(days=1)
    
                    try:
                        alt = importlib.import_module("altair")
                    except Exception:
                        st.info("Top stories trend chart unavailable in this environment.")
                        alt = None
    
                    if alt is not None:
                        x_axis = alt.Axis(
                            title=None,
                            labelAngle=0,
                            format="%b %d, %-I %p" if show_time else "%b %d",
                        )
    
                        line = alt.Chart(summary_stats).mark_line(size=2).encode(
                            x=alt.X("Date:T", axis=x_axis),
                            y=alt.Y("Mentions:Q", axis=None),
                        )
    
                        points = alt.Chart(summary_stats).mark_circle(size=55, opacity=0).encode(
                            x="Date:T",
                            y="Mentions:Q",
                            tooltip=[
                                alt.Tooltip(
                                    "Date:T",
                                    title="Date",
                                    format="%b %d, %Y" if not show_time else "%b %d, %Y %-I:%M %p",
                                ),
                                alt.Tooltip("Mentions:Q", title="Mentions", format=","),
                            ],
                        )
    
                        chart = (line + points).properties(height=130)
                        st.altair_chart(chart, use_container_width=True)
    
    all_columns = list(source_df.columns)
    columns_to_keep = [
        "Group ID",
        "Headline",
        "Date",
        "Mentions",
        "Impressions",
        "Type",
        "Outlet",
        "URL",
        "Snippet",
        "Tags",
        "Coverage Flags",
        "Prime Example"
    ]
    
    extra_filter_columns = [
        col for col in all_columns
        if col.lower().startswith("tag")
        or "tag group" in col.lower()
        or "prominence" in col.lower()
    ]
    columns_to_keep.extend(extra_filter_columns)
    
    existing_columns = [col for col in columns_to_keep if col in source_df.columns]
    source_df = source_df[existing_columns].copy()
    source_df = normalize_top_stories_df(source_df)
    
    available_types = sorted([t for t in source_df["Type"].dropna().astype(str).unique().tolist() if t])
    available_flags = sorted([f for f in source_df["Coverage Flags"].dropna().astype(str).unique().tolist() if f])
    
    advanced_filter_columns = []
    for col in source_df.columns:
        col_lower = col.lower()
        if (
            col in ["Headline", "Outlet", "Coverage Flags", "Tags", "Language", "Country"]
            or col_lower.startswith("tag group:")
            or col_lower.startswith("tag ")
            or "tag group" in col_lower
            or "prominence" in col_lower
        ):
            advanced_filter_columns.append(col)
    
    advanced_filter_columns = list(dict.fromkeys(advanced_filter_columns))
    advanced_filter_columns.sort()
    
    min_available_date = source_df["Date"].min() if source_df["Date"].notna().any() else None
    max_available_date = source_df["Date"].max() if source_df["Date"].notna().any() else None
    
    with st.form(key="top_stories_filter_form"):
        filter_col1, filter_col2, filter_col3 = st.columns([2.5, 2, 4], gap="medium")
    
        with filter_col1:
            if min_available_date and max_available_date:
                date_range = st.date_input(
                    "Date range",
                    value=(min_available_date, max_available_date),
                    min_value=min_available_date,
                    max_value=max_available_date,
                )
            else:
                date_range = ()
    
        with filter_col2:
            exclude_types = st.multiselect(
                "Exclude media types",
                options=available_types,
                default=[],
            )
    
        with filter_col3:
            default_excluded_flags = [
                f for f in [
                    "Newswire?",
                    "Market Report Spam?",
                    "Stocks / Financials?",
                    "Advertorial?",
                    "User-Generated",
                ] if f in available_flags
            ]
    
            hidden_flags = {"Good Outlet", "Aggregator"}
            visible_flags = [f for f in available_flags if f not in hidden_flags]
            visible_defaults = [f for f in default_excluded_flags if f not in hidden_flags]
    
            exclude_coverage_flags = st.multiselect(
                "Exclude coverage flags",
                options=visible_flags,
                default=visible_defaults,
            )
    
        with st.expander("Advanced filters", expanded=False):
            adv1_col1, adv1_col2 = st.columns([2, 5], gap="small")
            with adv1_col1:
                adv_col_1 = st.selectbox("Condition 1 column", options=[""] + advanced_filter_columns, index=0)
            with adv1_col2:
                adv_value_1 = st.text_input(
                    "Condition 1 query",
                    help='Use boolean syntax like: ("Class action" OR lawsuit) AND NOT "dismissed"',
                )
    
            adv2_col1, adv2_col2 = st.columns([2, 5], gap="small")
            with adv2_col1:
                adv_col_2 = st.selectbox("Condition 2 column", options=[""] + advanced_filter_columns, index=0)
            with adv2_col2:
                adv_value_2 = st.text_input(
                    "Condition 2 query",
                    help='Use boolean syntax like: (High OR Moderate) AND NOT Low',
                )
    
            adv3_col1, adv3_col2 = st.columns([2, 5], gap="small")
            with adv3_col1:
                adv_col_3 = st.selectbox("Condition 3 column", options=[""] + advanced_filter_columns, index=0)
            with adv3_col2:
                adv_value_3 = st.text_input(
                    "Condition 3 query",
                    help='Use boolean syntax like: Canada OR USA',
                )
    
        action_col1, action_col2 = st.columns([2, 1], gap="small")
        with action_col1:
            generate_candidates = st.form_submit_button("Generate Possible Top Stories", type="primary")
        with action_col2:
            clear_generated = st.form_submit_button("Clear Generated Results")
    
    if clear_generated:
        reset_generated_candidates(st.session_state)
        st.rerun()
    
    if generate_candidates:
        if isinstance(date_range, tuple) and len(date_range) == 2:
            start_date, end_date = date_range
        else:
            start_date, end_date = None, None
    
        advanced_filters = [
            {"column": adv_col_1, "value": adv_value_1},
            {"column": adv_col_2, "value": adv_value_2},
            {"column": adv_col_3, "value": adv_value_3},
        ]
    
        filtered_df = apply_filters(
            df=source_df,
            start_date=start_date,
            end_date=end_date,
            exclude_types=exclude_types,
            exclude_coverage_flags=exclude_coverage_flags,
            advanced_filters=advanced_filters,
        )
    
        st.session_state.filtered_df = filtered_df.copy()
        st.session_state.df_grouped = build_grouped_story_candidates(filtered_df)
        st.session_state.df_grouped = st.session_state.df_grouped.sort_values(
            by=["Mentions", "Impressions"],
            ascending=[False, False],
        )
        st.session_state.top_stories_generated = True
    
    if st.session_state.top_stories_generated:
        filtered_count = len(st.session_state.filtered_df) if not st.session_state.filtered_df.empty else 0
        grouped_count = len(st.session_state.df_grouped) if not st.session_state.df_grouped.empty else 0
    
        st.caption(f"{filtered_count:,} mentions matched filters, producing {grouped_count:,} grouped story candidates.")
    
        df_to_display = st.session_state.df_grouped.copy()
    
        if not df_to_display.empty:
            df_to_display["Date"] = pd.to_datetime(df_to_display["Date"], errors="coerce").dt.date
            df_to_display["Example URL"] = df_to_display["Example URL"].fillna("").astype(str)
            df_to_display["Impressions"] = pd.to_numeric(df_to_display["Impressions"], errors="coerce").fillna(0).astype(int)
            df_to_display = df_to_display.sort_values(
                by=["Mentions", "Impressions"],
                ascending=[False, False],
            )
    
        df_to_display = remove_saved_candidates_from_display(
            df_to_display,
            st.session_state.added_df,
        )
    
        if df_to_display.empty:
            st.info("No story candidates matched the selected filters.")
        else:
            df_to_display["Top Story"] = False
    
            # Keep checkbox on the right, but control widths
            preferred_order = [
                "Headline",
                "Date",
                "Mentions",
                "Impressions",
                "Example URL",
                "Example Snippet",
                "Top Story",
            ]
    
            existing_order = [col for col in preferred_order if col in df_to_display.columns]
            remaining_cols = [col for col in df_to_display.columns if col not in existing_order]
    
            df_to_display = df_to_display[existing_order + remaining_cols]
    
            st.subheader(
                "Possible Top Stories",
                help='Check the "Top Story" box for those stories you want to select, then click "Save Selected" below.',
            )
    
            updated_data_custom = st.data_editor(
                df_to_display,
                key="df_by_custom",
                use_container_width=True,
                column_config={
                    "Headline": st.column_config.Column("Headline", width="large"),
                    "Date": st.column_config.Column("Date", width="small"),
                    "Mentions": st.column_config.Column("Mentions", width="small"),
                    "Impressions": st.column_config.NumberColumn("Impressions", width="small", format="%,d"),
    
                    # aggressively shrink these
                    "Example URL": st.column_config.LinkColumn("Example URL", width="small"),
                    "Example Snippet": st.column_config.Column("Example Snippet", width="small"),
    
                    # keep checkbox visible on right
                    "Top Story": st.column_config.CheckboxColumn("Top Story", width="small"),
    
                    # hide internals
                    "Group ID": None,
                    "Example Outlet": None,
                    "Example Type": None,
                },
                hide_index=True,
            )
    
    
            if st.button("Save Selected", key="by_custom", type="primary"):
                st.session_state.added_df = save_selected_rows(
                    updated_data_custom,
                    df_to_display,
                    st.session_state.added_df,
                )
    
                st.rerun()
    
    if len(st.session_state.added_df) > 0:
        st.subheader("Saved Top Stories")
    
        saved_df_full = st.session_state.added_df.copy()
        saved_df_full = normalize_top_stories_df(saved_df_full)
        saved_df_full = saved_df_full.sort_values(by="Date", ascending=True).reset_index(drop=True)
    
        saved_df_display = saved_df_full.copy()
    
        saved_columns = ["Headline", "Date", "Mentions", "Impressions", "Example URL"]
        existing_saved_columns = [col for col in saved_columns if col in saved_df_display.columns]
        saved_df_display = saved_df_display[existing_saved_columns].copy()
    
        saved_df_display["Delete"] = False
    
        if "Date" in saved_df_display.columns:
            date_column = saved_df_display.pop("Date")
            saved_df_display.insert(1, "Date", date_column)
    
        updated_data = st.data_editor(
            saved_df_display,
            use_container_width=True,
            column_config={
                "Delete": st.column_config.CheckboxColumn("Delete", width="small"),
                "Headline": st.column_config.Column("Headline", width="large"),
                "Date": st.column_config.Column("Date", width="small"),
                "Mentions": st.column_config.Column("Mentions", width="small"),
                "Impressions": st.column_config.NumberColumn("Impressions", width="small", format="%,d"),
                "Example URL": st.column_config.LinkColumn("Example URL", width="medium"),
            },
            hide_index=True,
            key="saved_stories_editor",
        )
    
        rows_to_delete = updated_data[updated_data["Delete"]].index.tolist()
    
        if rows_to_delete:
            st.session_state.added_df = (
                saved_df_full.drop(index=rows_to_delete)
                .reset_index(drop=True)
            )
            st.rerun()
