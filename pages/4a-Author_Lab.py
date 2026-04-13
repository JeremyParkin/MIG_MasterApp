from __future__ import annotations

import warnings

import pandas as pd
import streamlit as st

from processing.author_insights import (
    DEFAULT_AUTHOR_SUMMARY_MODEL,
    build_author_headline_table,
    build_author_metrics,
    generate_author_summary,
    init_author_lab_state,
    infer_author_themes,
)
from processing.author_outlets import (
    FORMAT_DICT,
    assign_outlet,
    build_auth_outlet_table,
    build_outlet_assignment_payload,
    fetch_outlet,
    get_auth_outlet_todo,
    get_author_search_urls,
    get_matched_authors_df,
    get_outlets_in_coverage,
    get_search_author_name,
    init_author_outlets_state,
    prepare_traditional_for_author_outlets,
    reset_outlet_skips,
    undo_last_outlet_assignment,
)

warnings.filterwarnings("ignore")

st.title("Author Lab")
st.caption("Experimental author workflow: outlet assignment plus author intelligence, shortlist curation, and AI summaries.")

if not st.session_state.get("standard_step", False):
    st.error("Please complete Basic Cleaning before trying this step.")
    st.stop()

init_author_outlets_state(st.session_state)
init_author_lab_state(st.session_state)

if st.session_state.get("pickle_load", False) is True and len(st.session_state.get("auth_outlet_table", [])) > 0:
    st.session_state.auth_outlet_table = st.session_state.auth_outlet_table.copy()
    st.session_state.auth_outlet_table["Outlet"] = st.session_state.auth_outlet_table["Outlet"].replace([None], "").fillna("")

st.session_state.df_traditional = prepare_traditional_for_author_outlets(st.session_state.df_traditional)

controls_col1, controls_col2, controls_col3 = st.columns([1.2, 1, 1], gap="medium")
with controls_col1:
    st.session_state.top_auths_by = st.selectbox(
        "Rank authors by",
        ["Mentions", "Impressions"],
        index=0 if st.session_state.get("top_auths_by", "Mentions") == "Mentions" else 1,
        on_change=lambda: reset_outlet_skips(st.session_state),
    )
with controls_col2:
    shortlist_target = st.number_input(
        "Shortlist target",
        min_value=5,
        max_value=25,
        value=int(st.session_state.get("author_lab_target_count", 10)),
        step=1,
    )
    st.session_state.author_lab_target_count = shortlist_target
with controls_col3:
    if st.button("Undo Last Outlet", disabled=not st.session_state.get("last_outlet_assignment")):
        undo_last_outlet_assignment(st.session_state)
        st.rerun()

if len(st.session_state.get("auth_outlet_table", pd.DataFrame())) == 0:
    st.session_state.auth_outlet_table = build_auth_outlet_table(
        st.session_state.df_traditional.copy(),
        st.session_state.get("top_auths_by", "Mentions"),
    )
else:
    st.session_state.auth_outlet_table = build_auth_outlet_table(
        st.session_state.df_traditional.copy(),
        st.session_state.get("top_auths_by", "Mentions"),
        existing_assignments=st.session_state.auth_outlet_table.copy(),
    )

author_metrics, author_story_rows = build_author_metrics(
    st.session_state.df_traditional,
    auth_outlet_table=st.session_state.auth_outlet_table,
)

if author_metrics.empty:
    st.info("No author data available yet.")
    st.stop()

auth_outlet_todo = get_auth_outlet_todo(st.session_state.auth_outlet_table)

left_col, right_col = st.columns([1.05, 1.35], gap="large")

with left_col:
    st.subheader("Outlet Assignment Queue")

    if auth_outlet_todo.empty:
        st.success("All authors in the current queue have assigned outlets.")
    else:
        current_idx = min(st.session_state.auth_outlet_skipped, len(auth_outlet_todo) - 1)
        current_author = auth_outlet_todo.iloc[current_idx]["Author"]

        top_bar1, top_bar2, top_bar3 = st.columns(3)
        with top_bar1:
            st.metric("Remaining", len(auth_outlet_todo) - current_idx)
        with top_bar2:
            if st.button("Skip Author"):
                st.session_state.auth_outlet_skipped = min(current_idx + 1, len(auth_outlet_todo))
                st.rerun()
        with top_bar3:
            if st.button("Reset Queue"):
                st.session_state.auth_outlet_skipped = 0
                st.rerun()

        st.markdown(f"### {current_author}")

        search_author_name = get_search_author_name(current_author)
        search_results, api_debug = fetch_outlet(search_author_name, st.secrets)
        outlets_in_coverage = get_outlets_in_coverage(st.session_state.df_traditional, current_author)
        outlets_in_coverage_list = pd.Index(outlets_in_coverage["Outlet"].tolist()).insert(0, "Freelance")
        matched_authors, db_outlets, possibles = get_matched_authors_df(search_results, outlets_in_coverage_list)

        exp1, exp2 = st.columns([1, 1])
        with exp1:
            st.write("**Outlets seen in coverage**")
            st.dataframe(outlets_in_coverage, use_container_width=True, hide_index=True)
        with exp2:
            st.write("**Database matches**")
            if matched_authors.empty:
                st.info("No direct database match found.")
                if api_debug.get("error"):
                    with st.expander("API debug"):
                        st.write(api_debug.get("error"))
            else:
                st.dataframe(matched_authors, use_container_width=True, hide_index=True)

        muckrack_url, linkedin_url = get_author_search_urls(current_author)
        st.markdown(f"[Search Muckrack]({muckrack_url})")
        st.markdown(f"[Search LinkedIn]({linkedin_url})")

        with st.form("author_lab_assign_outlet", clear_on_submit=True):
            form_col1, form_col2 = st.columns(2)
            with form_col1:
                if len(matched_authors) > 0:
                    selected_outlet = st.selectbox("Choose an outlet", possibles)
                else:
                    selected_outlet = st.selectbox("Choose an outlet", outlets_in_coverage_list)
            with form_col2:
                manual_outlet = st.text_input("Or type a custom outlet")

            submitted = st.form_submit_button("Assign Outlet", type="primary")

        if submitted:
            new_outlet = manual_outlet.strip() if manual_outlet.strip() else selected_outlet
            st.session_state.last_outlet_assignment = build_outlet_assignment_payload(
                st.session_state.auth_outlet_table,
                current_author,
                previous_skip=current_idx,
            )
            st.session_state.auth_outlet_table = assign_outlet(
                st.session_state.auth_outlet_table,
                current_author,
                new_outlet,
            )
            st.rerun()

with right_col:
    st.subheader("Author Intelligence")

    default_author = auth_outlet_todo.iloc[min(st.session_state.auth_outlet_skipped, len(auth_outlet_todo) - 1)]["Author"] if not auth_outlet_todo.empty else author_metrics.iloc[0]["Author"]
    inspected_author = st.selectbox(
        "Inspect author",
        options=author_metrics["Author"].tolist(),
        index=author_metrics["Author"].tolist().index(default_author) if default_author in author_metrics["Author"].tolist() else 0,
    )

    author_row = author_metrics.loc[author_metrics["Author"] == inspected_author].iloc[0]
    themes = infer_author_themes(author_story_rows, inspected_author)
    headline_table = build_author_headline_table(author_story_rows, inspected_author)

    metrics_row1, metrics_row2, metrics_row3, metrics_row4 = st.columns(4)
    with metrics_row1:
        st.metric("Unique stories", int(author_row["Unique_Stories"]))
    with metrics_row2:
        st.metric("Syndicated pickups", int(author_row["Syndicated_Pickups"]))
    with metrics_row3:
        st.metric("Mentions", int(author_row["Mention_Total"]))
    with metrics_row4:
        st.metric("Good-outlet stories", int(author_row["Good_Outlet_Stories"]))

    st.caption(
        f"Assigned outlet: {author_row.get('Assigned Outlet', '') or 'Unassigned'} | "
        f"Coverage primary outlet: {author_row.get('Coverage_Primary_Outlet', '') or 'Unknown'} | "
        f"Likely themes: {', '.join(themes) if themes else 'Not enough signal'}"
    )

    st.dataframe(headline_table, use_container_width=True, hide_index=True)

st.divider()
st.subheader("Curated Final Authors")
st.caption("Pick the authors who should make the final report. This goes beyond raw volume by showing story shape and outlet quality.")

rank_map = {
    "Mentions": ["Mention_Total", "Impressions", "Unique_Stories"],
    "Impressions": ["Impressions", "Mention_Total", "Unique_Stories"],
}
sort_cols = rank_map.get(st.session_state.top_auths_by, rank_map["Mentions"])
curation_df = author_metrics.copy().sort_values(sort_cols, ascending=False).reset_index(drop=True)
selected_lookup = set(st.session_state.get("author_lab_selected_authors", []))
curation_df["Selected"] = curation_df["Author"].isin(selected_lookup)

display_df = curation_df[[
    "Selected",
    "Author",
    "Assigned Outlet",
    "Coverage_Primary_Outlet",
    "Unique_Stories",
    "Syndicated_Pickups",
    "Good_Outlet_Stories",
    "Mention_Total",
    "Impressions",
    "Syndication Ratio",
]].copy()

edited = st.data_editor(
    display_df,
    use_container_width=True,
    hide_index=True,
    disabled=[
        "Author",
        "Assigned Outlet",
        "Coverage_Primary_Outlet",
        "Unique_Stories",
        "Syndicated_Pickups",
        "Good_Outlet_Stories",
        "Mention_Total",
        "Impressions",
        "Syndication Ratio",
    ],
    column_config={
        "Selected": st.column_config.CheckboxColumn("Final list"),
        "Syndication Ratio": st.column_config.NumberColumn("Syndication ratio", format="%.2f"),
    },
)

selected_authors = edited.loc[edited["Selected"], "Author"].tolist()
st.session_state.author_lab_selected_authors = selected_authors

shortlist_col1, shortlist_col2, shortlist_col3 = st.columns([1.3, 1.3, 3])
with shortlist_col1:
    st.metric("Selected authors", len(selected_authors))
with shortlist_col2:
    target_delta = len(selected_authors) - int(shortlist_target)
    st.metric("Vs target", target_delta)
with shortlist_col3:
    if selected_authors:
        st.caption("Current shortlist: " + ", ".join(selected_authors[:12]) + ("..." if len(selected_authors) > 12 else ""))

st.divider()
st.subheader("AI Author Summaries")
st.caption("Generate short report-ready summaries for the shortlisted authors from their grouped coverage patterns.")

generate_disabled = len(selected_authors) == 0
if st.button("Generate AI summaries for selected authors", type="primary", disabled=generate_disabled):
    summaries = dict(st.session_state.get("author_lab_summaries", {}))
    errors: list[str] = []

    with st.spinner("Generating author summaries..."):
        for author_name in selected_authors:
            author_row = author_metrics.loc[author_metrics["Author"] == author_name].iloc[0]
            headline_df = build_author_headline_table(author_story_rows, author_name, limit=8)
            try:
                summary_text, _, _ = generate_author_summary(
                    author_name=author_name,
                    author_row=author_row,
                    headline_df=headline_df,
                    api_key=st.secrets["key"],
                    model=DEFAULT_AUTHOR_SUMMARY_MODEL,
                )
                summaries[author_name] = summary_text
            except Exception as e:
                errors.append(f"{author_name}: {e}")

    st.session_state.author_lab_summaries = summaries
    if errors:
        for err in errors:
            st.warning(err)

summary_store = st.session_state.get("author_lab_summaries", {})
if selected_authors:
    for author_name in selected_authors:
        author_row = author_metrics.loc[author_metrics["Author"] == author_name].iloc[0]
        assigned = author_row.get("Assigned Outlet", "") or author_row.get("Coverage_Primary_Outlet", "") or "Unknown"
        with st.expander(f"{author_name} | {assigned}", expanded=False):
            st.write(summary_store.get(author_name, "No AI summary generated yet."))
