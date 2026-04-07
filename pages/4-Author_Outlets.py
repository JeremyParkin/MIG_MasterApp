from __future__ import annotations

import warnings

import pandas as pd
import streamlit as st

from processing.author_outlets import (
    FORMAT_DICT,
    init_author_outlets_state,
    undo_last_outlet_assignment,
    reset_outlet_skips,
    fetch_outlet,
    prepare_traditional_for_author_outlets,
    build_auth_outlet_table,
    get_auth_outlet_todo,
    apply_author_name_fix,
    get_matched_authors_df,
    get_outlets_in_coverage,
    build_outlet_assignment_payload,
    assign_outlet,
    get_author_search_urls,
    get_search_author_name,
)
from utils.dataframe_helpers import top_x_by_mentions

warnings.filterwarnings("ignore")

st.title("Author - Outlets")

if not st.session_state.get("standard_step", False):
    st.error("Please complete Basic Cleaning before trying this step.")
    st.stop()

init_author_outlets_state(st.session_state)

if st.session_state.get("pickle_load", False) is True and len(st.session_state.auth_outlet_table) > 0:
    st.session_state.auth_outlet_table = st.session_state.auth_outlet_table.copy()
    st.session_state.auth_outlet_table["Outlet"] = st.session_state.auth_outlet_table["Outlet"].replace([None], "").fillna("")

st.session_state.df_traditional = prepare_traditional_for_author_outlets(
    st.session_state.df_traditional
)

hide_table_row_index = """
    <style>
    tbody th {display:none}
    .blank {display:none}
    .row_heading.level0 {width:0; display:none}
    </style>
"""
st.markdown(hide_table_row_index, unsafe_allow_html=True)

st.session_state.top_auths_by = st.selectbox(
    "Top Authors by:",
    ["Mentions", "Impressions"],
    on_change=lambda: reset_outlet_skips(st.session_state),
)

if len(st.session_state.auth_outlet_table) == 0:
    st.session_state.auth_outlet_table = build_auth_outlet_table(
        st.session_state.df_traditional.copy(),
        st.session_state.top_auths_by,
    )
else:
    st.session_state.auth_outlet_table = build_auth_outlet_table(
        st.session_state.df_traditional.copy(),
        st.session_state.top_auths_by,
        existing_assignments=st.session_state.auth_outlet_table.copy(),
    )

auth_outlet_todo = get_auth_outlet_todo(st.session_state.auth_outlet_table)

if st.session_state.auth_outlet_skipped < len(auth_outlet_todo):
    original_author_name = auth_outlet_todo.iloc[st.session_state.auth_outlet_skipped]["Author"]

    if st.session_state.get("last_author_for_fix") != original_author_name:
        st.session_state.author_fix_input = original_author_name
        st.session_state.last_author_for_fix = original_author_name

    def apply_author_fix_callback():
        new_name = st.session_state.author_fix_input.strip()
        old_name = original_author_name

        if not new_name:
            return

        if new_name != old_name:
            apply_author_name_fix(st.session_state, old_name, new_name)

    with st.expander("Author name fix tools", expanded=False):
        st.text_input(
            "Correct author name",
            key="author_fix_input",
            on_change=apply_author_fix_callback,
            help="Edit the name and press Enter to apply the correction to all matching rows.",
        )
        st.caption(
            "This updates every instance of this author in the cleaned dataset and refreshes the author-outlet workflow."
        )

    header_col, skip_col, reset_col, undo_col = st.columns([2, 1, 1, 1])

    with header_col:
        st.markdown(
            f"""
            <h2 style="color: goldenrod; padding-top:0!important; margin-top:0;">
                {original_author_name}
            </h2>
            """,
            unsafe_allow_html=True,
        )

    with skip_col:
        st.write(" ")
        next_auth = st.button("Skip to Next Author")
        if next_auth:
            st.session_state.auth_outlet_skipped += 1
            st.rerun()

    with reset_col:
        st.write(" ")
        reset_counter = st.button("Reset Skips")
        if reset_counter:
            st.session_state.auth_outlet_skipped = 0
            st.rerun()

    with undo_col:
        st.write(" ")
        undo_available = st.session_state.get("last_outlet_assignment") is not None
        undo_clicked = st.button(
            "Undo Last Outlet",
            disabled=not undo_available,
            help="Removes the most recently assigned outlet and returns that author to the queue.",
        )
        if undo_clicked:
            undo_last_outlet_assignment(st.session_state)
            st.rerun()

    match_author_name = original_author_name
    search_author_name = get_search_author_name(match_author_name)
    search_results, api_debug = fetch_outlet(search_author_name, st.secrets)

    def name_match(series):
        non_match = "color: #985331;"
        match = "color: goldenrod"
        return [non_match if cell_value != match_author_name else match for cell_value in series]

    outlets_in_coverage = get_outlets_in_coverage(
        st.session_state.df_traditional,
        original_author_name,
    )

    outlets_in_coverage_list = pd.Index(outlets_in_coverage["Outlet"].tolist())
    outlets_in_coverage_list = outlets_in_coverage_list.insert(0, "Freelance")

    matched_authors, db_outlets, possibles = get_matched_authors_df(
        search_results=search_results,
        outlets_in_coverage_list=outlets_in_coverage_list,
    )

    form_block = st.container()
    info_block = st.container()

    with info_block:
        col1, col2, col3 = st.columns([8, 1, 16])

        with col1:
            st.subheader("Outlets in CSV")

            outlets_in_coverage_styled = outlets_in_coverage.style.apply(
                lambda x: [
                    "background-color: goldenrod; color: black" if v in db_outlets else ""
                    for v in x
                ],
                axis=1,
                subset="Outlet",
            )

            if len(outlets_in_coverage) > 7:
                st.dataframe(
                    outlets_in_coverage_styled,
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.dataframe(
                    outlets_in_coverage_styled,
                    use_container_width=True,
                    hide_index=True,
                )

        with col2:
            st.write(" ")

        with col3:
            st.subheader("Media Database Results")

            show_debug = not api_debug.get("ok") or api_debug.get("error")

            if len(matched_authors) == 0:
                st.warning("NO MATCH FOUND")

                if show_debug:
                    with st.expander("API debug details", expanded=False):
                        st.write("Status code:", api_debug.get("status_code"))
                        st.write("Request ok:", api_debug.get("ok"))
                        st.write("Error:", api_debug.get("error"))
                        st.write("JSON keys:", api_debug.get("json_keys"))
                        st.write("Response preview:")
                        st.code(api_debug.get("response_text_preview", ""), language="json")
            else:
                coverage_outlet_values = outlets_in_coverage["Outlet"].tolist()
                matched_authors_display = matched_authors.copy()

                preferred_cols = [c for c in ["Name", "Outlet", "Country", "Title"] if
                                  c in matched_authors_display.columns]
                matched_authors_display = matched_authors_display[preferred_cols]

                styled_matches = (
                    matched_authors_display.style
                    .apply(
                        lambda col: [
                            "background-color: goldenrod; color: black"
                            if v in coverage_outlet_values else ""
                            for v in col
                        ],
                        subset=["Outlet"],
                        axis=0,
                    )
                    .apply(name_match, axis=0, subset=["Name"])
                )

                # styled_matches = (
                #     matched_authors.style
                #     .apply(
                #         lambda x: [
                #             "background: goldenrod; color: black"
                #             if v in outlets_in_coverage["Outlet"].tolist() else ""
                #             for v in x
                #         ],
                #         axis=1,
                #     )
                #     .apply(name_match, axis=0, subset="Name")
                # )

                st.dataframe(
                    styled_matches,
                    use_container_width=True,
                    hide_index=True,
                )

            muckrack_url, linkedin_url = get_author_search_urls(match_author_name)

            st.markdown(
                f'&nbsp;&nbsp;» <a href="{muckrack_url}" target="_blank" style="text-decoration:underline; color:lightblue;">Search Muckrack for {match_author_name}</a>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'&nbsp;&nbsp;» <a href="{linkedin_url}" target="_blank" style="text-decoration:underline; color:lightblue;">Search LinkedIn for {match_author_name}</a>',
                unsafe_allow_html=True,
            )

    with form_block:
        with st.form("auth_updater", clear_on_submit=True):
            col1, col2, col3 = st.columns([8, 1, 8])

            with col1:
                if len(matched_authors) > 0:
                    box_outlet = st.selectbox(
                        "Pick outlet from DATABASE MATCHES",
                        possibles,
                        help="Pick from one of the outlets associated with this author name.",
                    )
                else:
                    box_outlet = st.selectbox(
                        'Pick outlet from COVERAGE or "Freelance"',
                        outlets_in_coverage_list,
                    )

            with col2:
                st.write(" ")
                st.subheader("OR")

            with col3:
                string_outlet = st.text_input(
                    "Write in an outlet name",
                    help="Override the selection by writing a custom outlet name.",
                )

            submitted = st.form_submit_button("Assign Outlet", type="primary")

    if submitted:
        new_outlet = string_outlet.strip() if len(string_outlet.strip()) > 0 else box_outlet

        st.session_state.last_outlet_assignment = build_outlet_assignment_payload(
            st.session_state.auth_outlet_table,
            original_author_name,
            previous_skip=st.session_state.auth_outlet_skipped,
        )

        st.session_state.auth_outlet_table = assign_outlet(
            st.session_state.auth_outlet_table,
            original_author_name,
            new_outlet,
        )

        st.rerun()

    st.divider()

    bottom_col1, bottom_col2, bottom_col3 = st.columns([8, 1, 4])

    with bottom_col1:
        st.subheader("Top Authors")

        table_df = st.session_state.auth_outlet_table[["Author", "Outlet", "Mentions", "Impressions"]].copy()
        table_df = table_df.fillna("")

        if st.session_state.top_auths_by == "Mentions":
            table_df = table_df.sort_values(["Mentions", "Impressions"], ascending=False).head(15)
        else:
            table_df = table_df.sort_values(["Impressions", "Mentions"], ascending=False).head(15)

        st.dataframe(
            table_df.style.format(FORMAT_DICT, na_rep=" "),
            use_container_width=True,
            hide_index=True,
        )

    with bottom_col2:
        st.write(" ")

    with bottom_col3:
        st.subheader("Outlets assigned")
        assigned = len(
            st.session_state.auth_outlet_table.loc[
                st.session_state.auth_outlet_table["Outlet"] != ""
            ]
        )
        st.metric(label="Assigned", value=assigned)

else:
    st.info("You've reached the end of the list!")
    st.write(f"Authors skipped: {st.session_state.auth_outlet_skipped}")

    if st.session_state.auth_outlet_skipped > 0:
        reset_counter = st.button("Reset Counter")
        if reset_counter:
            st.session_state.auth_outlet_skipped = 0
            st.rerun()
    else:
        st.write("✓ Nothing left to update here.")