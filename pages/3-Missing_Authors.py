from __future__ import annotations

import urllib.parse
import warnings

import pandas as pd
import streamlit as st

from processing.missing_authors import (
    init_missing_authors_state,
    prepare_author_working_df,
    get_available_visible_flags,
    build_fixable_headline_table,
    fixable_headline_stats,
    get_headline_authors,
    get_possible_authors,
    apply_author_fix,
    build_last_author_fix_payload,
    undo_last_author_fix,
)
from utils.dataframe_helpers import top_x_by_mentions

warnings.filterwarnings("ignore")

st.title("Authors - Missing")

if not st.session_state.get("standard_step", False):
    st.error("Please complete Basic Cleaning before trying this step.")
    st.stop()

if len(st.session_state.get("df_traditional", [])) == 0:
    st.subheader("No traditional media in data. Skip to next step.")
    st.stop()

init_missing_authors_state(st.session_state)

counter = st.session_state.auth_skip_counter
reviewed = st.session_state.auth_reviewed_count

hide_table_row_index = """
    <style>
    tbody th {display:none}
    .blank {display:none}
    </style>
"""
st.markdown(hide_table_row_index, unsafe_allow_html=True)

base_working_df = st.session_state.df_traditional.copy()
visible_flags, visible_defaults = get_available_visible_flags(base_working_df)

excluded_flags = st.multiselect(
    "Exclude coverage flags",
    options=visible_flags,
    default=visible_defaults,
    help="Exclude selected flagged coverage from the missing-author workflow on this page.",
)

author_working_df = prepare_author_working_df(
    st.session_state.df_traditional,
    excluded_flags=excluded_flags,
)

headline_table = build_fixable_headline_table(author_working_df)
temp_headline_list = headline_table.copy()

if len(temp_headline_list) == 0:
    st.success("No fixable missing-author headlines remain in the current filtered view.")

elif counter < len(temp_headline_list):
    headline_text = temp_headline_list.iloc[counter]["Headline"]
    encoded_headline = urllib.parse.quote(f'"{headline_text}"')
    google_search_url = f"https://www.google.com/search?q={encoded_headline}"

    headline_authors_df = get_headline_authors(author_working_df, headline_text).copy()
    possibles = get_possible_authors(author_working_df, headline_text)

    but1, col3, but2, but4 = st.columns(4)

    with but1:
        next_auth = st.button("Skip to Next Headline")
        if next_auth:
            st.session_state.auth_skip_counter = counter + 1
            st.rerun()

    with col3:
        if counter > 0:
            st.write(f"Skipped: {counter}")

    with but2:
        if counter > 0:
            reset_counter = st.button("Reset Skip Counter")
            if reset_counter:
                st.session_state.auth_skip_counter = 0
                st.rerun()

    with but4:
        undo_available = st.session_state.get("last_author_fix") is not None
        undo_clicked = st.button(
            "Undo Last Author Update",
            disabled=not undo_available,
            help="Reverses the most recent author update applied on this page.",
        )
        if undo_clicked:
            undo_last_author_fix(st.session_state)
            st.rerun()

    form_block = st.container()
    info_block = st.container()

    with info_block:
        col1, col2, col3 = st.columns([12, 1, 9])

        with col1:
            st.subheader("Headline")
            st.table(headline_table.iloc[[counter]])
            st.markdown(
                f'&nbsp;&nbsp;» <a href="{google_search_url}" target="_blank" style="text-decoration:underline; color:lightblue;">Search Google for this headline</a>',
                unsafe_allow_html=True,
            )

        with col2:
            st.write(" ")

        with col3:
            st.subheader("Authors in CSV")
            st.table(headline_authors_df)

    with form_block:
        with st.form("auth_updater", clear_on_submit=True):
            col1, col2, col3 = st.columns([8, 1, 8])

            with col1:
                box_author = st.selectbox(
                    "Pick from possible Authors",
                    possibles,
                    help="Pick from one of the authors already associated with this headline.",
                )

            with col2:
                st.write(" ")
                st.subheader("OR")

            with col3:
                string_author = st.text_input(
                    "Write in the author name",
                    help="Override above selection by writing in a custom name.",
                )

            submitted = st.form_submit_button("Update Author", type="primary")

        if submitted:
            new_author = string_author.strip() if len(string_author.strip()) > 0 else box_author

            if not new_author:
                st.warning("Please choose or enter an author name.")
            else:
                st.session_state.last_author_fix = build_last_author_fix_payload(
                    st.session_state.df_traditional,
                    headline_text,
                    previous_reviewed_count=st.session_state.auth_reviewed_count,
                )

                st.session_state.df_traditional = apply_author_fix(
                    st.session_state.df_traditional,
                    headline_text,
                    new_author,
                )

                st.session_state.auth_reviewed_count = reviewed + 1
                st.rerun()

else:
    st.info("You've reached the end of the list!")

    top_end_col1, top_end_col2 = st.columns([1, 1])

    with top_end_col1:
        if counter > 0:
            reset_counter = st.button("Reset Counter")
            if reset_counter:
                st.session_state.auth_skip_counter = 0
                st.rerun()

    with top_end_col2:
        undo_available = st.session_state.get("last_author_fix") is not None
        undo_clicked = st.button(
            "Undo Last Author Update",
            disabled=not undo_available,
            help="Reverses the most recent author update applied on this page.",
        )
        if undo_clicked:
            undo_last_author_fix(st.session_state)
            st.rerun()

    if counter == 0:
        st.success("✓ Nothing left to update here.")

st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Original Top Authors")

    media_type_column = "Type" if "Type" in st.session_state.df_untouched.columns else "Media Type"

    filtered_df = st.session_state.df_untouched[
        st.session_state.df_untouched[media_type_column].isin(
            ["PRINT", "ONLINE_NEWS", "ONLINE", "BLOGS", "PRESS_RELEASE"]
        )
    ].copy()

    if "Mentions" not in filtered_df.columns:
        filtered_df["Mentions"] = 1

    original_top_authors = top_x_by_mentions(filtered_df, "Author")
    st.write(original_top_authors)

with col2:
    st.subheader("New Top Authors")
    st.dataframe(top_x_by_mentions(author_working_df, "Author"), use_container_width=True)

with col3:
    st.subheader("Fixable Author Stats")
    remaining = fixable_headline_stats(
        author_working_df,
        counter=st.session_state.auth_skip_counter,
        primary="Headline",
        secondary="Author",
    )

    statscol1, statscol2 = st.columns(2)

    with statscol1:
        reviewed_display = (
            len(temp_headline_list) - remaining["remaining"] + reviewed
            if len(temp_headline_list) > 0
            else reviewed
        )
        st.metric(label="Reviewed", value=reviewed_display)
        st.metric(label="Updated", value=reviewed)

    with statscol2:
        st.metric(label="Remaining in this view", value=remaining["remaining"])