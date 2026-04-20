# session.py
from __future__ import annotations

import pandas as pd
import streamlit as st


DEFAULT_STRING_VARS = {
    "top_auths_by": "Mentions",
    "export_name": "",
    "client_name": "",
    "auth_skip_counter": 0,
    "auth_outlet_skipped": 0,
}

DEFAULT_DF_KEYS = [
    "df_traditional",
    "df_social",
    "df_dupes",
    "auth_outlet_table",
    "df_untouched",
    "author_outlets",
    "blank_set",
    "added_df",
    "markdown_content",
    "filtered_df",
    "df_grouped",
    "selected_df",
    "selected_rows",
    "top_stories",
    "auth_outlet_todo",
]

DEFAULT_STEP_VARS = [
    "upload_step",
    "standard_step",
    "translated_headline",
    "translated_summary",
    "translated_snippet",
    "filled",
    "pickle_load",
]


def init_getting_started_state() -> None:
    for key, value in DEFAULT_STRING_VARS.items():
        if key not in st.session_state:
            st.session_state[key] = value

    for key in DEFAULT_DF_KEYS:
        if key not in st.session_state:
            st.session_state[key] = pd.DataFrame()

    for key in DEFAULT_STEP_VARS:
        if key not in st.session_state:
            st.session_state[key] = False

    if "ave_col" not in st.session_state:
        st.session_state["ave_col"] = "AVE"

    if "original_ave_col" not in st.session_state:
        st.session_state["original_ave_col"] = None


def clear_all_session_state() -> None:
    for key in list(st.session_state.keys()):
        del st.session_state[key]
