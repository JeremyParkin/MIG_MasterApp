from __future__ import annotations

import streamlit as st

if not st.session_state.get("standard_step", False):
    st.title("Top Stories Summaries")
    st.error("Please complete Basic Cleaning before trying this step.")
    st.stop()

from ui.top_story_summaries_view import render_top_story_summaries

render_top_story_summaries()
