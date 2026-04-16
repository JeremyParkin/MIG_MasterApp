from __future__ import annotations

import streamlit as st

st.title("Top Stories")
st.caption("Build the saved top-stories list, then generate report-ready insights in one combined workflow.")

if not st.session_state.get("standard_step", False):
    st.error("Please complete Basic Cleaning before trying this step.")
    st.stop()

from ui.top_stories_selection_view import render_top_stories_selection
from ui.top_story_summaries_view import render_top_story_summaries

if st.session_state.get("top_stories_section") == "Summaries":
    st.session_state.top_stories_section = "Insights"
st.session_state.setdefault("top_stories_section", "Selection")

st.markdown(
    """
    <style>
    .top-stories-step-note {
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

step1, step2 = st.columns(2, gap="small")
with step1:
    if st.button(
        "1. Selection",
        key="top_stories_workflow_step_selection",
        use_container_width=True,
        type="primary" if st.session_state.top_stories_section == "Selection" else "secondary",
    ):
        st.session_state.top_stories_section = "Selection"
with step2:
    if st.button(
        "2. Insights",
        key="top_stories_workflow_step_summaries",
        use_container_width=True,
        type="primary" if st.session_state.top_stories_section == "Insights" else "secondary",
    ):
        st.session_state.top_stories_section = "Insights"

st.markdown(
    '<div class="top-stories-step-note">Work left to right: save the final top stories, then generate report-ready insights.</div>',
    unsafe_allow_html=True,
)

if st.session_state.get("top_stories_section") == "Insights":
    render_top_story_summaries()
else:
    st.session_state.top_stories_section = "Selection"
    render_top_stories_selection()
