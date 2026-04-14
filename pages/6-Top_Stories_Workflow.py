from __future__ import annotations

from pathlib import Path
import runpy

import streamlit as st


st.title("Top Stories")
st.caption("Build the saved top-stories list, then generate summary outputs in one combined workflow.")

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
        "2. Summaries",
        key="top_stories_workflow_step_summaries",
        use_container_width=True,
        type="primary" if st.session_state.top_stories_section == "Summaries" else "secondary",
    ):
        st.session_state.top_stories_section = "Summaries"

st.markdown(
    '<div class="top-stories-step-note">Work left to right: save the final top stories, then generate report-ready summary outputs.</div>',
    unsafe_allow_html=True,
)

legacy_pages_dir = Path(__file__).resolve().parent.parent / "legacy_pages"

if st.session_state.get("top_stories_section") == "Summaries":
    runpy.run_path(str(legacy_pages_dir / "7-Summaries.py"), run_name="__main__")
else:
    st.session_state.top_stories_section = "Selection"
    runpy.run_path(str(legacy_pages_dir / "6-Top_Stories.py"), run_name="__main__")
