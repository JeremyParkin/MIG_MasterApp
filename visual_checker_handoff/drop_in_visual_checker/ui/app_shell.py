from __future__ import annotations

import streamlit as st

from processing.visual_relevance import estimate_cost
from utils.session import ensure_visual_checker_state


def standard_sidebar(target=None) -> None:
    target = target or st.sidebar
    ensure_visual_checker_state()

    with target.container():
        st.logo(
            "https://www.agilitypr.com/wp-content/uploads/2024/12/agility-logo-white.png",
            size="large",
        )
        st.markdown(
            """
            <div style="line-height:1.35; margin:0.1rem 0 0.6rem 0;">
              <div style="font-weight:600;">Visual Relevance Checker</div>
              <div style="opacity:0.72; font-size:0.9rem;">Standalone workflow</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        settings = st.session_state.visual_settings
        usage = st.session_state.visual_session_usage
        cost = estimate_cost(settings.get("model", ""), usage)
        st.caption(f"Session API estimate: USD${cost:,.4f}")


def build_pages() -> list:
    return [
        st.Page("pages/1_Upload.py", title="Upload", icon=":material/upload_file:"),
        st.Page("pages/2_Setup.py", title="Setup", icon=":material/tune:"),
        st.Page("pages/3_Run_Batches.py", title="Run Batches", icon=":material/play_circle:"),
        st.Page("pages/4_QA.py", title="QA", icon=":material/rule:"),
        st.Page("pages/5_Download.py", title="Download", icon=":material/download:"),
    ]


def run_navigation(position: str = "sidebar") -> None:
    nav = st.navigation(build_pages(), position=position)
    sidebar_shell = st.sidebar.empty()
    standard_sidebar(sidebar_shell)
    nav.run()
    standard_sidebar(sidebar_shell)
