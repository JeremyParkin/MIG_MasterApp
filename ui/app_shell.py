# app_shell.py
from __future__ import annotations
import streamlit as st

from utils.api_meter import init_api_meter, get_api_cost_usd
from ui.page_help import render_sidebar_page_help


# ----------------------------
# Sidebar
# ----------------------------
def standard_sidebar(target=None, *, key_suffix: str = "default") -> None:
    """Render branding + feedback link + session cost meter."""
    target = target or st.sidebar
    sidebar_container = target.container()
    with sidebar_container:
        st.markdown(
            """
            <style>
            section[data-testid="stSidebar"] .block-container {
                padding-top: 0.45rem;
            }
            section[data-testid="stSidebarNav"] {
                margin-top: 0;
            }
            section[data-testid="stSidebarNav"] ul {
                gap: 0.1rem;
            }
            section[data-testid="stSidebarNav"] li {
                margin: 0;
            }
            section[data-testid="stSidebarNav"] a {
                padding-top: 0.22rem !important;
                padding-bottom: 0.22rem !important;
                min-height: 0 !important;
            }
            section[data-testid="stSidebar"] [data-testid="stImage"] {
                margin-bottom: 0.35rem;
            }
            section[data-testid="stSidebar"] p {
                margin-bottom: 0.2rem;
            }
            .sidebar-app-meta {
                margin: 0.05rem 0 0.45rem 0;
                line-height: 1.35;
            }
            .sidebar-app-title {
                font-weight: 600;
                margin-bottom: 0.05rem;
            }
            .sidebar-app-subtle {
                color: rgba(250, 250, 250, 0.68);
                font-size: 0.9rem;
            }
            .sidebar-cost {
                margin: 0.2rem 0 0.45rem 0;
                color: rgba(250, 250, 250, 0.78);
                font-size: 0.92rem;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
        st.image(
            "https://www.agilitypr.com/wp-content/uploads/2024/12/agility-logo-white.png",
            width=210,
        )
        st.markdown(
            """
            <div class="sidebar-app-meta">
              <div class="sidebar-app-title">MIG Master App</div>
              <div class="sidebar-app-subtle">Version: April 2026 · <a href="https://forms.office.com/Pages/ResponsePage.aspx?id=GvcJkLbBVUumZQrrWC6V07d2jCu79C5FsfEZJPZEfZxUNVlIVDRNNVBQVEgxQVFXNEM5VldUMkpXNS4u" target="_blank">Feedback</a></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        init_api_meter()
        cost_usd = get_api_cost_usd()
        st.markdown(f'<div class="sidebar-cost">Est. session cost USD${cost_usd:,.4f}</div>', unsafe_allow_html=True)
        render_sidebar_page_help(sidebar_container, key_suffix=key_suffix)


# ----------------------------
# Navigation
# ----------------------------
def build_pages() -> list:
    """Return list of Streamlit pages."""
    return [
        st.Page("pages/1-Getting_Started.py", title="Getting Started", icon=":material/play_arrow:"),
        st.Page("pages/2-Basic_Cleaning.py", title="Basic Cleaning", icon=":material/mop:"),
        st.Page("pages/2a-Analysis_Context.py", title="Analysis Context", icon=":material/tune:"),
        st.Page("pages/3-Authors.py", title="Authors", icon=":material/groups:"),
        st.Page("pages/4-Outlets.py", title="Outlets", icon=":material/apartment:"),
        st.Page("pages/5-Translation.py", title="Translation", icon=":material/mediation:"),
        st.Page("pages/6-Top_Stories_Workflow.py", title="Top Stories", icon=":material/newspaper:"),
        st.Page("pages/7-Regions.py", title="Regions", icon=":material/public:"),
        st.Page("pages/8-Tagging.py", title="Tagging", icon=":material/sell:"),
        st.Page("pages/9-AI_Sentiment.py", title="Sentiment", icon=":material/auto_awesome:"),
        st.Page("pages/11-Download.py", title="Download", icon=":material/download:"),
        st.Page("pages/12-Save_Load.py", title="Save & Load", icon=":material/save:"),
    ]


def run_navigation(position: str = "sidebar") -> None:
    """Run navigation + sidebar shell."""
    nav = st.navigation(build_pages(), position=position)
    nav_title = str(getattr(nav, "title", "") or "").strip()
    st.session_state.page_help_page = nav_title
    st.session_state.page_help_step = ""
    sidebar_shell = st.sidebar.empty()
    standard_sidebar(sidebar_shell, key_suffix="pre")
    nav.run()
    if not str(st.session_state.get("page_help_page", "") or "").strip():
        st.session_state.page_help_page = nav_title
    standard_sidebar(sidebar_shell, key_suffix="post")
