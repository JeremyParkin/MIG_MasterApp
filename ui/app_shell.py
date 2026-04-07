from __future__ import annotations
import streamlit as st

from utils.api_meter import init_api_meter, get_api_cost_usd


# ----------------------------
# Sidebar
# ----------------------------
def standard_sidebar() -> None:
    """Render branding + feedback link + session cost meter."""
    st.sidebar.image(
        "https://www.agilitypr.com/wp-content/uploads/2024/12/agility-logo-white.png",
        width=230,
    )

    st.sidebar.markdown("MIG Master App")
    st.sidebar.caption(
        "Version: April 2026 - "
        "[Feedback](https://forms.office.com/Pages/ResponsePage.aspx?id=GvcJkLbBVUumZQrrWC6V07d2jCu79C5FsfEZJPZEfZxUNVlIVDRNNVBQVEgxQVFXNEM5VldUMkpXNS4u)"
    )

    # API meter
    init_api_meter()
    cost_usd = get_api_cost_usd()
    st.sidebar.caption(f"Est. session cost USD${cost_usd:,.4f}")


# ----------------------------
# Navigation
# ----------------------------
def build_pages() -> list:
    """Return list of Streamlit pages."""
    return [
        st.Page("pages/1-Getting_Started.py", title="Getting Started", icon=":material/play_arrow:"),
        st.Page("pages/2-Basic_Cleaning.py", title="Basic Cleaning", icon=":material/mop:"),
        st.Page("pages/3-Missing_Authors.py", title="Missing Authors", icon=":material/ink_pen:"),
        st.Page("pages/4-Author_Outlets.py", title="Authors Outlets", icon=":material/cell_tower:"),
        st.Page("pages/5-Translation.py", title="Translation", icon=":material/mediation:"),
        st.Page("pages/6-Top_Stories.py", title="Top Stories", icon=":material/newspaper:"),
        st.Page("pages/7-Summaries.py", title="Summaries", icon=":material/chat:"),
        st.Page("pages/8-Tagging.py", title="Tagging", icon=":material/sell:"),
        st.Page("pages/9-AI_Sentiment.py", title="AI Sentiment", icon=":material/auto_awesome:"),
        st.Page("pages/10-Spot_Checks.py", title="Spot Checks", icon=":material/fact_check:"),
        st.Page("pages/11-Download.py", title="Download", icon=":material/download:"),
        st.Page("pages/12-Save_Load.py", title="Save & Load", icon=":material/save:"),
    ]


def run_navigation(position: str = "sidebar") -> None:
    """Run navigation + sidebar shell."""
    nav = st.navigation(build_pages(), position=position)
    standard_sidebar()
    nav.run()