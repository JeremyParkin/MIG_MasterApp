from __future__ import annotations

import streamlit as st

from ui.app_shell import run_navigation

st.set_page_config(
    page_title="Visual Relevance Checker",
    page_icon="https://www.agilitypr.com/wp-content/uploads/2025/01/favicon.png",
    layout="wide",
)

run_navigation(position="sidebar")
