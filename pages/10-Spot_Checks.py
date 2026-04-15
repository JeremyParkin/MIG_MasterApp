from __future__ import annotations

import streamlit as st

if not st.session_state.get("sentiment_config_step", False):
    st.title("Spot Checks")
    st.caption("Review, correct, and finalize sentiment decisions on grouped stories before exporting the finished dataset.")
    st.error("Please complete AI Sentiment setup before trying this step.")
    st.stop()

from ui.spot_checks_view import render_spot_checks_page

render_spot_checks_page()
