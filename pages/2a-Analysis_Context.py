from __future__ import annotations

import streamlit as st
from streamlit_tags import st_tags

from processing.analysis_context import (
    build_analysis_context_caption,
    build_analysis_context_text,
    get_analysis_context_payload,
    init_analysis_context_state,
    save_analysis_context,
)


st.title("Analysis Context")
st.caption("Optionally define the topic, aliases, spokespeople, products, and analytical guidance that should shape AI outputs across the app.")

if not st.session_state.get("standard_step", False):
    st.error("Please complete Basic Cleaning before trying this step.")
    st.stop()

init_analysis_context_state(st.session_state)
payload = get_analysis_context_payload(st.session_state)
if "analysis_context_save_success" not in st.session_state:
    st.session_state.analysis_context_save_success = False

if payload["client_name"]:
    st.info(f"Client from Getting Started: {payload['client_name']}")
else:
    st.info("No client name is set yet. You can still define the analysis focus below.")

st.write("**Shared analysis context**")
col1, col2 = st.columns(2, gap="medium")
with col1:
    primary_name = st.text_input(
        "Primary topic or entity of interest",
        value=payload["primary_name"],
        help="This can be the client itself, or a broader topic you want the AI to focus on instead.",
    )
with col2:
    st.caption("This shared context is reused by Top Stories, Sentiment, Authors, and Outlets.")
    context_caption = build_analysis_context_caption(st.session_state)
    if context_caption:
        st.caption(context_caption)

alternate_names = st_tags(
    label="Alternate names / aliases",
    text="Press enter to add more",
    maxtags=20,
    value=payload["alternate_names"],
    key="analysis_context_aliases_tags",
)
spokespeople = st_tags(
    label="Key spokespeople",
    text="Press enter to add more",
    maxtags=20,
    value=payload["spokespeople"],
    key="analysis_context_spokespeople_tags",
)
products = st_tags(
    label="Products / sub-brands / initiatives",
    text="Press enter to add more",
    maxtags=20,
    value=payload["products"],
    key="analysis_context_products_tags",
)
guidance = st.text_area(
    "Additional rationale, context, or guidance (optional)",
    value=payload["guidance"],
    height=110,
    help="Use this for analytical framing, nuances, or focus that should shape AI-generated summaries and observations.",
)

if st.button("Save Analysis Context", type="primary"):
    save_analysis_context(
        st.session_state,
        primary_name=primary_name,
        alternate_names=alternate_names,
        spokespeople=spokespeople,
        products=products,
        guidance=guidance,
    )
    st.session_state.analysis_context_save_success = True

if st.session_state.get("analysis_context_save_success"):
    st.success("Analysis context saved.")

with st.expander("Current shared context preview", expanded=False):
    preview = build_analysis_context_text(st.session_state)
    if preview:
        st.code(preview, language="text")
    else:
        st.info("No analysis context has been saved yet.")
