from __future__ import annotations

import streamlit as st
from streamlit_tags import st_tags

from processing.analysis_context import (
    apply_analysis_context_suggestions,
    build_analysis_context_text,
    build_analysis_context_discovery_prompt,
    DEFAULT_ANALYSIS_CONTEXT_MODEL,
    generate_analysis_context_suggestions,
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
if "analysis_context_save_success" not in st.session_state:
    st.session_state.analysis_context_save_success = False
if "analysis_context_suggestion_success" not in st.session_state:
    st.session_state.analysis_context_suggestion_success = False
if "analysis_context_tag_widget_version" not in st.session_state:
    st.session_state.analysis_context_tag_widget_version = 0
pending_suggestions = st.session_state.pop("analysis_context_pending_suggestions", None)
if pending_suggestions:
    apply_analysis_context_suggestions(st.session_state, pending_suggestions)
    st.session_state.analysis_context_tag_widget_version += 1
    st.session_state.analysis_context_suggestion_success = True

payload = get_analysis_context_payload(st.session_state)

col1, col2 = st.columns(2, gap="medium")
with col1:
    client_name = st.text_input(
        "Client name",
        value=payload["client_name"],
        help="Usually carried over from Getting Started, but you can adjust it here if needed.",
    )
with col2:
    primary_name = st.text_input(
        "Primary topic or entity of interest",
        value=payload["primary_name"],
        help="This can be the client itself, or a broader topic you want the AI to focus on instead.",
    )

helper_col1, helper_col2, helper_col3 = st.columns([1, 0.55, 1.85], gap="small")
with helper_col1:
    if st.button("Suggest context items with AI", type="primary", key="analysis_context_ai_suggest", use_container_width=True):
        try:
            with st.spinner("Generating context suggestions..."):
                suggestions, _, _ = generate_analysis_context_suggestions(
                    client_name=client_name,
                    primary_name=primary_name,
                    alternate_names=payload["alternate_names"],
                    spokespeople=payload["spokespeople"],
                    products=payload["products"],
                    guidance=payload["guidance"],
                    api_key=st.secrets["key"],
                    model=DEFAULT_ANALYSIS_CONTEXT_MODEL,
                )
            st.session_state.analysis_context_pending_suggestions = suggestions
            st.rerun()
        except Exception as e:
            st.error(f"Could not generate context suggestions: {e}")
with helper_col2:
    if st.button("Clear all below", key="analysis_context_clear_below", use_container_width=True):
        st.session_state.analysis_alternate_names = []
        st.session_state.analysis_spokespeople = []
        st.session_state.analysis_products = []
        st.session_state.analysis_guidance = ""
        st.session_state.ui_alternate_names = []
        st.session_state.ui_spokespeople = []
        st.session_state.ui_products = []
        st.session_state.ui_toning_rationale = ""
        st.session_state.analysis_context_suggestion_payload = None
        st.session_state.analysis_context_tag_widget_version += 1
        st.rerun()

if st.session_state.get("analysis_context_suggestion_success"):
    st.success("AI context suggestions added to the fields above.")
    st.session_state.analysis_context_suggestion_success = False

tag_key_suffix = st.session_state.analysis_context_tag_widget_version
alternate_names = st_tags(
    label="Alternate names / aliases",
    text="Press enter to add more",
    maxtags=20,
    value=payload["alternate_names"],
    key=f"analysis_context_aliases_tags_{tag_key_suffix}",
)
spokespeople = st_tags(
    label="Key spokespeople",
    text="Press enter to add more",
    maxtags=20,
    value=payload["spokespeople"],
    key=f"analysis_context_spokespeople_tags_{tag_key_suffix}",
)
products = st_tags(
    label="Products / sub-brands / initiatives",
    text="Press enter to add more",
    maxtags=20,
    value=payload["products"],
    key=f"analysis_context_products_tags_{tag_key_suffix}",
)
guidance = st.text_area(
    "Additional rationale, context, or guidance (optional)",
    value=payload["guidance"],
    height=110,
    help="Use this for analytical framing, nuances, or focus that should shape AI-generated summaries and observations.",
)

with st.expander("AI suggestion rationale", expanded=False):
    suggestion_payload = st.session_state.get("analysis_context_suggestion_payload")
    if suggestion_payload:
        assessment = str(suggestion_payload.get("assessment", "") or "").strip()
        if assessment:
            st.write("**Assessment**")
            st.write(assessment)

        section_map = [
            ("Alternate names / aliases", suggestion_payload.get("aliases", [])),
            ("Key spokespeople", suggestion_payload.get("spokespeople", [])),
            ("Products / sub-brands / initiatives", suggestion_payload.get("products", [])),
        ]
        for heading, items in section_map:
            if not items:
                continue
            st.write(f"**{heading}**")
            for item in items:
                name = str(item.get("name", "") or "").strip()
                detail = str(item.get("detail", "") or "").strip()
                if not name:
                    continue
                st.markdown(f"- `{name}`" + (f": {detail}" if detail else ""))
    else:
        st.info("No AI suggestions have been generated yet.")

save_col, save_status_col = st.columns([0.32, 0.68], gap="small")
with save_col:
    if st.button("Save Analysis Context", type="primary", use_container_width=True):
        save_analysis_context(
            st.session_state,
            client_name=client_name,
            primary_name=primary_name,
            alternate_names=alternate_names,
            spokespeople=spokespeople,
            products=products,
            guidance=guidance,
        )
        st.session_state.analysis_context_save_success = True

with save_status_col:
    if st.session_state.get("analysis_context_save_success"):
        st.success("Analysis context saved.")

with st.expander("Current shared context preview", expanded=False):
    preview = build_analysis_context_text(st.session_state)
    if preview:
        st.code(preview, language="text")
    else:
        st.info("No analysis context has been saved yet.")

with st.expander("AI prompt preview", expanded=False):
    st.code(
        build_analysis_context_discovery_prompt(
            client_name=client_name,
            primary_name=primary_name,
            alternate_names=payload["alternate_names"],
            spokespeople=payload["spokespeople"],
            products=payload["products"],
            guidance=payload["guidance"],
        ),
        language="text",
    )
