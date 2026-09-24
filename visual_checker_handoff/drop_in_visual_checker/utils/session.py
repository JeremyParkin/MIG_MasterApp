from __future__ import annotations

import streamlit as st

from processing.visual_relevance import (
    DEFAULT_MAX_IMAGES_PER_URL,
    DEFAULT_MODEL,
    DEFAULT_REQUEST_TIMEOUT,
    DEFAULT_RETRY_BATCH_SIZE,
    DEFAULT_STRAGGLER_TIMEOUT,
    zero_usage,
)


DEFAULT_CRITERIA = (
    "Relevant visuals include runway/catwalk images, models wearing client-relevant apparel, "
    "event venues, backstage scenes, and event attendees. Exclude stock photos, headshots, "
    "product-only images, logos, banners, unrelated news photos, and generic decoration."
)


def ensure_visual_checker_state() -> None:
    if "visual_workbook_key" not in st.session_state:
        st.session_state.visual_workbook_key = ""
    if "visual_uploaded_name" not in st.session_state:
        st.session_state.visual_uploaded_name = ""
    if "visual_workbook_bytes" not in st.session_state:
        st.session_state.visual_workbook_bytes = b""
    if "visual_sheet_names" not in st.session_state:
        st.session_state.visual_sheet_names = []
    if "visual_results_by_row" not in st.session_state:
        st.session_state.visual_results_by_row = {}
    if "visual_selection_key" not in st.session_state:
        st.session_state.visual_selection_key = ""
    if "visual_session_usage" not in st.session_state:
        st.session_state.visual_session_usage = zero_usage()
    if "visual_session_processed_urls" not in st.session_state:
        st.session_state.visual_session_processed_urls = 0
    if "visual_settings_saved" not in st.session_state:
        st.session_state.visual_settings_saved = False
    if "visual_settings" not in st.session_state:
        st.session_state.visual_settings = {
            "api_key": "",
            "model": DEFAULT_MODEL,
            "batch_size": 25,
            "retry_batch_size": DEFAULT_RETRY_BATCH_SIZE,
            "workers": 4,
            "request_timeout": DEFAULT_REQUEST_TIMEOUT,
            "straggler_timeout": DEFAULT_STRAGGLER_TIMEOUT,
            "max_images_per_url": DEFAULT_MAX_IMAGES_PER_URL,
            "sheet_name": "",
            "url_column": "",
            "criteria": DEFAULT_CRITERIA,
            "cache_path": "visual_cache.json",
        }


def clear_visual_checker_state() -> None:
    for key in list(st.session_state.keys()):
        if key.startswith("visual_"):
            del st.session_state[key]
    ensure_visual_checker_state()


def require_workbook() -> None:
    ensure_visual_checker_state()
    if not st.session_state.visual_workbook_bytes:
        st.error("Upload a workbook before continuing.")
        st.stop()


def require_setup() -> None:
    require_workbook()
    if not st.session_state.visual_settings_saved:
        st.error("Complete setup before continuing.")
        st.stop()
