from __future__ import annotations

import os

import streamlit as st

from processing.visual_relevance import get_sheet_data, likely_url_columns, urls_from_rows
from utils.session import require_workbook

require_workbook()

st.title("Setup")
st.caption("Map workbook columns, define relevance criteria, and set processing parameters.")

settings = dict(st.session_state.visual_settings)

st.subheader("Column Mapping")
sheet_name = st.selectbox(
    "Sheet",
    st.session_state.visual_sheet_names,
    index=st.session_state.visual_sheet_names.index(settings["sheet_name"])
    if settings.get("sheet_name") in st.session_state.visual_sheet_names
    else 0,
)
headers, rows = get_sheet_data(st.session_state.visual_workbook_bytes, sheet_name)
url_options = likely_url_columns(headers)
url_column = st.selectbox(
    "URL column",
    url_options,
    index=url_options.index(settings["url_column"]) if settings.get("url_column") in url_options else 0,
)
urls_by_row = urls_from_rows(rows, headers, str(url_column))

metric_cols = st.columns(3)
metric_cols[0].metric("Rows", len(rows))
metric_cols[1].metric("Rows with URLs", len(urls_by_row))
metric_cols[2].metric("Unique URLs", len({url for _, url in urls_by_row}))

st.divider()
st.subheader("Image Relevance Criteria")
criteria = st.text_area(
    "Criteria",
    value=settings.get("criteria", ""),
    height=170,
    help="Describe what should count as a relevant image for this project.",
)

st.divider()
st.subheader("Processing")
api_key = st.text_input(
    "OpenAI API key",
    value=settings.get("api_key") or os.environ.get("OPENAI_API_KEY", ""),
    type="password",
    help="Stored only in Streamlit session state unless this project later adds a secrets flow.",
)

col1, col2, col3 = st.columns(3)
with col1:
    model = st.text_input("Vision model", value=settings.get("model", "gpt-4.1-mini"))
    batch_size = st.number_input("Batch size", min_value=1, max_value=200, value=int(settings.get("batch_size", 25)), step=5)
    retry_batch_size = st.number_input("Retry batch size", min_value=1, max_value=200, value=int(settings.get("retry_batch_size", 25)), step=5)
with col2:
    workers = st.slider("Parallel fetches", min_value=1, max_value=12, value=int(settings.get("workers", 4)))
    max_images_per_url = st.slider("Images to check per URL", min_value=1, max_value=8, value=int(settings.get("max_images_per_url", 4)))
with col3:
    request_timeout = st.number_input("Network/API timeout seconds", min_value=5, max_value=120, value=int(settings.get("request_timeout", 15)), step=5)
    straggler_timeout = st.number_input("Straggler cutoff seconds", min_value=15, max_value=300, value=int(settings.get("straggler_timeout", 90)), step=15)
    cache_path = st.text_input("Cache file", value=settings.get("cache_path", "visual_cache.json"))

if st.button("Save setup", type="primary"):
    st.session_state.visual_settings = {
        "api_key": api_key,
        "model": model.strip() or "gpt-4.1-mini",
        "batch_size": int(batch_size),
        "retry_batch_size": int(retry_batch_size),
        "workers": int(workers),
        "request_timeout": int(request_timeout),
        "straggler_timeout": int(straggler_timeout),
        "max_images_per_url": int(max_images_per_url),
        "sheet_name": str(sheet_name),
        "url_column": str(url_column),
        "criteria": criteria.strip(),
        "cache_path": cache_path.strip() or "visual_cache.json",
    }
    selection_key = f"{sheet_name}:{url_column}"
    if st.session_state.visual_selection_key != selection_key:
        st.session_state.visual_selection_key = selection_key
        st.session_state.visual_results_by_row = {}
    st.session_state.visual_settings_saved = True
    st.success("Setup saved.")

if st.session_state.visual_settings_saved:
    st.info("Setup is saved. Continue to Run Batches when ready.")
