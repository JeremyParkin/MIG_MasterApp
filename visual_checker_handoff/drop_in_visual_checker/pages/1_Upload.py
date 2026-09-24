from __future__ import annotations

import streamlit as st

from processing.visual_relevance import workbook_key, workbook_row_count, workbook_sheet_names
from utils.session import clear_visual_checker_state, ensure_visual_checker_state

ensure_visual_checker_state()

st.title("Upload")
st.caption("Upload the workbook that contains article URLs for visual relevance checking.")

uploaded_file = st.file_uploader(
    "Upload an Excel workbook",
    type=["xlsx"],
    accept_multiple_files=False,
    help="Use the original monitoring workbook or an app-produced workbook with URL rows.",
)

if uploaded_file is not None:
    uploaded_bytes = uploaded_file.getvalue()
    key = workbook_key(uploaded_bytes)
    if st.session_state.visual_workbook_key != key:
        st.session_state.visual_workbook_key = key
        st.session_state.visual_uploaded_name = uploaded_file.name
        st.session_state.visual_workbook_bytes = uploaded_bytes
        st.session_state.visual_sheet_names = workbook_sheet_names(uploaded_bytes)
        st.session_state.visual_results_by_row = {}
        st.session_state.visual_selection_key = ""
        st.session_state.visual_settings_saved = False
        st.success(f"File uploaded: {uploaded_file.name}")
    else:
        st.info(f"Current file: {uploaded_file.name}")

if st.session_state.visual_workbook_bytes:
    st.divider()
    conf_col, button_col = st.columns([4, 1], vertical_alignment="center")
    with conf_col:
        st.success(f"Ready: {st.session_state.visual_uploaded_name}")
    with button_col:
        if st.button("Start Over", use_container_width=True):
            clear_visual_checker_state()
            st.rerun()

    st.subheader("Workbook")
    metric_cols = st.columns(3)
    metric_cols[0].metric("Sheets", len(st.session_state.visual_sheet_names))
    first_sheet = st.session_state.visual_sheet_names[0] if st.session_state.visual_sheet_names else ""
    metric_cols[1].metric("First sheet rows", workbook_row_count(st.session_state.visual_workbook_bytes, first_sheet) if first_sheet else 0)
    metric_cols[2].metric("Workbook key", st.session_state.visual_workbook_key[:8])

    st.dataframe(
        [{"sheet": sheet, "rows": workbook_row_count(st.session_state.visual_workbook_bytes, sheet)} for sheet in st.session_state.visual_sheet_names],
        hide_index=True,
        use_container_width=True,
    )
else:
    st.info("Upload an `.xlsx` file to begin.")
