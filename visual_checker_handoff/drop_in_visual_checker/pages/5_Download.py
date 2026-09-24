from __future__ import annotations

from pathlib import Path

import streamlit as st

from processing.visual_relevance import workbook_bytes_with_results
from utils.session import require_setup
from utils.time_display import format_local_timestamp

require_setup()

st.title("Download")
st.caption("Build and download the workbook enriched with visual relevance results.")

settings = st.session_state.visual_settings
results = st.session_state.visual_results_by_row

if not results:
    st.error("Run at least one batch before building the enriched workbook.")
    st.stop()

st.subheader("Enriched workbook")
had_workbook = "visual_enriched_workbook_bytes" in st.session_state

if st.button("Build enriched workbook", type="primary"):
    try:
        with st.spinner("Building workbook now..."):
            st.session_state.visual_enriched_workbook_bytes = workbook_bytes_with_results(
                st.session_state.visual_workbook_bytes,
                settings["sheet_name"],
                settings["url_column"],
                results,
            )
            st.session_state.visual_enriched_workbook_built_at = format_local_timestamp()
        action_word = "rebuilt" if had_workbook else "built"
        st.success(f"Enriched workbook {action_word} at {st.session_state.visual_enriched_workbook_built_at}.")
    except Exception as exc:
        st.error(f"Error building workbook: {exc}")

if "visual_enriched_workbook_bytes" in st.session_state:
    output_name = Path(st.session_state.visual_uploaded_name).with_suffix("").name + "_visuals_checked.xlsx"
    st.download_button(
        "Download enriched workbook",
        data=st.session_state.visual_enriched_workbook_bytes,
        file_name=output_name,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    st.caption(
        f"Current workbook built: {st.session_state.visual_enriched_workbook_built_at} · "
        f"Rows with results: {len(results):,}"
    )
