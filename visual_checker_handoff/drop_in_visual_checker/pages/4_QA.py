from __future__ import annotations

import streamlit as st

from processing.visual_relevance import get_sheet_data, urls_from_rows
from utils.session import require_setup

require_setup()

st.title("QA")
st.caption("Review and correction tools will live here as the workflow matures.")

settings = st.session_state.visual_settings
headers, rows = get_sheet_data(st.session_state.visual_workbook_bytes, settings["sheet_name"])
urls_by_row = urls_from_rows(rows, headers, settings["url_column"])
results = st.session_state.visual_results_by_row

if not results:
    st.info("Run at least one batch before QA.")
    st.stop()

verdict_filter = st.multiselect(
    "Verdicts",
    sorted({result.get("verdict", "") for result in results.values() if result.get("verdict")}),
    default=[],
)

qa_rows = []
for row, url in urls_by_row:
    result = results.get(row)
    if not result:
        continue
    if verdict_filter and result.get("verdict") not in verdict_filter:
        continue
    qa_rows.append(
        {
            "source_row": row + 2,
            "verdict": result.get("verdict", ""),
            "reason": result.get("reason", ""),
            "checked_image": result.get("image_url", ""),
            "article_url": url,
        }
    )

st.dataframe(qa_rows, use_container_width=True, hide_index=True)
st.caption("Future additions: manual overrides, sampling, image preview, and audit notes.")
