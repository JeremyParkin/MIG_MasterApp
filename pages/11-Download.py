from __future__ import annotations

import warnings

import pandas as pd
import streamlit as st

from processing.download_exports import build_clean_workbook_bytes
from processing.notebooklm_exports import build_notebooklm_zip

warnings.filterwarnings("ignore")

st.title("Download")

if not st.session_state.get("standard_step", False):
    st.error("Please complete Basic Cleaning before trying this step.")
    st.stop()

st.divider()
st.subheader("Clean data workbook")

had_clean_workbook = "clean_excel_bytes" in st.session_state

build_xlsx = st.button("Build cleaned data workbook", key="build_clean_workbook")
if build_xlsx:
    try:
        with st.spinner("Building workbook now..."):
            st.session_state.clean_excel_bytes = build_clean_workbook_bytes(st.session_state)
            st.session_state.clean_excel_built_at = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
            action_word = "rebuilt" if had_clean_workbook else "built"
            st.success(f"Cleaned workbook {action_word} at {st.session_state.clean_excel_built_at}")
    except Exception as e:
        st.error(f"Error building Excel workbook: {e}")

if "clean_excel_bytes" in st.session_state:
    export_name = f"{st.session_state.export_name} - clean_data.xlsx"
    st.download_button(
        "Download cleaned data workbook",
        st.session_state.clean_excel_bytes,
        file_name=export_name,
        type="primary",
        key="download_clean_workbook",
    )

    if "clean_excel_built_at" in st.session_state:
        st.caption(f"Current workbook built: {st.session_state.clean_excel_built_at}")

st.divider()
st.subheader("NotebookLM bundle")

had_notebooklm_bundle = "notebooklm_zip_bytes" in st.session_state

build_nlm = st.button(
    "Build NotebookLM bundle (zip)",
    key="build_notebooklm_bundle",
    help=(
        "Creates a zip of JSON-formatted text files for NotebookLM. "
        "If the dataset exceeds ~50 files worth of content, "
        "a random sample is taken to stay within upload limits."
    ),
)

if build_nlm:
    try:
        with st.spinner("Building NotebookLM bundle..."):
            nlm_zip_io, nlm_info = build_notebooklm_zip(
                st.session_state.df_traditional,
                st.session_state.df_social,
                client_name=st.session_state.client_name,
            )
        st.session_state.notebooklm_zip_bytes = nlm_zip_io.getvalue()
        st.session_state.notebooklm_info = nlm_info
        st.session_state.notebooklm_built_at = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        action_word = "rebuilt" if had_notebooklm_bundle else "built"
        st.success(f"NotebookLM bundle {action_word} at {st.session_state.notebooklm_built_at}")
    except ValueError as e:
        st.error(str(e))
    except Exception as e:
        st.error(f"Error building NotebookLM bundle: {e}")

if "notebooklm_zip_bytes" in st.session_state:
    info = st.session_state.get("notebooklm_info", {})
    client_short = info.get("client_short", "Client")
    zip_filename = f"NLM_prepared_data-{client_short}.zip"

    st.download_button(
        "Download NotebookLM bundle",
        data=st.session_state.notebooklm_zip_bytes,
        file_name=zip_filename,
        type="primary",
    )

    if "notebooklm_built_at" in st.session_state:
        st.caption(f"Current NotebookLM bundle built: {st.session_state.notebooklm_built_at}")

    if info:
        st.caption(
            f"Rows in cleaned dataset: {info.get('total_rows', 0):,} · "
            f"Rows included in bundle: {info.get('rows_included', 0):,} · "
            f"Files: {info.get('files_created', 0)} (max {info.get('max_files', 0)}), "
            f"Rows/file cap: {info.get('max_rows_per_file', 0)}, "
            f"Word limit/file: {info.get('max_words_per_file', 0):,}, "
            f"Size limit/file: {info.get('max_bytes_per_file', 0) // (1024 * 1024)} MB."
        )

st.divider()
client_name = st.session_state.client_name

with st.expander("NotebookLM prompt examples for this client"):
    st.markdown(
        f"""
### Executive Summary
Generate a concise 2-paragraph executive summary of the media coverage of **{client_name}**.  
Present the information as though it is going to be included as a lead into a media briefing for an executive.  
Focus on high-level concepts rather than specific facts and figures.

---

### High-level coverage summary
You are analyzing media coverage for **{client_name}**.  
Summarize the overall coverage tone, key themes, and notable storylines across this dataset.  
Highlight the most influential outlets and authors, and note any spikes in volume, sentiment shifts, or recurring issues.
"""
    )