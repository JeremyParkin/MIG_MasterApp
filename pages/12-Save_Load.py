# 12-Save_Load.py
from __future__ import annotations

import dill
import streamlit as st
from ui.page_help import set_page_help_context
from utils.time_display import (
    current_timestamp_filename_string,
    format_local_timestamp,
)
from utils.session_snapshot import (
    build_serializable_session_payload,
    load_session_state_from_file,
)


st.title("Save & Load")
st.caption("Save the current session to resume later, or load a previous session snapshot back into the app.")
set_page_help_context(st.session_state, "Save & Load")
st.divider()


st.header("Save")

if not st.session_state.get("upload_step", False):
    st.error("Please upload a CSV before saving.")
elif not st.session_state.get("standard_step", False):
    st.error("Please run Basic Cleaning before saving.")
else:
    st.info("Save your current processing session as a .pkl file.")

    payload, skipped_keys = build_serializable_session_payload(st.session_state)
    dt_string = current_timestamp_filename_string()
    client_name = st.session_state.get("client_name", "session").strip() or "session"
    file_name = f"{client_name} - {dt_string}.pkl"

    st.download_button(
        label="Download Session File",
        data=dill.dumps(payload),
        file_name=file_name,
        mime="application/octet-stream",
        type="primary",
    )

    if skipped_keys:
        st.caption(
            "Skipped non-serializable keys: "
            + ", ".join(sorted(skipped_keys))
        )
    st.caption("Generated download files are excluded from the session snapshot and can be rebuilt after loading.")

st.divider()
st.header("Load")
st.info("Load a previously saved .pkl session file.")

uploaded_file = st.file_uploader("Restore a Previous Session", type="pkl", label_visibility="collapsed")

if uploaded_file is not None:
    try:
        load_session_state_from_file(st.session_state, uploaded_file)
        loaded_saved_at = st.session_state.get("loaded_session_saved_at")
        if loaded_saved_at:
            st.success(f"Session state loaded successfully. Snapshot saved at {format_local_timestamp(loaded_saved_at)}.")
        else:
            st.success("Session state loaded successfully.")
    except Exception as e:
        st.error(f"Could not load session file: {e}")
