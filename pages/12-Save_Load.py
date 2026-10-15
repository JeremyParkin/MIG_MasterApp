# 12-Save_Load.py
from __future__ import annotations

import streamlit as st
from ui.page_help import set_page_help_context
from ui.session_file_controls import render_session_build_link, render_session_download_link
from utils.session_download import register_session_download
from utils.time_display import format_local_timestamp
from utils.session_snapshot import (
    build_session_snapshot_bytes,
    build_session_snapshot_filename,
    load_session_state_from_file,
)


st.title("Save & Load")
st.caption("Save your session or restore a saved session.")
set_page_help_context(st.session_state, "Save & Load")
st.divider()


st.header("Save")

if not st.session_state.get("upload_step", False):
    st.error("Please upload a CSV before saving.")
elif not st.session_state.get("standard_step", False):
    st.error("Please run Basic Cleaning before saving.")
else:
    # Private state is excluded from saved sessions, preventing recursive file growth.
    prepared_file = st.session_state.get("_prepared_session_file")
    if prepared_file:
        # Release the old inline encoding if this session survived an app update.
        prepared_file.pop("encoded_data", None)
    download_url = None
    build_message = None
    build_error = None
    if st.session_state.pop("_session_file_build_requested", False):
        try:
            with st.spinner("Building session file..."):
                snapshot_bytes, skipped_keys = build_session_snapshot_bytes(st.session_state)
                built_file = {
                    "data": snapshot_bytes,
                    "file_name": build_session_snapshot_filename(st.session_state),
                    "skipped_keys": skipped_keys,
                }
                download_url = register_session_download(snapshot_bytes, built_file["file_name"])
                built_file["built_at"] = format_local_timestamp()
            action_word = "rebuilt" if prepared_file else "built"
            prepared_file = built_file
            st.session_state._prepared_session_file = prepared_file
            build_message = f"Session file {action_word} at {prepared_file['built_at']}"
        except Exception as e:
            build_error = f"Error building session file: {e}"

    render_session_build_link(rebuild=bool(prepared_file))
    if build_message:
        st.success(build_message)
    if build_error:
        st.error(build_error)

    if prepared_file:
        try:
            download_url = download_url or register_session_download(
                prepared_file["data"], prepared_file["file_name"]
            )
            render_session_download_link(download_url, prepared_file["file_name"])
        except Exception as e:
            st.error(f"Could not prepare session download: {e}")
        st.caption(f"Current session file built: {prepared_file['built_at']}")
        if prepared_file["skipped_keys"]:
            st.caption("Skipped non-serializable keys: " + ", ".join(sorted(prepared_file["skipped_keys"])))

st.divider()
st.header("Load")

uploaded_file = st.file_uploader(
    "Restore a Previous Session", type="pkl", label_visibility="collapsed", key="_restore_session_file"
)

if uploaded_file is not None and uploaded_file.file_id != st.session_state.get("_loaded_session_upload_id"):
    try:
        load_session_state_from_file(st.session_state, uploaded_file)
        st.session_state._loaded_session_upload_id = uploaded_file.file_id
        loaded_saved_at = st.session_state.get("loaded_session_saved_at")
        if loaded_saved_at:
            st.session_state._session_load_message = (
                f"Session state loaded successfully. Saved at {format_local_timestamp(loaded_saved_at)}."
            )
        else:
            st.session_state._session_load_message = "Session state loaded successfully."
        st.rerun()
    except Exception as e:
        st.error(f"Could not load session file: {e}")

if st.session_state.get("_session_load_message"):
    st.success(st.session_state._session_load_message)
