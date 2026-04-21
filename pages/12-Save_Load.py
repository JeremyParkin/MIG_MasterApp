# 12-Save_Load.py
from __future__ import annotations

from datetime import datetime
import io

import dill
import pandas as pd
import streamlit as st

SNAPSHOT_VERSION = 2
EXCLUDED_SESSION_KEYS = {
    "clean_excel_bytes",
    "clean_excel_built_at",
    "report_copy_docx_bytes",
    "report_copy_built_at",
    "notebooklm_zip_bytes",
    "notebooklm_info",
    "notebooklm_built_at",
}


st.title("Save & Load")
st.caption("Save the current session to resume later, or load a previous session snapshot back into the app.")
st.divider()


def _discover_dataframe_keys() -> list[str]:
    return sorted(
        [
            key
            for key, value in st.session_state.items()
            if isinstance(value, pd.DataFrame)
        ]
    )


def _build_serializable_session_payload() -> tuple[dict, list[str]]:
    payload: dict = {
        "_snapshot_version": SNAPSHOT_VERSION,
        "_saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    skipped: list[str] = []

    dataframe_keys = _discover_dataframe_keys()
    payload["df_names"] = dataframe_keys

    for key, value in st.session_state.items():
        if key.startswith("_") or key in EXCLUDED_SESSION_KEYS:
            continue

        try:
            dill.dumps(value)
        except Exception:
            skipped.append(key)
            continue

        payload[key] = value

    return payload, skipped


def _restore_dataframe_value(value) -> pd.DataFrame | None:
    if isinstance(value, pd.DataFrame):
        restored = value.copy()
    elif isinstance(value, str):
        try:
            restored = pd.read_csv(io.StringIO(value))
        except Exception:
            return None
    else:
        return None

    if "Date" in restored.columns:
        restored["Date"] = pd.to_datetime(restored["Date"], errors="coerce")

    return restored


def load_session_state(uploaded_file) -> None:
    uploaded_file.seek(0)
    session_data = dill.loads(uploaded_file.read())
    loaded_saved_at = session_data.get("_saved_at")
    loaded_snapshot_version = session_data.get("_snapshot_version")

    for key in list(st.session_state.keys()):
        del st.session_state[key]

    saved_df_names = session_data.get("df_names", [])
    restored_df_names: list[str] = []

    for df_name in saved_df_names:
        if df_name not in session_data:
            continue

        restored_df = _restore_dataframe_value(session_data[df_name])
        if restored_df is None:
            continue

        st.session_state[df_name] = restored_df
        restored_df_names.append(df_name)

    for key, value in session_data.items():
        if key in restored_df_names or key in {"df_names", "_snapshot_version", "_saved_at"}:
            continue
        st.session_state[key] = value

    st.session_state.df_names = restored_df_names if restored_df_names else saved_df_names
    st.session_state.pickle_load = True
    if loaded_saved_at:
        st.session_state.loaded_session_saved_at = loaded_saved_at
    if loaded_snapshot_version is not None:
        st.session_state.loaded_session_snapshot_version = loaded_snapshot_version


st.header("Save")

if not st.session_state.get("upload_step", False):
    st.error("Please upload a CSV before saving.")
elif not st.session_state.get("standard_step", False):
    st.error("Please run Basic Cleaning before saving.")
else:
    st.info("Save your current processing session as a .pkl file.")

    payload, skipped_keys = _build_serializable_session_payload()
    dt_string = datetime.now().strftime("%Y-%m-%d-%H-%M")
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
        load_session_state(uploaded_file)
        loaded_saved_at = st.session_state.get("loaded_session_saved_at")
        if loaded_saved_at:
            st.success(f"Session state loaded successfully. Snapshot saved at {loaded_saved_at}.")
        else:
            st.success("Session state loaded successfully.")
    except Exception as e:
        st.error(f"Could not load session file: {e}")
