from __future__ import annotations

import io
from typing import Any

import dill
import pandas as pd

from utils.session_timing import build_session_timing_snapshot_fields, restore_session_timing_after_load
from utils.time_display import current_timestamp_filename_string, current_timestamp_storage_string


SNAPSHOT_VERSION = 2
EXCLUDED_SESSION_KEYS = {
    "clean_excel_bytes",
    "clean_excel_built_at",
    "report_copy_docx_bytes",
    "report_copy_built_at",
    "notebooklm_zip_bytes",
    "notebooklm_info",
    "notebooklm_built_at",
    # Transient button values cannot be assigned back to live Streamlit widgets.
    "build_session_file",
    "sidebar_page_help_open_pre",
    "sidebar_page_help_open_post",
}


def _collect_session_payload(session_state) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "_snapshot_version": SNAPSHOT_VERSION,
        "_saved_at": current_timestamp_storage_string(),
    }
    for key, value in session_state.items():
        if key.startswith("_") or key in EXCLUDED_SESSION_KEYS:
            continue
        payload[key] = value
    payload["df_names"] = sorted(
        key for key, value in payload.items() if isinstance(value, pd.DataFrame)
    )
    payload.update(build_session_timing_snapshot_fields(session_state))
    return payload


def _filter_serializable_values(payload: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    serializable: dict[str, Any] = {}
    skipped: list[str] = []
    for key, value in payload.items():
        try:
            dill.dumps(value)
        except Exception:
            skipped.append(key)
            continue
        serializable[key] = value
    serializable["df_names"] = [key for key in payload["df_names"] if key in serializable]
    return serializable, skipped


def build_serializable_session_payload(session_state) -> tuple[dict[str, Any], list[str]]:
    return _filter_serializable_values(_collect_session_payload(session_state))


def build_session_snapshot_bytes(
    session_state,
    *,
    payload_overrides: dict[str, Any] | None = None,
) -> tuple[bytes, list[str]]:
    payload = _collect_session_payload(session_state)
    if payload_overrides:
        payload.update(payload_overrides)
    try:
        # Most sessions are fully serializable. Avoid serializing each dataframe twice.
        return dill.dumps(payload), []
    except Exception:
        payload, skipped = _filter_serializable_values(payload)
        return dill.dumps(payload), skipped


def build_session_snapshot_filename(session_state) -> str:
    client_name = str(session_state.get("client_name", "session") or "session").strip() or "session"
    return f"{client_name} - {current_timestamp_filename_string()}.pkl"


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


def load_session_state_from_file(session_state, uploaded_file) -> None:
    from processing.ai_tagging import ensure_canonical_tag_definitions, remove_reserved_tag_from_text

    uploaded_file.seek(0)
    session_data = dill.loads(uploaded_file.read())
    loaded_saved_at = session_data.get("_saved_at")
    loaded_snapshot_version = session_data.get("_snapshot_version")

    for key in list(session_state.keys()):
        del session_state[key]

    saved_df_names = session_data.get("df_names", [])
    restored_df_names: list[str] = []
    for df_name in saved_df_names:
        if df_name not in session_data:
            continue
        restored_df = _restore_dataframe_value(session_data[df_name])
        if restored_df is None:
            continue
        session_state[df_name] = restored_df
        restored_df_names.append(df_name)

    for key, value in session_data.items():
        if (
            key in restored_df_names
            or key in EXCLUDED_SESSION_KEYS
            or key in {"df_names", "_snapshot_version", "_saved_at"}
        ):
            continue
        session_state[key] = value

    if "tag_definitions" in session_state:
        session_state.tag_definitions = ensure_canonical_tag_definitions(session_state.tag_definitions)
    if "tags_text" in session_state:
        session_state.tags_text = remove_reserved_tag_from_text(session_state.tags_text)
    session_state.df_names = restored_df_names if restored_df_names else saved_df_names
    session_state.pickle_load = True
    if loaded_saved_at:
        session_state.loaded_session_saved_at = loaded_saved_at
    if loaded_snapshot_version is not None:
        session_state.loaded_session_snapshot_version = loaded_snapshot_version
    restore_session_timing_after_load(session_state)
