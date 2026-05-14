from __future__ import annotations

from utils.time_display import current_timestamp_storage_string, format_local_timestamp


def init_session_timing_state(session_state) -> None:
    session_state.setdefault("session_started_at", None)
    session_state.setdefault("session_last_resumed_at", None)
    session_state.setdefault("session_accumulated_seconds", 0.0)


def ensure_session_timing_started(session_state) -> None:
    init_session_timing_state(session_state)
    if session_state.get("session_started_at"):
        return

    now_value = current_timestamp_storage_string()
    session_state.session_started_at = now_value
    session_state.session_last_resumed_at = now_value
    session_state.session_accumulated_seconds = float(session_state.get("session_accumulated_seconds", 0.0) or 0.0)


def get_current_session_duration_seconds(session_state) -> float:
    init_session_timing_state(session_state)

    accumulated = float(session_state.get("session_accumulated_seconds", 0.0) or 0.0)
    last_resumed_at = session_state.get("session_last_resumed_at")
    if not last_resumed_at:
        return accumulated

    from pandas import Timestamp, isna, to_datetime

    parsed_last = to_datetime(last_resumed_at, errors="coerce")
    if isna(parsed_last):
        return accumulated

    parsed_now = to_datetime(current_timestamp_storage_string(), errors="coerce")
    if isna(parsed_now):
        return accumulated

    elapsed = (Timestamp(parsed_now) - Timestamp(parsed_last)).total_seconds()
    if elapsed < 0:
        elapsed = 0
    return accumulated + float(elapsed)


def build_session_timing_snapshot_fields(session_state) -> dict[str, object]:
    init_session_timing_state(session_state)
    started_at = session_state.get("session_started_at") or current_timestamp_storage_string()
    return {
        "session_started_at": started_at,
        "session_accumulated_seconds": float(get_current_session_duration_seconds(session_state)),
        "session_last_resumed_at": current_timestamp_storage_string(),
    }


def restore_session_timing_after_load(session_state) -> None:
    init_session_timing_state(session_state)
    if not session_state.get("session_started_at"):
        ensure_session_timing_started(session_state)
        return

    try:
        session_state.session_accumulated_seconds = float(session_state.get("session_accumulated_seconds", 0.0) or 0.0)
    except Exception:
        session_state.session_accumulated_seconds = 0.0
    session_state.session_last_resumed_at = current_timestamp_storage_string()


def format_session_duration(seconds: float | int | None) -> str:
    total_seconds = int(max(0, float(seconds or 0)))
    minutes, _ = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)

    if days > 0:
        return f"{days}d {hours}h {minutes:02d}m"
    if hours > 0:
        return f"{hours}h {minutes:02d}m"
    return f"{minutes}m"


def format_session_started(session_state) -> str:
    started_at = session_state.get("session_started_at")
    if not started_at:
        return ""
    return format_local_timestamp(started_at)
