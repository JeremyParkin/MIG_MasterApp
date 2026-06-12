from __future__ import annotations

import math

from utils.session_snapshot import build_session_snapshot_bytes, build_session_snapshot_filename


MIN_CHECKPOINT_GROUPS = 2000


def _progress_key(workflow: str, stage: str) -> str:
    return f"ai_checkpoint_{workflow}_{stage}_completed"


def _next_key(workflow: str, stage: str) -> str:
    return f"ai_checkpoint_{workflow}_{stage}_next"


def record_checkpoint_progress(
    session_state,
    *,
    workflow: str,
    stage: str,
    successful_results: int,
    interval: int,
) -> None:
    progress_key = _progress_key(workflow, stage)
    next_key = _next_key(workflow, stage)
    session_state[progress_key] = int(session_state.get(progress_key, 0) or 0) + max(0, int(successful_results))
    session_state.setdefault(next_key, int(interval))


def reset_workflow_checkpoints(session_state, workflow: str) -> None:
    prefix = f"ai_checkpoint_{workflow}_"
    for key in list(session_state.keys()):
        if str(key).startswith(prefix):
            del session_state[key]


def acknowledge_checkpoint(session_state, *, workflow: str, stage: str, interval: int) -> None:
    completed = int(session_state.get(_progress_key(workflow, stage), 0) or 0)
    session_state[_next_key(workflow, stage)] = _next_checkpoint_after(completed, interval)


def _next_checkpoint_after(completed: int, interval: int) -> int:
    return max(
        int(interval),
        int(math.floor(completed / interval) + 1) * int(interval),
    )


def render_checkpoint_save_reminder(
    session_state,
    *,
    workflow: str,
    stage: str,
    dataset_group_count: int,
    interval: int,
) -> None:
    import streamlit as st

    if int(dataset_group_count) <= MIN_CHECKPOINT_GROUPS:
        return

    completed = int(session_state.get(_progress_key(workflow, stage), 0) or 0)
    next_checkpoint = int(session_state.get(_next_key(workflow, stage), interval) or interval)
    if completed < next_checkpoint:
        return

    stage_label = "first opinion" if stage == "first" else "second opinion"
    st.warning(
        f"You have completed {completed:,} successful {workflow} {stage_label} results since this dataset was prepared. "
        "Download a session checkpoint before continuing."
    )
    snapshot_bytes, skipped = build_session_snapshot_bytes(
        session_state,
        payload_overrides={
            _next_key(workflow, stage): _next_checkpoint_after(completed, interval),
        },
    )
    st.download_button(
        "Download session checkpoint",
        data=snapshot_bytes,
        file_name=build_session_snapshot_filename(session_state),
        mime="application/octet-stream",
        key=f"download_checkpoint_{workflow}_{stage}_{next_checkpoint}",
        on_click=acknowledge_checkpoint,
        kwargs={
            "session_state": session_state,
            "workflow": workflow,
            "stage": stage,
            "interval": interval,
        },
    )
    if skipped:
        st.caption("Some non-serializable temporary UI state was excluded from this checkpoint.")
