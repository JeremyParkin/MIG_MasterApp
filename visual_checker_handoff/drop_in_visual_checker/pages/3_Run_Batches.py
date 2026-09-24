from __future__ import annotations

import streamlit as st

from processing.visual_relevance import (
    add_usage,
    cache_counts,
    completed_results_by_row,
    estimate_cost,
    get_sheet_data,
    process_batch,
    urls_from_rows,
)
from utils.session import require_setup

require_setup()

st.title("Run Batches")
st.caption("Process uncached URLs, retry transient failures, and recheck negative results.")

settings = st.session_state.visual_settings
headers, rows = get_sheet_data(st.session_state.visual_workbook_bytes, settings["sheet_name"])
urls_by_row = urls_from_rows(rows, headers, settings["url_column"])
st.session_state.visual_results_by_row.update(completed_results_by_row(urls_by_row, settings["cache_path"]))
counts = cache_counts(urls_by_row, settings["cache_path"])

metric_cols = st.columns(7)
metric_cols[0].metric("Rows with URLs", counts["rows_with_urls"])
metric_cols[1].metric("Unique URLs", counts["unique_urls"])
metric_cols[2].metric("Remaining", counts["remaining"])
metric_cols[3].metric("Cached", counts["cached"])
metric_cols[4].metric("Retryable", counts["retryable"])
metric_cols[5].metric("Blocked", counts["blocked"])
metric_cols[6].metric("Not Found", counts["not_found"])
st.caption(f"NO / NO IMAGE: {counts['negative']}")

usage = st.session_state.visual_session_usage
cost_cols = st.columns(4)
cost_cols[0].metric("Session API estimate", f"${estimate_cost(settings['model'], usage):,.4f}")
cost_cols[1].metric("Processed this session", st.session_state.visual_session_processed_urls)
cost_cols[2].metric("Input tokens", f"{usage['input_tokens']:,}")
cost_cols[3].metric("Output tokens", f"{usage['output_tokens']:,}")

if st.button("Reset session cost"):
    st.session_state.visual_session_usage = {"input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0}
    st.session_state.visual_session_processed_urls = 0
    st.rerun()

if not settings.get("api_key"):
    st.warning("Add an OpenAI API key on Setup to enable processing.")

can_run = bool(settings.get("api_key") and urls_by_row)
action_cols = st.columns([1, 1, 1, 3])
run_clicked = action_cols[0].button("Run next batch", type="primary", disabled=not can_run)
retry_clicked = action_cols[1].button("Retry failures", disabled=not (can_run and counts["retryable"]))
recheck_no_clicked = action_cols[2].button("Recheck NOs", disabled=not (can_run and counts["negative"]))


def run_selected_batch(mode: str) -> None:
    progress = st.progress(0, text="Preparing batch...")

    def update_progress(completed: int, timed_out: int, total: int) -> None:
        progress.progress(
            (completed + timed_out) / total,
            text=f"Checked {completed} of {total} URLs ({timed_out} timed out)",
        )

    attempted, completed, timed_out, cache, batch_usage = process_batch(
        urls_by_row=urls_by_row,
        api_key=settings["api_key"],
        model=settings["model"],
        batch_size=settings["retry_batch_size"] if mode in {"retry", "negative"} else settings["batch_size"],
        workers=settings["workers"],
        request_timeout=settings["request_timeout"],
        straggler_timeout=settings["straggler_timeout"],
        max_images_per_url=settings["max_images_per_url"],
        criteria=settings["criteria"],
        cache_path=settings["cache_path"],
        retry_errors=mode == "retry",
        retry_verdicts={"NO", "NO IMAGE"} if mode == "negative" else None,
        progress_callback=update_progress,
    )
    st.session_state.visual_session_usage = add_usage(st.session_state.visual_session_usage, batch_usage)
    st.session_state.visual_session_processed_urls += completed
    for row, url in urls_by_row:
        if url in cache:
            st.session_state.visual_results_by_row[row] = cache[url]
    if attempted:
        message = f"Processed {completed} of {attempted} URL{'s' if attempted != 1 else ''}."
        if timed_out:
            message += f" Left {timed_out} timed-out URL{'s' if timed_out != 1 else ''} in the queue."
        st.success(message)
    else:
        st.info("No matching URLs are currently queued.")


if run_clicked:
    with st.spinner("Fetching pages and checking visuals..."):
        run_selected_batch("new")
if retry_clicked:
    with st.spinner("Retrying transient failures..."):
        run_selected_batch("retry")
if recheck_no_clicked:
    with st.spinner("Rechecking negative rows..."):
        run_selected_batch("negative")

results = st.session_state.visual_results_by_row
if results:
    preview_rows = []
    for row, url in urls_by_row:
        result = results.get(row)
        if result:
            preview_rows.append(
                {
                    "row": row + 2,
                    "url": url,
                    "verdict": result["verdict"],
                    "image_url": result["image_url"],
                    "images_checked": result.get("images_checked", ""),
                    "error_retries": result.get("error_retry_count", ""),
                    "no_rechecks": result.get("negative_recheck_count", ""),
                    "reason": result["reason"],
                }
            )
    st.divider()
    st.subheader("Results Preview")
    st.dataframe(preview_rows, use_container_width=True, hide_index=True)
else:
    st.info("No cached or processed results are available yet.")
