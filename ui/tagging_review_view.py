from __future__ import annotations


def render_tagging_review_page(*, review_stage: str) -> None:
    import pandas as pd
    import streamlit as st

    from processing.analysis_context import get_analysis_context_payload
    from processing.ai_tagging import (
        auto_assign_resolved_tag_matches,
        compute_all_tagged_candidates,
        DEFAULT_TAGGING_REVIEW_BATCH_SIZE,
        DEFAULT_TAGGING_REVIEW_CONFIDENCE_THRESHOLD,
        DEFAULT_TAGGING_REVIEW_MODEL,
        DEFAULT_TAGGING_MAX_WORKERS,
        compute_tag_review_candidates,
        ensure_tag_review_columns,
        filter_tag_candidates_for_review_mode,
        get_effective_ai_tag_confidence_series,
        get_effective_tag_series,
        normalize_tag_list,
        recommend_tag_second_opinion_batch_size,
        run_batch_tag_second_opinion,
        set_assigned_tag,
    )
    from processing.sentiment_config import build_tolerant_regex_str
    from processing.spot_checks import (
        escape_markdown,
        highlight_with_tolerant_regex,
        translate_text,
        apply_translation_to_group,
    )
    from utils.api_meter import apply_usage_to_session
    from utils.time_display import format_local_timestamp

    def sync_tagging_state(unique_df: pd.DataFrame, rows_df: pd.DataFrame) -> None:
        st.session_state.df_tagging_unique = unique_df
        st.session_state.df_tagging_rows = rows_df

    df_unique, df_rows = ensure_tag_review_columns(
        st.session_state.df_tagging_unique,
        st.session_state.df_tagging_rows,
    )
    sync_tagging_state(df_unique, df_rows)
    tagging_mode = st.session_state.get("tagging_mode", "Single best tag")

    review_base_candidates = compute_tag_review_candidates(st.session_state.df_tagging_unique, exclude_reviewed=False)
    base_candidates = compute_tag_review_candidates(st.session_state.df_tagging_unique, exclude_reviewed=True)
    all_tagged_candidates = compute_all_tagged_candidates(st.session_state.df_tagging_unique)
    ai_tag_all = df_unique.get("AI Tag", pd.Series(index=df_unique.index, dtype="object")).fillna("").astype(str).str.strip()
    review_tag_all = df_unique.get("Review AI Tag", pd.Series(index=df_unique.index, dtype="object")).fillna("").astype(str).str.strip()
    assigned_all = df_unique.get("Assigned Tag", pd.Series(index=df_unique.index, dtype="object")).fillna("").astype(str).str.strip()
    assigned_source_all = df_unique.get("Assigned Tag Source", pd.Series(index=df_unique.index, dtype="object")).fillna("").astype(str).str.strip()
    agreement_all = df_unique.get("AI Tag Agreement", pd.Series(index=df_unique.index, dtype="object")).fillna("").astype(str).str.strip()
    needs_review_all = df_unique.get("Needs Human Review", pd.Series(index=df_unique.index, dtype="object")).fillna("").astype(str).str.strip()

    with_first_opinion_count = int((ai_tag_all != "").sum())
    with_second_opinion_count = int((review_tag_all != "").sum())
    eligible_second_opinion_count = int(((assigned_all == "") & (ai_tag_all != "") & (review_tag_all == "")).sum())
    auto_resolved_count = int((assigned_source_all == "AI_AUTO_RESOLVED").sum())
    disagreement_count = int(((review_tag_all != "") & (agreement_all == "Disagree")).sum())
    needs_review_count = int(((review_tag_all != "") & (assigned_all == "") & (needs_review_all == "Yes")).sum())

    pre_review_message = st.session_state.get("tagging_pre_review_message")
    if pre_review_message and review_stage == "pre_review":
        st.success(pre_review_message)
        st.session_state.tagging_pre_review_message = None

    auto_resolve_message = st.session_state.get("tagging_auto_resolve_message")
    if auto_resolve_message and review_stage == "pre_review":
        st.success(auto_resolve_message)
        st.session_state.tagging_auto_resolve_message = None

    if review_stage == "pre_review":
        st.subheader("Step 3: AI Second Opinion")
        st.caption("Generate one second AI opinion per already-tagged story group so likely high-confidence matches can be auto-resolved before human spot checks.")

        if base_candidates.empty:
            st.success("All set — no grouped stories remain eligible for a second AI opinion.")
            summary1, summary2, summary3, summary4, summary5 = st.columns(5)
            with summary1:
                st.metric("With 1st opinion", f"{with_first_opinion_count:,}")
            with summary2:
                st.metric("With 2nd opinion", f"{with_second_opinion_count:,}")
            with summary3:
                st.metric("Still eligible", f"{eligible_second_opinion_count:,}")
            with summary4:
                st.metric("Auto-resolved by AI", f"{auto_resolved_count:,}")
            with summary5:
                st.metric("Needs human review", f"{needs_review_count:,}")
            return

        stored_source_count = int(st.session_state.get("tagging_second_opinion_target_source_count", 0) or 0)
        stored_target = int(st.session_state.get("tagging_second_opinion_target_batch", 0) or 0)
        if stored_target <= 0 or with_first_opinion_count > stored_source_count:
            stored_target = recommend_tag_second_opinion_batch_size(len(base_candidates))
            st.session_state.tagging_second_opinion_target_batch = stored_target
            st.session_state.tagging_second_opinion_target_source_count = with_first_opinion_count
        completed_second_opinion_count = with_second_opinion_count
        remaining_recommended = max(0, stored_target - completed_second_opinion_count)
        recommended_batch = min(len(base_candidates), remaining_recommended)
        if "tagging_pre_review_n" not in st.session_state:
            st.session_state.tagging_pre_review_n = max(
                1,
                min(
                    len(base_candidates),
                    recommended_batch or DEFAULT_TAGGING_REVIEW_BATCH_SIZE,
                ),
            )
        current_batch_size = int(st.session_state.get("tagging_pre_review_n", 0) or 0)
        if current_batch_size > len(base_candidates):
            current_batch_size = len(base_candidates)
        # If the widget is still sitting on the generic default while the recommendation
        # is something more specific, seed it from the recommendation instead.
        if (
            recommended_batch > 0
            and current_batch_size == DEFAULT_TAGGING_REVIEW_BATCH_SIZE
            and recommended_batch != DEFAULT_TAGGING_REVIEW_BATCH_SIZE
            and with_second_opinion_count == 0
        ):
            current_batch_size = recommended_batch
        if recommended_batch == 0 and len(base_candidates) > 0:
            current_batch_size = min(current_batch_size or min(10, len(base_candidates)), len(base_candidates))
        st.session_state.tagging_pre_review_n = max(1, current_batch_size)
        default_batch_size = st.session_state.tagging_pre_review_n

        st.caption("Second-opinion priority favors more syndicated, higher-visibility, lower-confidence stories before you move on to human review.")

        batch_col1, batch_col2 = st.columns([1.25, 1], gap="medium")
        with batch_col1:
            st.number_input(
                "Stories to send for AI second opinion",
                min_value=1,
                max_value=min(len(base_candidates), 200),
                value=max(1, default_batch_size),
                step=1,
                key="tagging_pre_review_n",
            )
        with batch_col2:
            st.metric("Recommended batch", f"{recommended_batch:,}")
            if recommended_batch > 0:
                st.caption(f"Selected for this batch: {int(st.session_state.get('tagging_pre_review_n', default_batch_size)):,} of {len(base_candidates):,} eligible groups.")
            else:
                st.caption("The current recommended second-opinion coverage has already been reached. You can still run more manually if desired.")

        with st.expander("Advanced auto-resolve options", expanded=False):
            threshold_value = st.number_input(
                "Auto-resolve confidence threshold",
                min_value=1,
                max_value=100,
                value=int(
                    st.session_state.get(
                        "tagging_review_low_conf_threshold",
                        DEFAULT_TAGGING_REVIEW_CONFIDENCE_THRESHOLD,
                    )
                ),
                step=1,
                key="tagging_review_low_conf_threshold",
            )
            st.caption("This only controls whether matching first and second AI opinions are auto-resolved. It does not control which stories are selected for second opinion.")
            if st.button("Apply threshold to reviewed matches", key="tagging_apply_auto_resolve_threshold", use_container_width=True):
                unique2, rows2, accepted_count, reopened_count = auto_assign_resolved_tag_matches(
                    st.session_state.df_tagging_unique,
                    st.session_state.df_tagging_rows,
                    tagging_mode=tagging_mode,
                    confidence_threshold=int(threshold_value),
                )
                sync_tagging_state(unique2, rows2)
                st.session_state.tagging_auto_resolve_message = (
                    f"Applied the auto-resolve threshold to existing reviewed matches. "
                    f"{accepted_count:,} grouped storie(s) were finalized and {reopened_count:,} were reopened."
                )
                st.rerun()

        action1, action2 = st.columns(2)
        with action1:
            if st.button("Run AI second opinion", type="secondary", use_container_width=True):
                tag_definitions = st.session_state.get("tag_definitions", {})
                st.session_state.pop("tagging_pre_review_message", None)
                selected_batch_size = int(st.session_state.get("tagging_pre_review_n", default_batch_size))
                progress_text = st.empty()
                progress_bar = st.progress(0.0)
                progress_text.caption(f"Running second opinions: 0/{selected_batch_size}")
                with st.spinner("Running AI second opinions on top tagging candidates..."):
                    unique2, rows2, batch_errors, total_in, total_out, batch_auto_resolved = run_batch_tag_second_opinion(
                        candidates_df=base_candidates,
                        df_unique=st.session_state.df_tagging_unique,
                        df_grouped=st.session_state.df_tagging_rows,
                        tag_definitions=tag_definitions,
                        tagging_mode=tagging_mode,
                        api_key=st.secrets["key"],
                        review_model=DEFAULT_TAGGING_REVIEW_MODEL,
                        limit=selected_batch_size,
                        max_workers=DEFAULT_TAGGING_MAX_WORKERS,
                        low_conf_threshold=int(
                            st.session_state.get(
                                "tagging_review_low_conf_threshold",
                                DEFAULT_TAGGING_REVIEW_CONFIDENCE_THRESHOLD,
                            )
                        ),
                        progress_callback=lambda completed, total: (
                            progress_bar.progress(completed / max(1, total)),
                            progress_text.caption(f"Running second opinions: {completed}/{total}"),
                        ),
                    )
                progress_bar.progress(1.0)
                progress_text.caption(f"Completed second opinions: {min(selected_batch_size, len(base_candidates))}/{min(selected_batch_size, len(base_candidates))}")
                sync_tagging_state(unique2, rows2)
                apply_usage_to_session(total_in, total_out, DEFAULT_TAGGING_REVIEW_MODEL)
                completed_at = format_local_timestamp()
                st.session_state.tagging_pre_review_message = (
                    f"AI second-opinion batch completed {completed_at}. Review counts updated below."
                )
                st.session_state["__last_tagging_pre_review_summary__"] = {
                    "errors": batch_errors,
                    "processed": min(selected_batch_size, len(base_candidates)),
                    "auto_resolved": batch_auto_resolved,
                }
                st.rerun()
        with action2:
            st.caption("High-confidence matching tag opinions can auto-resolve here before you move on to human spot checks.")

        summary1, summary2, summary3, summary4, summary5, summary6 = st.columns(6)
        with summary1:
            st.metric("With 1st opinion", f"{with_first_opinion_count:,}")
        with summary2:
            st.metric("With 2nd opinion", f"{with_second_opinion_count:,}")
        with summary3:
            st.metric("Still eligible", f"{len(base_candidates):,}")
        with summary4:
            st.metric("Auto-resolved by AI", f"{auto_resolved_count:,}")
        with summary5:
            st.metric("Needs human review", f"{needs_review_count:,}")
        with summary6:
            st.metric("Disagreements", f"{disagreement_count:,}")

        last_summary = st.session_state.get("__last_tagging_pre_review_summary__")
        if last_summary and (last_summary.get("processed") or last_summary.get("auto_resolved")):
            st.caption(
                f"Last second-opinion batch: {int(last_summary.get('processed', 0)):,} processed, {int(last_summary.get('auto_resolved', 0)):,} auto-resolved."
            )
        if last_summary and last_summary.get("errors"):
            with st.expander(f"Completed with {len(last_summary['errors'])} error(s)", expanded=False):
                for err in last_summary["errors"]:
                    st.write(err)
        return

    st.subheader("Step 4: Tag Spot Checks")
    st.caption("Review, correct, and finalize tag decisions on grouped stories that still need human judgment.")

    def build_all_tagging_candidates(unique_df: pd.DataFrame) -> pd.DataFrame:
        if unique_df is None or unique_df.empty:
            return pd.DataFrame()

        working = unique_df.copy()
        working["__effective_tag__"] = get_effective_tag_series(working).fillna("").astype(str).str.strip()
        working["__effective_ai_confidence__"] = get_effective_ai_tag_confidence_series(working)
        working["__assigned_blank__"] = working.get("Assigned Tag", pd.Series(index=working.index, dtype="object")).fillna("").astype(str).str.strip().eq("")
        working["__needs_review__"] = working.get("Needs Human Review", pd.Series(index=working.index, dtype="object")).fillna("").astype(str).str.strip().eq("Yes")
        working["__disagreement__"] = working.get("AI Tag Agreement", pd.Series(index=working.index, dtype="object")).fillna("").astype(str).str.strip().eq("Disagree")
        for col in ["Mentions", "Impressions", "Effective Reach"]:
            if col not in working.columns:
                working[col] = 0
            working[col] = pd.to_numeric(working[col], errors="coerce").fillna(0)
        working["__effective_ai_confidence__"] = pd.to_numeric(working["__effective_ai_confidence__"], errors="coerce").fillna(-1)
        return (
            working.sort_values(
                [
                    "__needs_review__",
                    "__disagreement__",
                    "__assigned_blank__",
                    "Mentions",
                    "Impressions",
                    "Effective Reach",
                    "__effective_ai_confidence__",
                ],
                ascending=[False, False, False, False, False, False, True],
            )
            .reset_index(drop=True)
        )

    all_coverage_candidates = build_all_tagging_candidates(st.session_state.df_tagging_unique)
    available_tag_buckets: list[str] = []
    if not all_coverage_candidates.empty and "__effective_tag__" in all_coverage_candidates.columns:
        seen_tags: set[str] = set()
        for value in all_coverage_candidates["__effective_tag__"].fillna("").astype(str):
            for tag_label in normalize_tag_list(value):
                key = tag_label.casefold()
                if key in seen_tags:
                    continue
                seen_tags.add(key)
                available_tag_buckets.append(tag_label)
        available_tag_buckets = sorted(available_tag_buckets, key=str.casefold)

    view_col1, view_col2 = st.columns([1.2, 1], gap="medium")
    with view_col1:
        review_mode = st.selectbox(
            "Spot check view",
            [
                "Flagged for human review",
                "Disagreements only",
                "All unresolved stories",
                "All Tagged Coverage",
                "All Coverage",
                "Specific Tag Bucket",
            ],
            key="tagging_review_mode",
        )
    selected_tag_bucket = None
    with view_col2:
        if review_mode == "Specific Tag Bucket":
            selected_tag_bucket = st.selectbox(
                "Tag bucket",
                available_tag_buckets or ["No labeled buckets available"],
                key="tagging_selected_bucket",
                disabled=not available_tag_buckets,
            )

    if review_mode == "All Tagged Coverage":
        review_source_df = all_tagged_candidates
    elif review_mode == "All Coverage":
        review_source_df = all_coverage_candidates
    elif review_mode == "Specific Tag Bucket":
        review_source_df = all_coverage_candidates
    else:
        review_source_df = review_base_candidates
    def _build_candidates_for_current_view(unique_df: pd.DataFrame) -> pd.DataFrame:
        base_review_df = compute_tag_review_candidates(unique_df, exclude_reviewed=False)
        tagged_df = compute_all_tagged_candidates(unique_df)
        coverage_df = build_all_tagging_candidates(unique_df)
        if review_mode == "All Tagged Coverage":
            source_df = tagged_df
        elif review_mode == "All Coverage":
            source_df = coverage_df
        elif review_mode == "Specific Tag Bucket":
            source_df = coverage_df
        else:
            source_df = base_review_df
        filtered = filter_tag_candidates_for_review_mode(source_df, review_mode)
        if review_mode == "Specific Tag Bucket":
            if selected_tag_bucket and selected_tag_bucket != "No labeled buckets available":
                selected_key = str(selected_tag_bucket).strip().casefold()
                filtered = filtered[
                    filtered.get("__effective_tag__", pd.Series(index=filtered.index, dtype="object"))
                    .fillna("")
                    .astype(str)
                    .map(lambda value: selected_key in {tag.casefold() for tag in normalize_tag_list(value)})
                ].copy()
            else:
                filtered = pd.DataFrame()
        return filtered

    candidates = _build_candidates_for_current_view(st.session_state.df_tagging_unique)

    summary1, summary2, summary3, summary4 = st.columns(4)
    with summary1:
        st.metric("Ready for review", f"{len(review_base_candidates):,}")
    with summary2:
        st.metric("Needs human review", f"{needs_review_count:,}")
    with summary3:
        st.metric("Disagreements", f"{disagreement_count:,}")
    with summary4:
        st.metric("In review queue", f"{len(candidates):,}")

    st.caption(
        "Flagged, Disagreements, and All unresolved keep the queue focused on review priority. "
        "All Coverage and bucket views broaden the browse set, but still sort unresolved and higher-visibility stories toward the top."
    )

    if candidates.empty:
        st.info("No grouped stories match the current view.")
        return

    st.session_state.tagging_review_idx = min(
        int(st.session_state.get("tagging_review_idx", 0)),
        len(candidates) - 1,
    )
    idx = int(st.session_state.tagging_review_idx)
    row = candidates.iloc[idx]
    current_group_id = int(row["Group ID"])

    def _safe_text(value) -> str:
        if pd.isna(value):
            return ""
        return str(value).strip()

    url = _safe_text(row.get("URL", ""))
    head_raw = _safe_text(row.get("Headline", ""))
    body_raw = _safe_text(row.get("Snippet", row.get("Example Snippet", "")))
    trans_head = _safe_text(row.get("Translated Headline", ""))
    trans_body = _safe_text(row.get("Translated Body", ""))
    headline = trans_head if trans_head else head_raw
    snippet = trans_body if trans_body else body_raw

    ai_tag = _safe_text(row.get("AI Tag", ""))
    ai_conf = pd.to_numeric(pd.Series([row.get("AI Tag Confidence")]), errors="coerce").iloc[0]
    ai_rsn = _safe_text(row.get("AI Tag Rationale", ""))
    review_tag = _safe_text(row.get("Review AI Tag", ""))
    review_conf = pd.to_numeric(pd.Series([row.get("Review AI Confidence")]), errors="coerce").iloc[0]
    review_rsn = _safe_text(row.get("Review AI Rationale", ""))
    agreement = _safe_text(row.get("AI Tag Agreement", ""))
    needs_review = _safe_text(row.get("Needs Human Review", ""))

    def _fmt_tag(label: str, conf) -> str:
        if not label.strip():
            return "Not available"
        if pd.notna(conf):
            return f"{label} ({int(conf)})"
        return label

    analysis_payload = get_analysis_context_payload(st.session_state)
    display_keywords = []
    display_keywords.extend(analysis_payload.get("primary_names", []))
    display_keywords.extend(analysis_payload.get("alternate_names", []))
    display_keywords.extend(analysis_payload.get("spokespeople", []))
    display_keywords.extend(analysis_payload.get("products", []))
    display_keywords.extend(analysis_payload.get("highlight_keywords", []))
    seen_cf: set[str] = set()
    keywords: list[str] = []
    for item in display_keywords:
        cleaned = _safe_text(item)
        if not cleaned:
            continue
        cf = cleaned.casefold()
        if cf in seen_cf:
            continue
        seen_cf.add(cf)
        keywords.append(cleaned)
    tolerant_pat_str = build_tolerant_regex_str(keywords)

    with st.sidebar:
        if st.button("Translate"):
            try:
                th = translate_text(head_raw) if head_raw else None
                tb = translate_text(body_raw) if body_raw else None
                unique2, rows2 = apply_translation_to_group(
                    st.session_state.df_tagging_unique,
                    st.session_state.df_tagging_rows,
                    current_group_id,
                    th,
                    tb,
                )
                sync_tagging_state(unique2, rows2)
                st.rerun()
            except Exception as e:
                st.error(f"Translation failed: {e}")

    left, right = st.columns([3.8, 1.4], gap="large")

    with left:
        highlighted_head = highlight_with_tolerant_regex(
            escape_markdown(headline),
            tolerant_pat_str,
            keywords,
        )
        highlighted_body = highlight_with_tolerant_regex(
            escape_markdown(snippet),
            tolerant_pat_str,
            keywords,
        )
        st.markdown(f"#### {highlighted_head}", unsafe_allow_html=True)
        if snippet:
            st.markdown(highlighted_body, unsafe_allow_html=True)
        st.divider()
        if url:
            st.markdown(url)
        meta_bits = [
            _safe_text(row.get("Date", "")),
            _safe_text(row.get("Outlet", "")),
            _safe_text(row.get("Type", "")),
            f"Mentions: {int(pd.to_numeric(pd.Series([row.get('Mentions', 0)]), errors='coerce').fillna(0).iloc[0]):,}",
            f"Impressions: {int(pd.to_numeric(pd.Series([row.get('Impressions', 0)]), errors='coerce').fillna(0).iloc[0]):,}",
        ]
        meta_line = " | ".join([part for part in meta_bits if part])
        if meta_line:
            st.caption(meta_line)

    with right:
        current_assignment_source = _safe_text(row.get("Assigned Tag Source", ""))
        if agreement == "Disagree":
            st.warning("AI opinions disagree.")
        elif current_assignment_source == "AI_AUTO_RESOLVED":
            st.success("Auto-resolved by AI (high confidence match).")
        elif needs_review == "Yes":
            st.info("Flagged for review.")

        st.write(f"**1st**: {_fmt_tag(ai_tag, ai_conf)}")
        if review_tag:
            st.write(f"**2nd**: {_fmt_tag(review_tag, review_conf)}")

        tag_names = list(st.session_state.get("tag_definitions", {}).keys())
        selected_label = None
        if tag_names:
            st.write("**Choose final tag**")
            if tagging_mode == "Multiple applicable tags":
                default_selected = normalize_tag_list(
                    _safe_text(row.get("Assigned Tag"))
                    or _safe_text(row.get("Review AI Tag"))
                    or _safe_text(row.get("AI Tag"))
                )
                for label in tag_names:
                    default_checked = label in default_selected
                    checkbox_key = f"tag_review_multi_{current_group_id}_{label}"
                    if checkbox_key not in st.session_state:
                        st.session_state[checkbox_key] = default_checked
                    st.checkbox(label, key=checkbox_key)
                if st.button("Save selected tags", key=f"tag_review_save_multi_{current_group_id}", use_container_width=True):
                    chosen = [
                        label
                        for label in tag_names
                        if st.session_state.get(f"tag_review_multi_{current_group_id}_{label}", False)
                    ]
                    selected_label = ", ".join(chosen) if chosen else None
            else:
                for label in tag_names:
                    if st.button(label, key=f"tag_review_assign_{current_group_id}_{label}", use_container_width=True):
                        selected_label = label

        if st.button("Re-run review AI", use_container_width=True, disabled=not bool(ai_tag)):
            tag_definitions = st.session_state.get("tag_definitions", {})
            rerun_candidates = candidates[candidates["Group ID"] == current_group_id].copy()
            unique2, rows2, _, total_in, total_out, _ = run_batch_tag_second_opinion(
                candidates_df=rerun_candidates,
                df_unique=st.session_state.df_tagging_unique,
                df_grouped=st.session_state.df_tagging_rows,
                tag_definitions=tag_definitions,
                tagging_mode=tagging_mode,
                api_key=st.secrets["key"],
                review_model=DEFAULT_TAGGING_REVIEW_MODEL,
                limit=1,
                max_workers=1,
                low_conf_threshold=int(
                    st.session_state.get(
                        "tagging_review_low_conf_threshold",
                        DEFAULT_TAGGING_REVIEW_CONFIDENCE_THRESHOLD,
                    )
                ),
            )
            sync_tagging_state(unique2, rows2)
            apply_usage_to_session(total_in, total_out, DEFAULT_TAGGING_REVIEW_MODEL)
            st.rerun()

        if ai_rsn or review_rsn:
            with st.expander("AI opinion reasoning", expanded=False):
                if ai_rsn:
                    st.write("**1st AI rationale**")
                    st.write(ai_rsn)
                if review_rsn:
                    st.write("**2nd AI rationale**")
                    st.write(review_rsn)

        st.divider()
        nav1, nav2 = st.columns(2)
        with nav1:
            if st.button("", disabled=(idx <= 0), use_container_width=True, icon=":material/skip_previous:", help="Previous story"):
                st.session_state.tagging_review_idx = max(0, idx - 1)
                st.rerun()
        with nav2:
            if st.button("", disabled=(idx >= len(candidates) - 1), use_container_width=True, icon=":material/skip_next:", help="Next story"):
                st.session_state.tagging_review_idx = min(len(candidates) - 1, idx + 1)
                st.rerun()
        st.caption(f"Story {idx + 1} of {len(candidates)} in current view")

    if selected_label is not None:
        unique2, rows2 = set_assigned_tag(
            st.session_state.df_tagging_unique,
            st.session_state.df_tagging_rows,
            current_group_id,
            ", ".join(normalize_tag_list(selected_label)),
        )
        sync_tagging_state(unique2, rows2)

        new_filtered = _build_candidates_for_current_view(st.session_state.df_tagging_unique)
        if new_filtered.empty:
            st.rerun()
        else:
            st.session_state.tagging_review_idx = min(idx, len(new_filtered) - 1)
            st.rerun()
