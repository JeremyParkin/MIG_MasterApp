from __future__ import annotations


def render_tagging_review_page(*, review_stage: str) -> None:
    from datetime import datetime

    import pandas as pd
    import streamlit as st

    from processing.analysis_context import get_analysis_context_payload
    from processing.ai_tagging import (
        compute_all_tagged_candidates,
        DEFAULT_TAGGING_REVIEW_BATCH_SIZE,
        DEFAULT_TAGGING_REVIEW_CONFIDENCE_THRESHOLD,
        DEFAULT_TAGGING_REVIEW_MODEL,
        DEFAULT_TAGGING_MAX_WORKERS,
        compute_tag_review_candidates,
        ensure_tag_review_columns,
        filter_tag_candidates_for_review_mode,
        normalize_tag_list,
        run_batch_tag_second_opinion,
        set_assigned_tag,
    )
    from processing.sentiment_config import build_tolerant_regex_str
    from processing.spot_checks import escape_markdown, highlight_with_tolerant_regex
    from utils.api_meter import apply_usage_to_session

    def sync_tagging_state(unique_df: pd.DataFrame, rows_df: pd.DataFrame) -> None:
        st.session_state.df_tagging_unique = unique_df
        st.session_state.df_tagging_rows = rows_df

    df_unique, df_rows = ensure_tag_review_columns(
        st.session_state.df_tagging_unique,
        st.session_state.df_tagging_rows,
    )
    sync_tagging_state(df_unique, df_rows)
    tagging_mode = st.session_state.get("tagging_mode", "Single best tag")

    base_candidates = compute_tag_review_candidates(st.session_state.df_tagging_unique)
    all_tagged_candidates = compute_all_tagged_candidates(st.session_state.df_tagging_unique)
    reviewed_candidates = (
        base_candidates[base_candidates["Review AI Tag"].notna()].copy()
        if "Review AI Tag" in base_candidates.columns
        else base_candidates.iloc[0:0].copy()
    )
    disagreement_count = int(
        (reviewed_candidates.get("AI Tag Agreement", pd.Series(dtype="object")).fillna("") == "Disagree").sum()
    ) if not reviewed_candidates.empty else 0
    needs_review_count = int(
        (reviewed_candidates.get("Needs Human Review", pd.Series(dtype="object")).fillna("") == "Yes").sum()
    ) if not reviewed_candidates.empty else 0

    pre_review_message = st.session_state.get("tagging_pre_review_message")
    if pre_review_message and review_stage == "pre_review":
        st.success(pre_review_message)
        st.session_state.tagging_pre_review_message = None

    if review_stage == "pre_review":
        st.subheader("Step 3: AI Pre-Review")
        st.caption("Run a second AI pass on top tagging candidates so likely agreements can be auto-resolved before human spot checks.")

        if base_candidates.empty:
            st.success("All set — no remaining grouped stories need tag review.")
            summary1, summary2, summary3 = st.columns(3)
            with summary1:
                st.metric("Ready for review", "0")
            with summary2:
                st.metric("Needs human review", f"{needs_review_count:,}")
            with summary3:
                st.metric("Disagreements", f"{disagreement_count:,}")
            return

        adv1, adv2 = st.columns(2)
        with adv1:
            st.number_input(
                "Top candidates to send for AI pre-review",
                min_value=1,
                max_value=min(len(base_candidates), 200),
                value=min(
                    int(st.session_state.get("tagging_pre_review_n", DEFAULT_TAGGING_REVIEW_BATCH_SIZE)),
                    len(base_candidates),
                ),
                step=1,
                key="tagging_pre_review_n",
            )
        with adv2:
            st.number_input(
                "Review confidence cutoff",
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

        action1, action2 = st.columns(2)
        with action1:
            if st.button("Run AI pre-review", type="secondary", use_container_width=True):
                tag_definitions = st.session_state.get("tag_definitions", {})
                st.session_state.pop("tagging_pre_review_message", None)
                with st.spinner("Running AI second opinions on top tagging candidates..."):
                    unique2, rows2, batch_errors, total_in, total_out = run_batch_tag_second_opinion(
                        candidates_df=base_candidates,
                        df_unique=st.session_state.df_tagging_unique,
                        df_grouped=st.session_state.df_tagging_rows,
                        tag_definitions=tag_definitions,
                        tagging_mode=tagging_mode,
                        api_key=st.secrets["key"],
                        review_model=DEFAULT_TAGGING_REVIEW_MODEL,
                        limit=int(st.session_state.get("tagging_pre_review_n", min(50, len(base_candidates)))),
                        max_workers=DEFAULT_TAGGING_MAX_WORKERS,
                        low_conf_threshold=int(
                            st.session_state.get(
                                "tagging_review_low_conf_threshold",
                                DEFAULT_TAGGING_REVIEW_CONFIDENCE_THRESHOLD,
                            )
                        ),
                    )
                sync_tagging_state(unique2, rows2)
                apply_usage_to_session(total_in, total_out, DEFAULT_TAGGING_REVIEW_MODEL)
                completed_at = datetime.now().astimezone().strftime("%b %-d, %Y at %-I:%M %p")
                st.session_state.tagging_pre_review_message = (
                    f"AI pre-review batch completed {completed_at}. Review counts updated below."
                )
                st.session_state["__last_tagging_pre_review_summary__"] = {
                    "errors": batch_errors,
                }
                st.rerun()
        with action2:
            st.caption("High-confidence matching tag opinions are auto-assigned during AI pre-review.")

        summary1, summary2, summary3 = st.columns(3)
        with summary1:
            st.metric("Ready for review", f"{len(base_candidates):,}")
        with summary2:
            st.metric("Needs human review", f"{needs_review_count:,}")
        with summary3:
            st.metric("Disagreements", f"{disagreement_count:,}")

        last_summary = st.session_state.get("__last_tagging_pre_review_summary__")
        if last_summary and last_summary.get("errors"):
            with st.expander(f"Completed with {len(last_summary['errors'])} error(s)", expanded=False):
                for err in last_summary["errors"]:
                    st.write(err)
        return

    st.subheader("Step 4: Tag Spot Checks")
    st.caption("Review, correct, and finalize tag decisions on grouped stories that still need human judgment.")

    review_mode = st.radio(
        "Review queue",
        ["Flagged for human review", "Disagreements only", "All unresolved stories", "All Tagged Coverage"],
        horizontal=True,
        label_visibility="collapsed",
        key="tagging_review_mode",
    )
    review_source_df = all_tagged_candidates if review_mode == "All Tagged Coverage" else base_candidates
    candidates = filter_tag_candidates_for_review_mode(review_source_df, review_mode)

    summary1, summary2, summary3, summary4 = st.columns(4)
    with summary1:
        st.metric("Ready for review", f"{len(base_candidates):,}")
    with summary2:
        st.metric("Needs human review", f"{needs_review_count:,}")
    with summary3:
        st.metric("Disagreements", f"{disagreement_count:,}")
    with summary4:
        st.metric("In review queue", f"{len(candidates):,}")

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
    url = str(row.get("URL", "") or "").strip()
    headline = str(row.get("Headline", "") or "").strip()
    snippet = str(row.get("Snippet", row.get("Example Snippet", "")) or "").strip()

    ai_tag = str(row.get("AI Tag", "") or "").strip()
    ai_conf = pd.to_numeric(pd.Series([row.get("AI Tag Confidence")]), errors="coerce").iloc[0]
    ai_rsn = str(row.get("AI Tag Rationale", "") or "").strip()
    review_tag = str(row.get("Review AI Tag", "") or "").strip()
    review_conf = pd.to_numeric(pd.Series([row.get("Review AI Confidence")]), errors="coerce").iloc[0]
    review_rsn = str(row.get("Review AI Rationale", "") or "").strip()
    agreement = str(row.get("AI Tag Agreement", "") or "").strip()
    needs_review = str(row.get("Needs Human Review", "") or "").strip()

    def _fmt_tag(label: str, conf) -> str:
        if not label.strip():
            return "Not available"
        if pd.notna(conf):
            return f"{label} ({int(conf)})"
        return label

    def _safe_text(value) -> str:
        if pd.isna(value):
            return ""
        return str(value or "").strip()

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
        cleaned = str(item or "").strip()
        if not cleaned:
            continue
        cf = cleaned.casefold()
        if cf in seen_cf:
            continue
        seen_cf.add(cf)
        keywords.append(cleaned)
    tolerant_pat_str = build_tolerant_regex_str(keywords)

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
        st.markdown(f"### {highlighted_head}", unsafe_allow_html=True)
        if snippet:
            st.markdown(highlighted_body, unsafe_allow_html=True)
        st.divider()
        if url:
            st.markdown(url)
        meta_bits = [
            str(row.get("Date", "") or "").strip(),
            str(row.get("Outlet", "") or "").strip(),
            str(row.get("Type", "") or "").strip(),
            f"Mentions: {int(pd.to_numeric(pd.Series([row.get('Mentions', 0)]), errors='coerce').fillna(0).iloc[0]):,}",
            f"Impressions: {int(pd.to_numeric(pd.Series([row.get('Impressions', 0)]), errors='coerce').fillna(0).iloc[0]):,}",
        ]
        meta_line = " | ".join([part for part in meta_bits if part])
        if meta_line:
            st.caption(meta_line)

    with right:
        if agreement == "Disagree":
            st.warning("AI opinions disagree.")
        elif agreement == "Match" and needs_review != "Yes":
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
            unique2, rows2, _, total_in, total_out = run_batch_tag_second_opinion(
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

        new_base = compute_tag_review_candidates(st.session_state.df_tagging_unique)
        new_all_tagged = compute_all_tagged_candidates(st.session_state.df_tagging_unique)
        new_filtered = filter_tag_candidates_for_review_mode(
            new_all_tagged if review_mode == "All Tagged Coverage" else new_base,
            review_mode,
        )
        if new_filtered.empty:
            st.rerun()
        else:
            st.session_state.tagging_review_idx = min(idx, len(new_filtered) - 1)
            st.rerun()
