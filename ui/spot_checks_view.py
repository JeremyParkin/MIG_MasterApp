from __future__ import annotations

def render_spot_checks_page(*, embedded_review: bool | None = None, spot_checks_mode: str | None = None) -> None:
    import streamlit as st
    if embedded_review is not None:
        st.session_state["sentiment_review_embedded"] = embedded_review
    if spot_checks_mode is not None:
        st.session_state["spot_checks_mode"] = spot_checks_mode
    # 10-Spot_Checks.py
    
    
    import warnings
    
    import pandas as pd
    import streamlit as st
    from processing.analysis_context import get_analysis_context_payload
    from processing.ai_sentiment import (
        build_effective_ai_sentiment_confidence_series,
        build_effective_ai_sentiment_series,
        build_sentiment_distribution,
    )
    from processing.spot_checks import (
        DEFAULT_CONF_THRESH,
        DEFAULT_SECOND_OPINION_MODEL,
        DEFAULT_REVIEW_CONFIDENCE_THRESHOLD,
        recommend_second_opinion_batch_size,
        init_spot_check_state,
        escape_markdown,
        highlight_with_tolerant_regex,
        translate_text,
        apply_translation_to_group,
        compute_candidates,
        build_story_prompt,
        call_ai_sentiment,
        write_second_opinion_to_group,
        set_assigned_sentiment,
        clear_assigned_sentiment,
        update_acceptance_tracking,
        ensure_review_columns,
        run_batch_second_opinion,
    )
    from utils.api_meter import (
        init_api_meter,
        apply_usage_to_session,
        estimate_cost_usd,
        get_api_cost_usd,
    )
    from utils.time_display import format_local_timestamp
    
    warnings.filterwarnings("ignore")
    
    st.markdown(
        """
        <style>
        .block-container{padding-top:2.75rem !important;}
        [class*="spot_btn_positive"] button,
        [class*="spot_btn_very_positive"] button,
        [class*="spot_btn_somewhat_positive"] button,
        [class*="spot_btn_neutral"] button,
        [class*="spot_btn_somewhat_negative"] button,
        [class*="spot_btn_negative"] button,
        [class*="spot_btn_very_negative"] button,
        [class*="spot_btn_not_relevant"] button {
            width: 100%;
            color: black !important;
            border: 0 !important;
            padding: 0.16rem 0.6rem !important;
            font-weight: 700 !important;
            font-size: 14px !important;
            border-radius: 5px !important;
            margin-bottom: 0 !important;
            box-shadow: none !important;
        }
        [class*="spot_btn_positive"],
        [class*="spot_btn_very_positive"],
        [class*="spot_btn_somewhat_positive"],
        [class*="spot_btn_neutral"],
        [class*="spot_btn_somewhat_negative"],
        [class*="spot_btn_negative"],
        [class*="spot_btn_very_negative"],
        [class*="spot_btn_not_relevant"] {
            margin-bottom: 0 !important;
            margin-bottom: -5px !important;
        }
        [class*="spot_btn_positive"] div[data-testid="stButton"],
        [class*="spot_btn_very_positive"] div[data-testid="stButton"],
        [class*="spot_btn_somewhat_positive"] div[data-testid="stButton"],
        [class*="spot_btn_neutral"] div[data-testid="stButton"],
        [class*="spot_btn_somewhat_negative"] div[data-testid="stButton"],
        [class*="spot_btn_negative"] div[data-testid="stButton"],
        [class*="spot_btn_very_negative"] div[data-testid="stButton"],
        [class*="spot_btn_not_relevant"] div[data-testid="stButton"] {
            margin: 0 !important;
        }
        [class*="spot_btn_positive"] div[data-testid="stVerticalBlock"],
        [class*="spot_btn_very_positive"] div[data-testid="stVerticalBlock"],
        [class*="spot_btn_somewhat_positive"] div[data-testid="stVerticalBlock"],
        [class*="spot_btn_neutral"] div[data-testid="stVerticalBlock"],
        [class*="spot_btn_somewhat_negative"] div[data-testid="stVerticalBlock"],
        [class*="spot_btn_negative"] div[data-testid="stVerticalBlock"],
        [class*="spot_btn_very_negative"] div[data-testid="stVerticalBlock"],
        [class*="spot_btn_not_relevant"] div[data-testid="stVerticalBlock"] {
            gap: 0 !important;
        }
        [class*="spot_btn_positive"] button { background-color: #2ecc71 !important; }
        [class*="spot_btn_very_positive"] button { background-color: #10ad82 !important; }
        [class*="spot_btn_somewhat_positive"] button { background-color: #72cc4a !important; }
        [class*="spot_btn_neutral"] button { background-color: #f1c40f !important; }
        [class*="spot_btn_somewhat_negative"] button { background-color: #e67e22 !important; }
        [class*="spot_btn_negative"] button { background-color: #e74c3c !important; }
        [class*="spot_btn_very_negative"] button { background-color: #c0392b !important; }
        [class*="spot_btn_not_relevant"] button { background-color: #7f8c8d !important; }
        [class*="spot_btn_positive"] button:hover,
        [class*="spot_btn_very_positive"] button:hover,
        [class*="spot_btn_somewhat_positive"] button:hover,
        [class*="spot_btn_neutral"] button:hover,
        [class*="spot_btn_somewhat_negative"] button:hover,
        [class*="spot_btn_negative"] button:hover,
        [class*="spot_btn_very_negative"] button:hover,
        [class*="spot_btn_not_relevant"] button:hover {
            color: black !important;
            filter: brightness(0.98);
        }
        [class*="spot_btn_positive"] div[data-testid="element-container"],
        [class*="spot_btn_very_positive"] div[data-testid="element-container"],
        [class*="spot_btn_somewhat_positive"] div[data-testid="element-container"],
        [class*="spot_btn_neutral"] div[data-testid="element-container"],
        [class*="spot_btn_somewhat_negative"] div[data-testid="element-container"],
        [class*="spot_btn_negative"] div[data-testid="element-container"],
        [class*="spot_btn_very_negative"] div[data-testid="element-container"],
        [class*="spot_btn_not_relevant"] div[data-testid="element-container"] {
            margin: 0 !important;
            padding: 0 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    embedded_review = bool(st.session_state.get("sentiment_review_embedded", False))
    spot_checks_mode = str(st.session_state.get("spot_checks_mode", "full") or "full")
    if embedded_review:
        if spot_checks_mode == "pre_review":
            st.subheader("Step 3: AI Second Opinion")
            st.caption("Generate one second AI opinion per already-labeled story group so likely high-confidence matches can be auto-resolved before human spot checks.")
        else:
            st.subheader("Step 4: Spot Checks")
            st.caption("Review, correct, and finalize sentiment decisions on grouped stories that still need human judgment.")
    else:
        st.title("Spot Checks")
        st.caption("Review, correct, and finalize sentiment decisions on grouped stories before exporting the finished dataset.")
    
    if not st.session_state.get("sentiment_config_step", False):
        st.error("Please complete AI Sentiment setup before trying this step.")
        st.stop()
    
    if not isinstance(st.session_state.get("df_sentiment_unique"), pd.DataFrame):
        st.error("Sentiment unique stories not found. Please complete earlier steps.")
        st.stop()
    
    init_spot_check_state(st.session_state)
    init_api_meter()
    
    
    def colored_button(label: str) -> bool:
        stable_key = f"spot_manual_{label.replace(' ', '_')}"
        style_key = stable_key.replace("spot_manual_", "spot_btn_").replace(" ", "_").lower()
        with st.container(key=style_key):
            return st.button(label, key=stable_key, use_container_width=True)
    
    
    def sync_sentiment_state(unique_df: pd.DataFrame, grouped_df: pd.DataFrame) -> None:
        translation_updates = pd.DataFrame()
        if isinstance(grouped_df, pd.DataFrame) and not grouped_df.empty and "Group ID" in grouped_df.columns:
            translation_cols = [col for col in ["Translated Headline", "Translated Body"] if col in grouped_df.columns]
            if translation_cols:
                translation_updates = grouped_df[["Group ID", *translation_cols]].copy()
                translation_updates = translation_updates.drop_duplicates(subset=["Group ID"], keep="last")

        def apply_translation_updates(target_df: pd.DataFrame | None) -> pd.DataFrame | None:
            if not isinstance(target_df, pd.DataFrame) or target_df.empty:
                return target_df
            if translation_updates.empty or "Group ID" not in target_df.columns:
                return target_df

            out = target_df.copy()
            for col in translation_updates.columns:
                if col == "Group ID":
                    continue
                if col not in out.columns:
                    out[col] = pd.NA
                value_map = translation_updates.set_index("Group ID")[col].to_dict()
                out[col] = out["Group ID"].map(value_map).combine_first(out[col])
            return out

        st.session_state.df_sentiment_unique = unique_df
        st.session_state.df_sentiment_grouped_rows = grouped_df
        st.session_state.df_sentiment_rows = grouped_df.copy()
        st.session_state.df_traditional = apply_translation_updates(st.session_state.get("df_traditional"))
        if "df_ai_grouped" in st.session_state:
            st.session_state.df_ai_grouped = apply_translation_updates(st.session_state.get("df_ai_grouped"))
    
    
    def filter_candidates_for_review_mode(
        candidates_df: pd.DataFrame,
        review_mode: str,
        low_conf_threshold: int,
    ) -> pd.DataFrame:
        if candidates_df.empty:
            return candidates_df
    
        out = candidates_df.copy()
    
        if review_mode == "Flagged for human review":
            if "Needs Human Review" in out.columns:
                flagged = out[out["Needs Human Review"] == "Yes"].copy()
                if not flagged.empty:
                    return flagged
            return out

        if review_mode == "Disagreements only":
            if "AI Agreement" in out.columns:
                disagreements = out[out["AI Agreement"].fillna("").astype(str).str.strip() == "Disagree"].copy()
                if not disagreements.empty:
                    return disagreements
            return out

        if review_mode == "All Toned Coverage":
            return out

        return out


    def build_all_toned_candidates(unique_df: pd.DataFrame) -> pd.DataFrame:
        if unique_df is None or unique_df.empty:
            return pd.DataFrame()

        working = unique_df.copy()
        assigned = working.get("Assigned Sentiment", pd.Series(index=working.index, dtype="object")).fillna("").astype(str).str.strip()
        effective_ai = build_effective_ai_sentiment_series(working)
        final_label = assigned.where(assigned != "", effective_ai).fillna("").astype(str).str.strip().str.upper()
        pool = working[final_label != ""].copy()
        if pool.empty:
            return pool

        pool["__effective_sentiment__"] = final_label.loc[pool.index]
        pool["__effective_ai_confidence__"] = build_effective_ai_sentiment_confidence_series(working).loc[pool.index]
        pool["__assigned_blank__"] = assigned.loc[pool.index].eq("")
        pool["__needs_review__"] = working.get("Needs Human Review", pd.Series(index=working.index, dtype="object")).fillna("").astype(str).str.strip().eq("Yes").loc[pool.index]
        pool["__disagreement__"] = working.get("AI Agreement", pd.Series(index=working.index, dtype="object")).fillna("").astype(str).str.strip().eq("Disagree").loc[pool.index]

        for col in ["Mentions", "Impressions", "Effective Reach"]:
            if col not in pool.columns:
                pool[col] = 0
            pool[col] = pd.to_numeric(pool[col], errors="coerce").fillna(0)

        pool["__effective_ai_confidence__"] = pd.to_numeric(pool["__effective_ai_confidence__"], errors="coerce").fillna(-1)
        pool = (
            pool.sort_values(
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
        return pool


    def build_all_sentiment_candidates(unique_df: pd.DataFrame) -> pd.DataFrame:
        if unique_df is None or unique_df.empty:
            return pd.DataFrame()

        working = unique_df.copy()
        assigned = working.get("Assigned Sentiment", pd.Series(index=working.index, dtype="object")).fillna("").astype(str).str.strip()
        effective_ai = build_effective_ai_sentiment_series(working)
        final_label = assigned.where(assigned != "", effective_ai).fillna("").astype(str).str.strip().str.upper()
        working["__effective_sentiment__"] = final_label
        working["__effective_ai_confidence__"] = build_effective_ai_sentiment_confidence_series(working)
        working["__assigned_blank__"] = assigned.eq("")
        working["__needs_review__"] = working.get("Needs Human Review", pd.Series(index=working.index, dtype="object")).fillna("").astype(str).str.strip().eq("Yes")
        working["__disagreement__"] = working.get("AI Agreement", pd.Series(index=working.index, dtype="object")).fillna("").astype(str).str.strip().eq("Disagree")
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
    
    
    # def build_sentiment_distribution(df_unique: pd.DataFrame, sentiment_type: str) -> pd.DataFrame:
    #     if sentiment_type == "5-way":
    #         order = [
    #             "VERY POSITIVE",
    #             "SOMEWHAT POSITIVE",
    #             "NEUTRAL",
    #             "SOMEWHAT NEGATIVE",
    #             "VERY NEGATIVE",
    #             "NOT RELEVANT",
    #             # "UNASSIGNED",
    #         ]
    #     else:
    #         order = [
    #             "POSITIVE",
    #             "NEUTRAL",
    #             "NEGATIVE",
    #             "NOT RELEVANT",
    #             # "UNASSIGNED",
    #         ]
    #
    #     assigned = df_unique.get("Assigned Sentiment", pd.Series(index=df_unique.index, dtype="object")).fillna("").astype(str).str.strip()
    #     ai = df_unique.get("AI Sentiment", pd.Series(index=df_unique.index, dtype="object")).fillna("").astype(str).str.strip()
    #
    #     final = assigned.where(assigned != "", ai)
    #     final = final.where(final != "", "UNASSIGNED").str.upper()
    #
    #     sentiment_counts = final.value_counts().rename_axis("Sentiment").reset_index(name="Count")
    #     base = pd.DataFrame({"Sentiment": order})
    #     out = base.merge(sentiment_counts, on="Sentiment", how="left")
    #     out["Count"] = out["Count"].fillna(0).astype(int)
    #     total = int(out["Count"].sum())
    #     out["Share"] = out["Count"] / total if total > 0 else 0
    #     return out
    
    
    def auto_accept_high_confidence_matches(
        unique_df: pd.DataFrame,
        grouped_df: pd.DataFrame,
        confidence_threshold: int,
    ) -> tuple[pd.DataFrame, pd.DataFrame, int, int]:
        unique = unique_df.copy()
        grouped = grouped_df.copy()
    
        required_cols = [
            "Group ID",
            "AI Sentiment",
            "AI Sentiment Confidence",
            "Review AI Sentiment",
            "Review AI Confidence",
            "AI Agreement",
            "Assigned Sentiment",
            "Assigned Sentiment Source",
        ]
        for col in required_cols:
            if col not in unique.columns:
                return unique, grouped, 0, 0

        ai_label = unique["AI Sentiment"].fillna("").astype(str).str.strip().str.upper()
        review_label = unique["Review AI Sentiment"].fillna("").astype(str).str.strip().str.upper()
        assigned = unique["Assigned Sentiment"].fillna("").astype(str).str.strip()
        assigned_source = unique["Assigned Sentiment Source"].fillna("").astype(str).str.strip()

        review_conf = pd.to_numeric(unique["Review AI Confidence"], errors="coerce")

        qualifying_mask = (
            (unique["AI Agreement"].fillna("") == "Match")
            & (ai_label != "")
            & (ai_label == review_label)
            & review_conf.ge(confidence_threshold)
        )
        assign_mask = (assigned == "") & qualifying_mask
        stale_mask = (assigned_source == "AI_AUTO_RESOLVED") & ~qualifying_mask

        reopened_count = int(stale_mask.sum())
        for gid in unique.loc[stale_mask, "Group ID"].tolist():
            unique, grouped = clear_assigned_sentiment(unique, grouped, gid)

        accepted_count = int(assign_mask.sum())
        gids_to_accept = unique.loc[assign_mask, "Group ID"].tolist()

        for gid in gids_to_accept:
            label = unique.loc[unique["Group ID"] == gid, "AI Sentiment"].iloc[0]
            unique, grouped = set_assigned_sentiment(unique, grouped, gid, label, source="AI_AUTO_RESOLVED")
            update_acceptance_tracking(st.session_state, gid, str(label).strip())

        return unique, grouped, accepted_count, reopened_count
    
    
    df_unique = st.session_state.df_sentiment_unique.copy()
    df_grouped = st.session_state.df_sentiment_grouped_rows.copy()
    df_unique, df_grouped = ensure_review_columns(df_unique, df_grouped)
    sync_sentiment_state(df_unique, df_grouped)
    
    if df_unique.empty:
        st.error("No sentiment-grouped stories found.")
        st.stop()
    
    analysis_payload = get_analysis_context_payload(st.session_state)
    sentiment_entity_terms: list[str] = []
    primary_name = str(analysis_payload.get("primary_name", "") or "").strip()
    if primary_name:
        sentiment_entity_terms.append(primary_name)
    sentiment_entity_terms.extend(analysis_payload.get("alternate_names", []))
    sentiment_entity_terms.extend(analysis_payload.get("spokespeople", []))
    sentiment_entity_terms.extend(analysis_payload.get("products", []))
    seen_entity_terms: set[str] = set()
    sentiment_entity_terms = [
        text
        for text in [
            str(value or "").strip()
            for value in sentiment_entity_terms
        ]
        if text and not (text.casefold() in seen_entity_terms or seen_entity_terms.add(text.casefold()))
    ]

    pre_prompt = st.session_state.get("pre_prompt", "")
    post_prompt = st.session_state.get("post_prompt", "")
    sentiment_instruction = st.session_state.get("sentiment_instruction", "")
    functions = st.session_state.get("functions", [])
    model_id = st.session_state.get("model_choice", "gpt-5.4-nano")
    
    _raw_st = st.session_state.get("sentiment_type", "3-way")
    _s = str(_raw_st).strip().lower()
    sentiment_type = "5-way" if _s.startswith("5") or "5-way" in _s else "3-way"
    
    display_keywords: list[str] = []
    primary_name = str(analysis_payload.get("primary_name", "") or "").strip()
    if primary_name:
        display_keywords.append(primary_name)
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
    from processing.sentiment_config import build_tolerant_regex_str
    tolerant_pat_str = build_tolerant_regex_str(keywords)
    
    review_base_candidates = compute_candidates(
        df_unique=st.session_state.df_sentiment_unique,
        df_grouped=st.session_state.df_sentiment_grouped_rows,
        sentiment_type=sentiment_type,
        conf_thresh=DEFAULT_CONF_THRESH,
    )
    base_candidates = compute_candidates(
        df_unique=st.session_state.df_sentiment_unique,
        df_grouped=st.session_state.df_sentiment_grouped_rows,
        sentiment_type=sentiment_type,
        conf_thresh=DEFAULT_CONF_THRESH,
        exclude_reviewed=True,
    )
    all_toned_candidates = build_all_toned_candidates(st.session_state.df_sentiment_unique)

    ai_label_all = df_unique.get("AI Sentiment", pd.Series(index=df_unique.index, dtype="object")).fillna("").astype(str).str.strip().str.upper()
    review_label_all = df_unique.get("Review AI Sentiment", pd.Series(index=df_unique.index, dtype="object")).fillna("").astype(str).str.strip().str.upper()
    assigned_all = df_unique.get("Assigned Sentiment", pd.Series(index=df_unique.index, dtype="object")).fillna("").astype(str).str.strip()
    assigned_source_all = df_unique.get("Assigned Sentiment Source", pd.Series(index=df_unique.index, dtype="object")).fillna("").astype(str).str.strip()
    agreement_all = df_unique.get("AI Agreement", pd.Series(index=df_unique.index, dtype="object")).fillna("").astype(str).str.strip()
    needs_review_all = df_unique.get("Needs Human Review", pd.Series(index=df_unique.index, dtype="object")).fillna("").astype(str).str.strip()

    with_first_opinion_count = int((ai_label_all != "").sum())
    with_second_opinion_count = int((review_label_all != "").sum())
    eligible_second_opinion_count = int(((assigned_all == "") & (ai_label_all != "") & (review_label_all == "")).sum())
    auto_resolved_count = int((assigned_source_all == "AI_AUTO_RESOLVED").sum())
    needs_review_count = int(((review_label_all != "") & (assigned_all == "") & (needs_review_all == "Yes")).sum())
    disagreement_count = int(((review_label_all != "") & (agreement_all == "Disagree")).sum())
    if sentiment_type == "3-way":
        negative_labels = {"NEGATIVE"}
    else:
        negative_labels = {"SOMEWHAT NEGATIVE", "VERY NEGATIVE"}
    sensitive_eligible_count = int(((assigned_all == "") & (review_label_all == "") & ai_label_all.isin(negative_labels)).sum())

    checked = len(st.session_state.spot_checked_groups)
    accepted = len(st.session_state.accepted_initial)
    acceptance_rate = (accepted / checked) if checked else 0.0
    
    
    pre_review_message = st.session_state.get("spotcheck_pre_review_message")
    if pre_review_message and spot_checks_mode == "pre_review":
        st.success(pre_review_message)
        st.session_state.spotcheck_pre_review_message = None

    auto_resolve_message = st.session_state.get("spotcheck_auto_resolve_message")
    if auto_resolve_message and spot_checks_mode == "pre_review":
        st.success(auto_resolve_message)
        st.session_state.spotcheck_auto_resolve_message = None
    
    if base_candidates.empty:
        if spot_checks_mode == "pre_review":
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
            st.stop()
        small1, small2 = st.columns(2)
        with small1:
            st.metric("Spot-checked", checked)
        with small2:
            st.metric("Acceptance rate", f"{acceptance_rate:.0%}")
        st.stop()
    
    if spot_checks_mode in {"full", "pre_review"}:
        controls_in_expander = spot_checks_mode == "full"
    
        def render_pre_review_controls() -> None:
            stored_source_count = int(st.session_state.get("sentiment_second_opinion_target_source_count", 0) or 0)
            stored_target = int(st.session_state.get("sentiment_second_opinion_target_batch", 0) or 0)
            if stored_target <= 0 or with_first_opinion_count > stored_source_count:
                stored_target = recommend_second_opinion_batch_size(len(base_candidates))
                st.session_state.sentiment_second_opinion_target_batch = stored_target
                st.session_state.sentiment_second_opinion_target_source_count = with_first_opinion_count
            completed_second_opinion_count = with_second_opinion_count
            remaining_recommended = max(0, stored_target - completed_second_opinion_count)
            recommended_batch = min(len(base_candidates), remaining_recommended)
            default_batch_size = min(
                int(st.session_state.get("spotcheck_auto_review_n", recommended_batch or min(50, len(base_candidates)))),
                len(base_candidates),
            )
            if recommended_batch == 0 and len(base_candidates) > 0:
                default_batch_size = min(default_batch_size or min(10, len(base_candidates)), len(base_candidates))
            st.caption("Second-opinion priority favors more syndicated, higher-visibility, lower-confidence stories. Negative sentiment stories also get an extra boost.")

            batch_col1, batch_col2 = st.columns([1.25, 1], gap="medium")
            with batch_col1:
                selected_batch_size = st.number_input(
                    "Stories to send for AI second opinion",
                    min_value=1,
                    max_value=min(len(base_candidates), 200),
                    value=max(1, default_batch_size),
                    step=1,
                    key="spotcheck_auto_review_n",
                )
            with batch_col2:
                st.metric("Recommended batch", f"{recommended_batch:,}")
                if recommended_batch > 0:
                    st.caption(f"Selected for this batch: {selected_batch_size:,} of {len(base_candidates):,} eligible groups.")
                else:
                    st.caption("The current recommended second-opinion coverage has already been reached. You can still run more manually if desired.")

            with st.expander("Advanced auto-resolve options", expanded=False):
                threshold_value = st.number_input(
                    "Auto-resolve confidence threshold",
                    min_value=1,
                    max_value=100,
                    value=int(st.session_state.get("spotcheck_low_conf_threshold", DEFAULT_REVIEW_CONFIDENCE_THRESHOLD)),
                    step=1,
                    key="spotcheck_low_conf_threshold",
                )
                st.caption("This only controls whether matching first and second AI opinions are auto-resolved. It does not control which stories are selected for second opinion.")
                if st.button("Apply threshold to reviewed matches", key="spotcheck_apply_auto_resolve_threshold", use_container_width=True):
                    unique2, grouped2, accepted_count, reopened_count = auto_accept_high_confidence_matches(
                        st.session_state.df_sentiment_unique,
                        st.session_state.df_sentiment_grouped_rows,
                        confidence_threshold=int(threshold_value),
                    )
                    sync_sentiment_state(unique2, grouped2)
                    st.session_state.spotcheck_auto_resolve_message = (
                        f"Applied the auto-resolve threshold to existing reviewed matches. "
                        f"{accepted_count:,} grouped storie(s) were finalized and {reopened_count:,} were reopened."
                    )
                    st.rerun()

            action1, action2 = st.columns(2)
    
            with action1:
                if st.button("Run AI second opinion", type="secondary"):
                    st.session_state.pop("__last_spot_check_ai_summary__", None)
                    progress_text = st.empty()
                    progress_bar = st.progress(0.0)
                    progress_text.caption(f"Running second opinions: 0/{selected_batch_size}")

                    with st.spinner("Running AI second opinions on top candidates..."):
                        unique2, grouped2, batch_errors, total_in, total_out, batch_auto_resolved = run_batch_second_opinion(
                            candidates_df=base_candidates,
                            df_unique=st.session_state.df_sentiment_unique,
                            df_grouped=st.session_state.df_sentiment_grouped_rows,
                            pre_prompt=pre_prompt,
                            sentiment_instruction=sentiment_instruction,
                            post_prompt=post_prompt,
                            functions=functions,
                            sentiment_type=sentiment_type,
                            api_key=st.secrets["key"],
                            review_model=DEFAULT_SECOND_OPINION_MODEL,
                            limit=int(selected_batch_size),
                            max_workers=8,
                            low_conf_threshold=int(st.session_state.get("spotcheck_low_conf_threshold", DEFAULT_REVIEW_CONFIDENCE_THRESHOLD)),
                            entity_terms=sentiment_entity_terms,
                            progress_callback=lambda completed, total: (
                                progress_bar.progress(completed / max(1, total)),
                                progress_text.caption(f"Running second opinions: {completed}/{total}"),
                            ),
                        )
                    progress_bar.progress(1.0)
                    progress_text.caption(f"Completed second opinions: {min(selected_batch_size, len(base_candidates))}/{min(selected_batch_size, len(base_candidates))}")
    
                    sync_sentiment_state(unique2, grouped2)
    
                    apply_usage_to_session(total_in, total_out, DEFAULT_SECOND_OPINION_MODEL)
    
                    batch_cost = estimate_cost_usd(total_in, total_out, DEFAULT_SECOND_OPINION_MODEL)
                    session_cost = get_api_cost_usd()
    
                    st.session_state["__last_spot_check_ai_summary__"] = {
                        "in_tok": total_in,
                        "out_tok": total_out,
                        "batch_cost": batch_cost,
                        "session_cost": session_cost,
                        "elapsed": 0.0,
                        "errors": batch_errors,
                        "processed": min(selected_batch_size, len(base_candidates)),
                        "auto_resolved": batch_auto_resolved,
                    }
                    completed_at = format_local_timestamp()
                    st.session_state.spotcheck_pre_review_message = (
                        f"AI second-opinion batch completed {completed_at}. Review counts updated below."
                    )
                    st.rerun()
    
            with action2:
                st.caption("High-confidence matching opinions can auto-resolve here before you move on to human spot checks.")
    
        if controls_in_expander:
            with st.expander("Advanced review options", expanded=False):
                render_pre_review_controls()
        else:
            render_pre_review_controls()
    
    if spot_checks_mode == "pre_review":
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
        last_summary = st.session_state.get("__last_spot_check_ai_summary__")
        if last_summary and (last_summary.get("processed") or last_summary.get("auto_resolved")):
            st.caption(
                f"Last second-opinion batch: {int(last_summary.get('processed', 0)):,} processed, {int(last_summary.get('auto_resolved', 0)):,} auto-resolved."
            )
        if sensitive_eligible_count > 0:
            st.caption(f"{sensitive_eligible_count:,} eligible story group(s) currently carry negative / sensitive first-pass sentiment and are being prioritized in the second-opinion queue.")
        st.stop()
    
    low_conf_threshold = int(
        st.session_state.get("spotcheck_low_conf_threshold", DEFAULT_REVIEW_CONFIDENCE_THRESHOLD)
    )
    
    all_coverage_candidates = build_all_sentiment_candidates(st.session_state.df_sentiment_unique)
    sentiment_bucket_options = [
        label
        for label in (
            [
                "POSITIVE",
                "NEUTRAL",
                "NEGATIVE",
                "NOT RELEVANT",
            ]
            if sentiment_type == "3-way"
            else [
                "VERY POSITIVE",
                "SOMEWHAT POSITIVE",
                "NEUTRAL",
                "SOMEWHAT NEGATIVE",
                "VERY NEGATIVE",
                "NOT RELEVANT",
            ]
        )
        if not all_coverage_candidates.empty
        and "__effective_sentiment__" in all_coverage_candidates.columns
        and all_coverage_candidates["__effective_sentiment__"].eq(label).any()
    ]

    view_col1, view_col2 = st.columns([1.2, 1], gap="medium")
    with view_col1:
        review_mode = st.selectbox(
            "Spot check view",
            [
                "Flagged for human review",
                "Disagreements only",
                "All unresolved stories",
                "All Toned Coverage",
                "All Coverage",
                "Specific Sentiment Bucket",
            ],
            key="spotcheck_review_mode",
        )
    selected_bucket = None
    with view_col2:
        if review_mode == "Specific Sentiment Bucket":
            selected_bucket = st.selectbox(
                "Sentiment bucket",
                sentiment_bucket_options or ["No labeled buckets available"],
                key="spotcheck_selected_bucket",
                disabled=not sentiment_bucket_options,
            )

    if review_mode == "All Toned Coverage":
        review_source_df = all_toned_candidates
    elif review_mode == "All Coverage":
        review_source_df = all_coverage_candidates
    elif review_mode == "Specific Sentiment Bucket":
        review_source_df = all_coverage_candidates
    else:
        review_source_df = review_base_candidates

    def _build_candidates_for_current_view(unique_df: pd.DataFrame, grouped_df: pd.DataFrame) -> pd.DataFrame:
        base_review_df = compute_candidates(
            df_unique=unique_df,
            df_grouped=grouped_df,
            sentiment_type=sentiment_type,
            conf_thresh=DEFAULT_CONF_THRESH,
        )
        toned_df = build_all_toned_candidates(unique_df)
        coverage_df = build_all_sentiment_candidates(unique_df)
        if review_mode == "All Toned Coverage":
            source_df = toned_df
        elif review_mode == "All Coverage":
            source_df = coverage_df
        elif review_mode == "Specific Sentiment Bucket":
            source_df = coverage_df
        else:
            source_df = base_review_df
        filtered = filter_candidates_for_review_mode(
            source_df,
            review_mode=review_mode,
            low_conf_threshold=low_conf_threshold,
        )
        if review_mode == "Specific Sentiment Bucket":
            if selected_bucket and selected_bucket != "No labeled buckets available":
                filtered = filtered[
                    filtered.get("__effective_sentiment__", pd.Series(index=filtered.index, dtype="object"))
                    .fillna("")
                    .astype(str)
                    .str.strip()
                    .str.upper()
                    .eq(str(selected_bucket).strip().upper())
                ].copy()
            else:
                filtered = pd.DataFrame()
        return filtered

    if spot_checks_mode == "spot_checks":
        pending_candidates = _build_candidates_for_current_view(
            st.session_state.df_sentiment_unique,
            st.session_state.df_sentiment_grouped_rows,
        )
        summary1, summary2, summary3, summary4 = st.columns(4)
        with summary1:
            st.metric("Ready for review", f"{len(review_base_candidates):,}")
        with summary2:
            st.metric("Needs human review", f"{needs_review_count:,}")
        with summary3:
            st.metric("Disagreements", f"{disagreement_count:,}")
        with summary4:
            st.metric("In review queue", f"{len(pending_candidates):,}")
    
    candidates = _build_candidates_for_current_view(
        st.session_state.df_sentiment_unique,
        st.session_state.df_sentiment_grouped_rows,
    )
    st.caption(
        "Flagged, Disagreements, and All unresolved keep the queue focused on review priority. "
        "All Coverage and bucket views broaden the browse set, but still sort unresolved and higher-visibility stories toward the top."
    )
    
    if candidates.empty:
        st.info("No stories match the current view.")
        small1, small2 = st.columns(2)
        with small1:
            st.metric("Spot-checked", checked)
        with small2:
            st.metric("Acceptance rate", f"{acceptance_rate:.0%}")
        st.stop()
    
    # ---------------------------
    # Current story selection
    # ---------------------------
    if st.session_state.spot_lock_gid is not None:
        locked_gid = st.session_state.spot_lock_gid
        locked_rows = candidates[candidates["Group ID"] == locked_gid]
        if not locked_rows.empty:
            row = locked_rows.iloc[0]
            idx = locked_rows.index[0]
            st.session_state.spot_idx = int(candidates.index.get_loc(idx))
        else:
            st.session_state.spot_lock_gid = None
            st.session_state.spot_idx = min(st.session_state.spot_idx, len(candidates) - 1)
            idx = st.session_state.spot_idx
            row = candidates.iloc[idx]
    else:
        st.session_state.spot_idx = min(st.session_state.spot_idx, len(candidates) - 1)
        idx = st.session_state.spot_idx
        row = candidates.iloc[idx]
    
    current_group_id = int(row["Group ID"])
    url = str(row.get("Example URL", row.get("URL", "")) or "")
    head_raw = row.get("Headline", "") or ""
    body_raw = row.get("Example Snippet", row.get("Snippet", "")) or ""
    
    trans_head = row.get("Translated Headline")
    trans_body = row.get("Translated Body")
    head_to_show = trans_head if (isinstance(trans_head, str) and trans_head.strip()) else head_raw
    body_to_show = trans_body if (isinstance(trans_body, str) and trans_body.strip()) else body_raw
    
    head_display = escape_markdown(head_to_show)
    body_display = escape_markdown(body_to_show)
    highlighted_head = highlight_with_tolerant_regex(head_display, tolerant_pat_str, keywords)
    highlighted_body = highlight_with_tolerant_regex(body_display, tolerant_pat_str, keywords)
    
    with st.sidebar:
        if st.button("Translate"):
            try:
                th = translate_text(head_raw) if str(head_raw).strip() else None
                tb = translate_text(body_raw) if str(body_raw).strip() else None

                st.session_state.spot_lock_gid = current_group_id

                unique, grouped = apply_translation_to_group(
                    st.session_state.df_sentiment_unique,
                    st.session_state.df_sentiment_grouped_rows,
                    current_group_id,
                    th,
                    tb,
                )
                sync_sentiment_state(unique, grouped)
                st.rerun()
            except Exception as e:
                st.error(f"Translation failed: {e}")
    
    left, right = st.columns([3.8, 1.4], gap="large")
    
    with left:
        st.markdown(f"#### {highlighted_head}", unsafe_allow_html=True)
        st.markdown(highlighted_body, unsafe_allow_html=True)
        st.divider()
        if url:
            st.markdown(url)
    
    with right:
        story_prompt = build_story_prompt(
            headline=head_raw,
            snippet=body_raw,
            pre_prompt=pre_prompt,
            sentiment_instruction=sentiment_instruction,
            post_prompt=post_prompt,
        )
    
        ai_label = row.get("AI Sentiment")
        if ai_label and current_group_id not in st.session_state.initial_ai_label:
            st.session_state.initial_ai_label[current_group_id] = str(ai_label).strip()
    
        if (not ai_label) and (not st.session_state.spot_ai_loading):
            st.session_state.spot_ai_loading = True
            st.session_state.spot_ai_refresh_requested = True
            st.session_state.spot_lock_gid = current_group_id
            st.session_state.spot_ai_model_override = model_id
            st.rerun()
    
        if st.session_state.spot_ai_loading and st.session_state.spot_ai_refresh_requested:
            with st.spinner("Generating AI opinion..."):
                ai_result, in_tok, out_tok, note = call_ai_sentiment(
                    story_prompt=story_prompt,
                    model_to_use=st.session_state.spot_ai_model_override or model_id,
                    functions=functions,
                    sentiment_type=sentiment_type,
                    api_key=st.secrets["key"],
                )

            if ai_result:
                from processing.ai_sentiment import enforce_not_relevant_direct_mention_rule

                ai_result = enforce_not_relevant_direct_mention_rule(
                    ai_result,
                    headline=head_raw,
                    snippet=body_raw,
                    entity_terms=sentiment_entity_terms,
                )
                label = ai_result.get("sentiment")
                conf = ai_result.get("confidence")
                why = ai_result.get("explanation")
    
                unique, grouped = write_second_opinion_to_group(
                    st.session_state.df_sentiment_unique,
                    st.session_state.df_sentiment_grouped_rows,
                    current_group_id,
                    label,
                    conf,
                    why,
                )
                sync_sentiment_state(unique, grouped)
    
                if current_group_id not in st.session_state.initial_ai_label and label:
                    st.session_state.initial_ai_label[current_group_id] = str(label).strip()
    
                apply_usage_to_session(
                    in_tok,
                    out_tok,
                    st.session_state.spot_ai_model_override or model_id,
                )
    
                batch_cost = estimate_cost_usd(
                    in_tok,
                    out_tok,
                    st.session_state.spot_ai_model_override or model_id,
                )
                session_cost = get_api_cost_usd()
    
                st.session_state["__last_spot_check_ai_summary__"] = {
                    "in_tok": total_in if "total_in" in locals() else in_tok,
                    "out_tok": total_out if "total_out" in locals() else out_tok,
                    "batch_cost": batch_cost,
                    "session_cost": session_cost,
                    "elapsed": 0.0,
                    "errors": [note] if note else [],
                }
    
            st.session_state.spot_ai_loading = False
            st.session_state.spot_ai_refresh_requested = False
            st.session_state.spot_ai_model_override = None
            st.session_state.spot_lock_gid = None
            st.rerun()
    
        row_fresh = st.session_state.df_sentiment_unique.loc[
            st.session_state.df_sentiment_unique["Group ID"] == current_group_id
        ].iloc[0]
    
        ai_label = row_fresh.get("AI Sentiment")
        ai_conf = row_fresh.get("AI Sentiment Confidence")
        ai_rsn = row_fresh.get("AI Sentiment Rationale")
    
        review_label = row_fresh.get("Review AI Sentiment")
        review_conf = row_fresh.get("Review AI Confidence")
        review_rsn = row_fresh.get("Review AI Rationale")
        agreement = row_fresh.get("AI Agreement")
        needs_review = row_fresh.get("Needs Human Review")
        agreement = "" if pd.isna(agreement) else str(agreement)
        needs_review = "" if pd.isna(needs_review) else str(needs_review)
    
    
        def _fmt_label(label, conf):
            # Safely handle pd.NA / None / empty
            if pd.isna(label) or str(label).strip() == "":
                return "Not available"
    
            conf_num = pd.to_numeric(pd.Series([conf]), errors="coerce").iloc[0]
            if pd.notna(conf_num):
                return f"{str(label).strip()} ({int(conf_num)})"
    
            return str(label).strip()
    
        if st.session_state.spot_ai_loading:
            st.info("AI is working…")
        else:
            current_assignment_source = row_fresh.get("Assigned Sentiment Source")
            current_assignment_source = "" if pd.isna(current_assignment_source) else str(current_assignment_source).strip()
            if agreement == "Disagree":
                st.warning("AI opinions disagree.")
            elif current_assignment_source == "AI_AUTO_RESOLVED":
                st.success("Auto-resolved by AI (high confidence match).")
            elif needs_review == "Yes":
                st.info("Flagged for review.")
    
            st.write(f"**1st**: {_fmt_label(ai_label, ai_conf)}")
    
            # if review_label:
            #     st.write(f"**2nd**: {_fmt_label(review_label, review_conf)}")
            if not pd.isna(review_label) and str(review_label).strip():
                st.write(f"**2nd**: {_fmt_label(review_label, review_conf)}")
    
            # st.write("**Choose final sentiment**")
    
        def _rebuild_and_advance(final_label: str):
            unique2, grouped2 = set_assigned_sentiment(
                st.session_state.df_sentiment_unique,
                st.session_state.df_sentiment_grouped_rows,
                current_group_id,
                final_label,
            )
            sync_sentiment_state(unique2, grouped2)
    
            update_acceptance_tracking(
                st.session_state,
                current_group_id,
                final_label,
            )
    
            new_filtered = _build_candidates_for_current_view(
                st.session_state.df_sentiment_unique,
                st.session_state.df_sentiment_grouped_rows,
            )
    
            if new_filtered.empty:
                st.rerun()
            else:
                st.session_state.spot_idx = min(idx, len(new_filtered) - 1)
                st.rerun()
    
        if sentiment_type == "5-way":
            manual_labels = [
                "VERY POSITIVE",
                "SOMEWHAT POSITIVE",
                "NEUTRAL",
                "SOMEWHAT NEGATIVE",
                "VERY NEGATIVE",
                "NOT RELEVANT",
            ]
        else:
            manual_labels = ["POSITIVE", "NEUTRAL", "NEGATIVE", "NOT RELEVANT"]
    
        clicked_override = None
        for lbl in manual_labels:
            if colored_button(lbl):
                clicked_override = lbl
                break
    
        if clicked_override:
            _rebuild_and_advance(clicked_override)

        if not pd.isna(review_label) and str(review_label).strip():
            if st.button("Re-run review AI", use_container_width=True):
                st.session_state.pop("__last_spot_check_ai_summary__", None)
                st.session_state.spot_ai_loading = True
                st.session_state.spot_ai_refresh_requested = True
                st.session_state.spot_ai_model_override = DEFAULT_SECOND_OPINION_MODEL
                st.session_state.spot_lock_gid = current_group_id
                st.rerun()
    
    
        has_ai_rsn = not pd.isna(ai_rsn) and str(ai_rsn).strip()
        has_review_rsn = not pd.isna(review_rsn) and str(review_rsn).strip()
    
        if has_ai_rsn or has_review_rsn:
            with st.expander("AI opinion reasoning", expanded=False):
                if has_ai_rsn:
                    st.write("**1st AI rationale**")
                    st.write(str(ai_rsn))
                if has_review_rsn:
                    st.write("**2nd AI rationale**")
                    st.write(str(review_rsn))
    
    
        st.divider()
    
        nav1, nav2 = st.columns(2)
        with nav1:
            if st.button("", disabled=(idx <= 0), use_container_width=True, icon=":material/skip_previous:", help="Previous story"):
                st.session_state.spot_idx = max(0, idx - 1)
                st.session_state.spot_lock_gid = None
                st.rerun()
        with nav2:
            if st.button("", disabled=(idx >= len(candidates) - 1), use_container_width=True, icon=":material/skip_next:", help="Next story"):
                st.session_state.spot_idx = min(len(candidates) - 1, idx + 1)
                st.session_state.spot_lock_gid = None
                st.rerun()
    
        st.caption(f"Story {idx + 1} of {len(candidates)} in current view")
    
    if spot_checks_mode != "spot_checks":
        st.caption(
            "Stories where both AI opinions match at high review confidence are auto-assigned. The remaining stories need your judgment."
        )
    
    sentiment_dist = build_sentiment_distribution(st.session_state.df_sentiment_unique, sentiment_type)
    
    if spot_checks_mode not in {"spot_checks", "distribution"}:
        with st.expander("Current sentiment distribution", expanded=False):
            sentiment_table = sentiment_dist.copy()
            sentiment_table["Share"] = (sentiment_table["Share"] * 100).map(lambda x: f"{x:.1f}%")
            st.dataframe(sentiment_table, hide_index=True, use_container_width=True)
