# 10-Spot_Checks.py

from __future__ import annotations

import warnings

import pandas as pd
import streamlit as st
from streamlit_extras.stylable_container import stylable_container

from processing.spot_checks import (
    DEFAULT_CONF_THRESH,
    DEFAULT_SECOND_OPINION_MODEL,
    DEFAULT_REVIEW_CONFIDENCE_THRESHOLD,
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

warnings.filterwarnings("ignore")

st.markdown("<style>.block-container{padding-top:2.75rem !important;}</style>", unsafe_allow_html=True)

if not st.session_state.get("sentiment_config_step", False):
    st.error("Please complete AI Sentiment setup before trying this step.")
    st.stop()

if not isinstance(st.session_state.get("df_sentiment_unique"), pd.DataFrame):
    st.error("Sentiment unique stories not found. Please complete earlier steps.")
    st.stop()

init_spot_check_state(st.session_state)
init_api_meter()


def colored_button(label: str, color: str) -> bool:
    css = f"""
    button {{
        width: 100%;
        background-color: {color} !important;
        color: black !important;
        border: 0;
        padding: 0.2rem 0.6rem;
        font-weight: bold !important;
        font-size: 14px;
        border-radius: 5px;
        margin-bottom: 10px;
    }}
    """
    stable_key = f"spot_manual_{label.replace(' ', '_')}"
    with stylable_container(key=f"wrap_{stable_key}", css_styles=css):
        return st.button(label, key=stable_key, use_container_width=True)


def sync_sentiment_state(unique_df: pd.DataFrame, grouped_df: pd.DataFrame) -> None:
    st.session_state.df_sentiment_unique = unique_df
    st.session_state.df_sentiment_grouped_rows = grouped_df
    st.session_state.df_sentiment_rows = grouped_df.copy()


def filter_candidates_for_review_mode(
    candidates_df: pd.DataFrame,
    review_mode: str,
    low_conf_threshold: int,
) -> pd.DataFrame:
    if candidates_df.empty:
        return candidates_df

    out = candidates_df.copy()

    if review_mode == "Flagged only":
        if "Needs Human Review" in out.columns:
            flagged = out[out["Needs Human Review"] == "Yes"].copy()
            if not flagged.empty:
                return flagged
        return out

    return out


def auto_accept_high_confidence_matches(
    unique_df: pd.DataFrame,
    grouped_df: pd.DataFrame,
    confidence_threshold: int,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
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
    ]
    for col in required_cols:
        if col not in unique.columns:
            return unique, grouped, 0

    ai_label = unique["AI Sentiment"].fillna("").astype(str).str.strip().str.upper()
    review_label = unique["Review AI Sentiment"].fillna("").astype(str).str.strip().str.upper()
    assigned = unique["Assigned Sentiment"].fillna("").astype(str).str.strip()

    ai_conf = pd.to_numeric(unique["AI Sentiment Confidence"], errors="coerce")
    review_conf = pd.to_numeric(unique["Review AI Confidence"], errors="coerce")

    mask = (
            (assigned == "")
            & (unique["AI Agreement"].fillna("") == "Match")
            & (ai_label != "")
            & (ai_label == review_label)
            & review_conf.ge(confidence_threshold)
    )


    accepted_count = int(mask.sum())
    if accepted_count == 0:
        return unique, grouped, 0

    gids_to_accept = unique.loc[mask, "Group ID"].tolist()

    for gid in gids_to_accept:
        label = unique.loc[unique["Group ID"] == gid, "AI Sentiment"].iloc[0]
        unique, grouped = set_assigned_sentiment(unique, grouped, gid, label)
        update_acceptance_tracking(st.session_state, gid, str(label).strip())

    return unique, grouped, accepted_count


df_unique = st.session_state.df_sentiment_unique.copy()
df_grouped = st.session_state.df_sentiment_grouped_rows.copy()
df_unique, df_grouped = ensure_review_columns(df_unique, df_grouped)
sync_sentiment_state(df_unique, df_grouped)

if df_unique.empty:
    st.error("No sentiment-grouped stories found.")
    st.stop()

pre_prompt = st.session_state.get("pre_prompt", "")
post_prompt = st.session_state.get("post_prompt", "")
sentiment_instruction = st.session_state.get("sentiment_instruction", "")
functions = st.session_state.get("functions", [])
model_id = st.session_state.get("model_choice", "gpt-5.4-nano")

_raw_st = st.session_state.get("sentiment_type", "3-way")
_s = str(_raw_st).strip().lower()
sentiment_type = "5-way" if _s.startswith("5") or "5-way" in _s else "3-way"

keywords = st.session_state.get("highlight_keyword", [])
if not isinstance(keywords, list):
    keywords = [str(keywords)] if keywords else []
keywords = [k for k in keywords if isinstance(k, str) and k.strip()]
tolerant_pat_str = st.session_state.get("highlight_regex_str")

base_candidates = compute_candidates(
    df_unique=st.session_state.df_sentiment_unique,
    df_grouped=st.session_state.df_sentiment_grouped_rows,
    sentiment_type=sentiment_type,
    conf_thresh=DEFAULT_CONF_THRESH,
)

checked = len(st.session_state.spot_checked_groups)
accepted = len(st.session_state.accepted_initial)
acceptance_rate = (accepted / checked) if checked else 0.0

reviewed_candidates = base_candidates[
    base_candidates["Review AI Sentiment"].notna()
].copy() if "Review AI Sentiment" in base_candidates.columns else base_candidates.iloc[0:0].copy()

disagreement_count = int(
    (reviewed_candidates.get("AI Agreement", pd.Series(dtype="object")) == "Disagree").sum()
) if not reviewed_candidates.empty else 0

needs_review_count = int(
    (reviewed_candidates.get("Needs Human Review", pd.Series(dtype="object")) == "Yes").sum()
) if not reviewed_candidates.empty else 0


last_batch = st.session_state.get("__last_spot_check_ai_summary__")
if last_batch:
    st.success("AI pre-review complete. Focus on the flagged stories below.")

if base_candidates.empty:
    st.success("All set — no remaining stories need spot checks.")
    small1, small2 = st.columns(2)
    with small1:
        st.metric("Spot-checked", checked)
    with small2:
        st.metric("Acceptance rate", f"{acceptance_rate:.0%}")
    st.stop()

# ---------------------------
# Simplified top section
# ---------------------------
top_message = ""
if reviewed_candidates.empty:
    top_message = f"{len(base_candidates):,} stories are ready for review."
else:
    top_message = f"{needs_review_count:,} stories need your attention."
    if disagreement_count > 0:
        top_message += f" {disagreement_count:,} disagreement{'s' if disagreement_count != 1 else ''} found."

st.markdown(f"### {top_message}")

with st.expander("Advanced review options", expanded=False):
    adv1, adv2 = st.columns(2)
    with adv1:
        auto_review_n = st.number_input(
            "Top candidates to send for AI pre-review",
            min_value=1,
            max_value=min(len(base_candidates), 200),
            value=min(50, len(base_candidates)),
            step=1,
            key="spotcheck_auto_review_n",
        )
    with adv2:
        low_conf_threshold = st.number_input(
            "Review confidence cutoff",
            min_value=1,
            max_value=100,
            value=int(st.session_state.get("spotcheck_low_conf_threshold", DEFAULT_REVIEW_CONFIDENCE_THRESHOLD)),
            step=1,
            key="spotcheck_low_conf_threshold",
        )

    action1, action2 = st.columns(2)

    with action1:
        if st.button("Run AI pre-review", type="secondary"):
            st.session_state.pop("__last_spot_check_ai_summary__", None)

            with st.spinner("Running AI second opinions on top candidates..."):
                unique2, grouped2, batch_errors, total_in, total_out = run_batch_second_opinion(
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
                    limit=int(auto_review_n),
                    max_workers=8,
                    low_conf_threshold=int(low_conf_threshold),
                )

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
            }
            st.rerun()

    with action2:
        st.caption("High-confidence matching opinions are auto-assigned during AI pre-review.")

review_mode = st.radio(
    "View",
    ["Flagged only", "All remaining"],
    horizontal=True,
    label_visibility="collapsed",
)

low_conf_threshold = int(
    st.session_state.get("spotcheck_low_conf_threshold", DEFAULT_REVIEW_CONFIDENCE_THRESHOLD)
)
candidates = filter_candidates_for_review_mode(
    base_candidates,
    review_mode=review_mode,
    low_conf_threshold=low_conf_threshold,
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
    st.markdown(f"### {highlighted_head}", unsafe_allow_html=True)
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
        if agreement == "Disagree":
            st.warning("AI opinions disagree.")
        elif agreement == "Match" and needs_review != "Yes":
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

        new_base = compute_candidates(
            st.session_state.df_sentiment_unique,
            st.session_state.df_sentiment_grouped_rows,
            sentiment_type,
            DEFAULT_CONF_THRESH,
        )
        new_filtered = filter_candidates_for_review_mode(
            new_base,
            review_mode=review_mode,
            low_conf_threshold=low_conf_threshold,
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
        palette = {
            "VERY POSITIVE": "#10ad82",
            "SOMEWHAT POSITIVE": "#72cc4a",
            "NEUTRAL": "#f1c40f",
            "SOMEWHAT NEGATIVE": "#e67e22",
            "VERY NEGATIVE": "#c0392b",
            "NOT RELEVANT": "#7f8c8d",
        }
    else:
        manual_labels = ["POSITIVE", "NEUTRAL", "NEGATIVE", "NOT RELEVANT"]
        palette = {
            "POSITIVE": "#2ecc71",
            "NEUTRAL": "#f1c40f",
            "NEGATIVE": "#e74c3c",
            "NOT RELEVANT": "#7f8c8d",
        }

    clicked_override = None
    for lbl in manual_labels:
        if colored_button(lbl, palette[lbl]):
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
        if st.button("◄ Prev", disabled=(idx <= 0), use_container_width=True):
            st.session_state.spot_idx = max(0, idx - 1)
            st.session_state.spot_lock_gid = None
            st.rerun()
    with nav2:
        if st.button("Next ►", disabled=(idx >= len(candidates) - 1), use_container_width=True):
            st.session_state.spot_idx = min(len(candidates) - 1, idx + 1)
            st.session_state.spot_lock_gid = None
            st.rerun()

    st.caption(f"Story {idx + 1} of {len(candidates)} in current view")

st.caption(
"Stories where both AI opinions match at high review confidence are auto-assigned. The remaining stories need your judgment."
)
