from __future__ import annotations


def render_top_stories_validation() -> None:
    import importlib
    import warnings

    import pandas as pd
    import streamlit as st

    import processing.top_stories as top_stories_module
    from processing.standard_cleaning import SOCIAL_TYPES

    warnings.filterwarnings("ignore")
    top_stories_module = importlib.reload(top_stories_module)

    build_prime_grouped_story_candidates = top_stories_module.build_prime_grouped_story_candidates
    build_story_identity_key = top_stories_module.build_story_identity_key
    build_source_candidate_table_from_candidates = top_stories_module.build_source_candidate_table_from_candidates
    normalize_top_stories_df = top_stories_module.normalize_top_stories_df
    parse_source_group_ids = top_stories_module.parse_source_group_ids
    rotate_saved_story_source_from_candidates = top_stories_module.rotate_saved_story_source_from_candidates
    strip_html_tags = top_stories_module.strip_html_tags

    st.markdown(
        """
        <style>
        div[data-testid="stButton"] button,
        div[data-testid="stLinkButton"] a {
            min-height: 2.8rem;
            font-weight: 600;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Step 2: Top Story Validation")
    st.caption("Review saved story links and rotate to the next-best source when the current example URL is weak or unavailable.")

    if len(st.session_state.get("added_df", [])) == 0:
        st.error("Please save your TOP STORIES before trying this step.")
        st.stop()

    if "df_ai_grouped" not in st.session_state or st.session_state.df_ai_grouped.empty:
        st.error("Grouped story data is missing. Please complete Basic Cleaning again.")
        st.stop()

    saved_df = normalize_top_stories_df(st.session_state.added_df.copy())
    grouped_source_df = st.session_state.df_ai_grouped
    source_df = grouped_source_df.copy()
    if "Type" in source_df.columns:
        source_df = source_df[~source_df["Type"].fillna("").astype(str).str.upper().isin(SOCIAL_TYPES)].copy()

    source_signature = (
        len(grouped_source_df),
        len(source_df),
        tuple(source_df.columns.tolist()),
        int(pd.to_numeric(source_df.get("Mentions", pd.Series(dtype="float64")), errors="coerce").fillna(0).sum()),
        int(pd.to_numeric(source_df.get("Impressions", pd.Series(dtype="float64")), errors="coerce").fillna(0).sum()),
        int(pd.to_numeric(source_df.get("Effective Reach", pd.Series(dtype="float64")), errors="coerce").fillna(0).sum()),
        int(pd.to_numeric(source_df.get("Prime Example", pd.Series(dtype="float64")), errors="coerce").fillna(0).sum()),
    )
    cached_signature = st.session_state.get("top_stories_validation_prime_candidates_signature")
    if cached_signature == source_signature and isinstance(
        st.session_state.get("top_stories_validation_prime_candidates"),
        pd.DataFrame,
    ):
        prime_source_candidates = st.session_state.top_stories_validation_prime_candidates
    else:
        prime_source_candidates = build_prime_grouped_story_candidates(source_df)
        st.session_state.top_stories_validation_prime_candidates = prime_source_candidates
        st.session_state.top_stories_validation_prime_candidates_signature = source_signature

    if saved_df.empty:
        st.info("No saved top stories available for validation.")
        st.stop()

    saved_df = saved_df.reset_index(drop=True)
    source_group_ids = (
        saved_df["Source Group IDs"]
        if "Source Group IDs" in saved_df.columns
        else pd.Series(index=saved_df.index, data="")
    )
    saved_df["_story_identity_key"] = [
        build_story_identity_key(source_group_ids.iloc[idx], saved_df.iloc[idx].get("Group ID"))
        for idx in range(len(saved_df))
    ]
    saved_keys = saved_df["_story_identity_key"].fillna("").astype(str).str.strip().tolist()
    saved_signature = tuple(saved_keys)
    prior_signature = st.session_state.get("top_stories_validation_saved_signature")
    validated_keys = {
        str(key).strip()
        for key in st.session_state.get("top_stories_validation_confirmed_keys", [])
        if str(key).strip()
    }
    if prior_signature != saved_signature:
        validated_keys = validated_keys.intersection(set(saved_keys))
        st.session_state.top_stories_validation_confirmed_keys = sorted(validated_keys)
        st.session_state.top_stories_validation_saved_signature = saved_signature
        st.session_state.top_stories_validation_index = 0

    queue_df = saved_df[
        ~saved_df["_story_identity_key"].fillna("").astype(str).str.strip().isin(validated_keys)
    ].copy().reset_index(drop=True)
    if queue_df.empty:
        st.success("All saved top stories have confirmed sources.")
        if saved_keys:
            st.caption(f"Confirmed {len(validated_keys)} of {len(saved_keys)} saved stories.")
        st.stop()

    st.session_state.setdefault("top_stories_validation_index", 0)
    current_index = int(st.session_state.get("top_stories_validation_index", 0) or 0)
    current_index = max(0, min(current_index, len(queue_df) - 1))
    st.session_state.top_stories_validation_index = current_index

    nav_left, nav_right = st.columns([3.8, 2.2], gap="medium")
    with nav_right:
        nav1, nav2, nav3, nav4 = st.columns(4, gap="small")
        with nav1:
            if st.button("", key="top_story_validation_first", use_container_width=True, disabled=current_index <= 0, icon=":material/first_page:", help="First story"):
                st.session_state.top_stories_validation_index = 0
                st.rerun()
        with nav2:
            if st.button("", key="top_story_validation_prev", use_container_width=True, disabled=current_index <= 0, icon=":material/skip_previous:", help="Previous story"):
                st.session_state.top_stories_validation_index = current_index - 1
                st.rerun()
        with nav3:
            if st.button("", key="top_story_validation_next", use_container_width=True, disabled=current_index >= len(queue_df) - 1, icon=":material/skip_next:", help="Next story"):
                st.session_state.top_stories_validation_index = current_index + 1
                st.rerun()
        with nav4:
            if st.button("", key="top_story_validation_last", use_container_width=True, disabled=current_index >= len(queue_df) - 1, icon=":material/last_page:", help="Last story"):
                st.session_state.top_stories_validation_index = len(queue_df) - 1
                st.rerun()
        st.caption(f"Reviewing story {current_index + 1} of {len(queue_df)}")

    row = queue_df.iloc[current_index]
    story_group_id = row.get("Group ID")
    story_identity_key = str(row.get("_story_identity_key", "") or "").strip()
    source_ids = parse_source_group_ids(row.get("Source Group IDs", ""), fallback_group_id=story_group_id)
    source_candidates = build_source_candidate_table_from_candidates(
        candidates=prime_source_candidates,
        source_group_ids=row.get("Source Group IDs", ""),
        fallback_group_id=story_group_id,
        require_url_if_available=bool(str(row.get("Example URL", "") or "").strip()),
    )
    source_count = max(len(source_candidates), 1)
    current_url = str(row.get("Example URL", "") or "").strip()
    current_outlet = str(row.get("Example Outlet", "") or "").strip()
    current_type = str(row.get("Example Type", "") or "").strip()

    current_rank = 1
    if not source_candidates.empty:
        for pos, (_, candidate) in enumerate(source_candidates.iterrows(), start=1):
            if (
                str(candidate.get("Example URL", "") or "").strip() == current_url
                and str(candidate.get("Example Outlet", "") or "").strip() == current_outlet
                and str(candidate.get("Example Type", "") or "").strip() == current_type
            ):
                current_rank = pos
                break

    container = st.container(border=True)
    with container:
        st.markdown(f"### {row.get('Headline', '')}")

        snippet = strip_html_tags(row.get("Example Snippet", ""))
        if snippet:
            st.caption(snippet[:420] + ("..." if len(snippet) > 420 else ""))

        meta_parts = []
        if pd.notna(row.get("Date")):
            meta_parts.append(str(row.get("Date")))
        if current_outlet:
            meta_parts.append(current_outlet)
        if current_type:
            meta_parts.append(current_type)
        mentions = int(pd.to_numeric(pd.Series([row.get("Mentions", 0)]), errors="coerce").fillna(0).iloc[0])
        impressions = int(pd.to_numeric(pd.Series([row.get("Impressions", 0)]), errors="coerce").fillna(0).iloc[0])
        meta_parts.append(f"Mentions: {mentions:,}")
        meta_parts.append(f"Impressions: {impressions:,}")
        st.caption(" | ".join(meta_parts))
        st.caption(f"Source option {current_rank} of {source_count}")

        action1, action2, action3, action4 = st.columns([1, 1, 1, 2], gap="small")
        with action1:
            if current_url:
                st.link_button("Open current link", current_url, use_container_width=True)
            else:
                st.button("Open current link", key=f"top_story_open_link_disabled_{current_index}", disabled=True, use_container_width=True)
        with action2:
            if st.button("Try next source", key=f"top_story_next_source_{current_index}", disabled=source_count <= 1):
                st.session_state.added_df = rotate_saved_story_source_from_candidates(
                    saved_df=st.session_state.added_df.copy(),
                    candidates=prime_source_candidates,
                    story_group_id=story_group_id,
                    step=1,
                )
                st.session_state.top_story_observation_output = None
                st.rerun()
        with action3:
            if st.button("Confirm source", key=f"top_story_confirm_source_{current_index}", use_container_width=True):
                confirmed = {
                    str(key).strip()
                    for key in st.session_state.get("top_stories_validation_confirmed_keys", [])
                    if str(key).strip()
                }
                if story_identity_key:
                    confirmed.add(story_identity_key)
                st.session_state.top_stories_validation_confirmed_keys = sorted(confirmed)
                st.session_state.top_stories_validation_index = min(current_index, max(len(queue_df) - 2, 0))
                st.rerun()
        with action4:
            if source_ids:
                st.caption(f"{len(source_ids)} source instance(s) available in this story family.")
