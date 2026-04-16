from __future__ import annotations


def render_top_stories_validation() -> None:
    import importlib
    import warnings

    import pandas as pd
    import streamlit as st

    import processing.top_stories as top_stories_module

    warnings.filterwarnings("ignore")
    top_stories_module = importlib.reload(top_stories_module)

    build_source_candidate_table = top_stories_module.build_source_candidate_table
    normalize_top_stories_df = top_stories_module.normalize_top_stories_df
    parse_source_group_ids = top_stories_module.parse_source_group_ids
    rotate_saved_story_source = top_stories_module.rotate_saved_story_source

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
    source_df = st.session_state.df_ai_grouped.copy()

    if saved_df.empty:
        st.info("No saved top stories available for validation.")
        st.stop()

    for idx, row in saved_df.reset_index(drop=True).iterrows():
        story_group_id = row.get("Group ID")
        source_ids = parse_source_group_ids(row.get("Source Group IDs", ""), fallback_group_id=story_group_id)
        source_candidates = build_source_candidate_table(
            df=source_df,
            source_group_ids=row.get("Source Group IDs", ""),
            fallback_group_id=story_group_id,
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

            snippet = str(row.get("Example Snippet", "") or "").strip()
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
            st.caption(f"Source {current_rank} of {source_count}")

            action1, action2, action3 = st.columns([1, 1, 3], gap="small")
            with action1:
                if current_url:
                    st.link_button("Open current link", current_url, use_container_width=True)
                else:
                    st.button("Open current link", key=f"top_story_open_link_disabled_{idx}", disabled=True, use_container_width=True)
            with action2:
                if st.button("Try next source", key=f"top_story_next_source_{idx}", disabled=source_count <= 1):
                    st.session_state.added_df = rotate_saved_story_source(
                        saved_df=st.session_state.added_df.copy(),
                        source_df=source_df,
                        story_group_id=story_group_id,
                        step=1,
                    )
                    st.session_state.top_story_observation_output = None
                    st.rerun()
            with action3:
                if source_ids:
                    st.caption(f"{len(source_ids)} source instance(s) available in this story family.")

    st.divider()
    if st.button("Continue to Insights", type="primary", key="top_stories_continue_to_insights"):
        st.session_state.top_stories_section = "Insights"
        st.rerun()
