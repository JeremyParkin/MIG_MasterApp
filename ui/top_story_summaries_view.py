from __future__ import annotations

def render_top_story_summaries() -> None:
    # 7-Summaries.py
    
    import warnings
    import importlib
    
    import streamlit as st
    
    from processing.analysis_context import (
        build_analysis_context_caption,
        get_analysis_context_payload,
        init_analysis_context_state,
    )
    
    from processing.top_story_summaries import (
        init_top_story_summary_state,
        normalize_summary_df,
        build_entity_context,
        build_prompt_preview,
        generate_outputs_for_dataframe,
        build_markdown_output,
        generate_top_story_observation,
        DEFAULT_MODEL,
    )
    import processing.top_stories as top_stories_module
    
    from utils.api_meter import apply_usage_to_session
    
    
    warnings.filterwarnings("ignore")
    top_stories_module = importlib.reload(top_stories_module)

    st.subheader("Step 3: Top Story Insights")
    st.caption("Generate the saved top-story outputs and review the overall observations and report copy.")
    
    if len(st.session_state.get("added_df", [])) == 0:
        st.error("Please select your TOP STORIES before trying this step.")
        st.stop()
    
    init_top_story_summary_state(st.session_state)
    init_analysis_context_state(st.session_state)
    
    batch = st.session_state.pop("__summary_batch_result__", None)
    
    if batch:
        st.success(f"Completed AI outputs for {batch['done']} storie(s).")
    
        if batch["errors"]:
            with st.expander(f"Completed with {len(batch['errors'])} error(s)"):
                for err in batch["errors"]:
                    st.write(err)
    
        st.caption(
            f"Token usage this batch: input={batch['in_tok']:,} • output={batch['out_tok']:,}"
        )
    
    df = normalize_summary_df(st.session_state.added_df.copy())
    df = df.sort_values(by="Date", ascending=True).reset_index(drop=True)
    
    st.subheader("Generate Analysis")
    
    analysis_payload = get_analysis_context_payload(st.session_state)
    primary_name = analysis_payload["primary_name"]
    alternate_names = analysis_payload["alternate_names"]
    spokespeople = analysis_payload["spokespeople"]
    products = analysis_payload["products"]
    additional_guidance = analysis_payload["guidance"]

    st.write("**Shared analysis context**")
    analysis_caption = build_analysis_context_caption(st.session_state)
    if analysis_caption:
        st.caption(analysis_caption)
    else:
        st.caption("No shared analysis context saved yet. Add it on the Analysis Context page.")
    
    generate_col1, generate_col2 = st.columns([1.5, 3], gap="medium")
    with generate_col1:
        submitted = st.button("Generate All Outputs", type="primary")
    with generate_col2:
        st.caption("Generates chart callout, top story summary, and entity sentiment together for each saved top story.")
    
    if submitted and not primary_name.strip():
        st.error("Primary entity is required to proceed.")
    
    if submitted and primary_name.strip():
        entity_context = build_entity_context(
            primary_name=primary_name,
            alternate_names=alternate_names,
            spokespeople=spokespeople,
            products=products,
            additional_guidance=additional_guidance,
        )
    
        progress_bar = st.progress(0)
        status = st.empty()
    
        # Wrapped progress version
        total_items = len(df)
        processed = 0
    
    
        def progress_generate():
            nonlocal_processed = {"count": 0}
            working_df = normalize_summary_df(df.copy())
            errors = []
            total_in = 0
            total_out = 0
    
            from concurrent.futures import ThreadPoolExecutor, as_completed
            from processing.top_story_summaries import generate_outputs_for_row, DEFAULT_MAX_WORKERS
    
            rows_for_workers = [(i, row.to_dict()) for i, row in working_df.iterrows()]
    
            with ThreadPoolExecutor(max_workers=DEFAULT_MAX_WORKERS) as executor:
                future_map = {
                    executor.submit(
                        generate_outputs_for_row,
                        row_tuple,
                        entity_context,
                        st.secrets["key"],
                    ): row_tuple[0]
                    for row_tuple in rows_for_workers
                }
    
                total = len(future_map)
    
                for future in as_completed(future_map):
                    i = future_map[future]
                    nonlocal_processed["count"] += 1
    
                    try:
                        row_index, outputs, error_message, in_tok, out_tok = future.result()
                        total_in += int(in_tok or 0)
                        total_out += int(out_tok or 0)
    
                        if error_message:
                            errors.append(f"Story {row_index + 1}: {error_message}")
                        else:
                            for col, value in outputs.items():
                                working_df.at[row_index, col] = value
    
                    except Exception as e:
                        errors.append(f"Story {i + 1}: {e}")
    
                    progress_bar.progress(int((nonlocal_processed["count"] / total) * 100))
                    status.caption(f"Processed {nonlocal_processed['count']:,} of {total:,} stories")
    
            return working_df, errors, total_in, total_out
    
    
    
        updated_df, errors, total_in, total_out = progress_generate()
        observation_output = None
        try:
            observation_output, obs_in, obs_out = generate_top_story_observation(
                df=updated_df,
                entity_name=primary_name,
                api_key=st.secrets["key"],
            )
            total_in += int(obs_in or 0)
            total_out += int(obs_out or 0)
        except Exception as e:
            observation_output = {"_error": str(e)}

        st.session_state.added_df = updated_df.copy()
        st.session_state.top_story_observation_output = observation_output
        df = updated_df.copy()
    
        apply_usage_to_session(total_in, total_out, DEFAULT_MODEL)
    
        st.session_state["__summary_batch_result__"] = {
            "done": len(updated_df),
            "errors": errors,
            "in_tok": total_in,
            "out_tok": total_out,
        }
    
        st.rerun()
    
    
    
    entity_context = ""
    if primary_name.strip():
        entity_context = build_entity_context(
            primary_name=primary_name,
            alternate_names=alternate_names,
            spokespeople=spokespeople,
            products=products,
            additional_guidance=additional_guidance,
        )
    
    with st.expander("Show AI prompt preview", expanded=False):
        st.caption("This shows the exact prompt template sent to OpenAI (with example story placeholders).")
        st.code(build_prompt_preview(entity_context), language="text")
    
    st.divider()
    st.subheader("Top Story Observations")
    st.caption("Uses the saved top stories plus generated summaries, callouts, and sentiment context to produce a concise overall observation.")

    observation_output = st.session_state.get("top_story_observation_output")
    if observation_output and observation_output.get("_error"):
        st.error(f"Could not generate top story observations: {observation_output['_error']}")
    elif observation_output:
        overall = str(observation_output.get("overall_observation", "") or "").strip()
        if overall:
            st.markdown("### Overall Observations")
            st.write(overall)
    else:
        st.info("Generate all outputs to build the overall observations block automatically.")

    st.divider()
    st.subheader("Report Copy")
    field_options = []
    if "Mentions" in df.columns:
        field_options.append("Mentions")
    if "Impressions" in df.columns:
        field_options.append("Impressions")
    if "Effective Reach" in df.columns:
        field_options.append("Effective reach")
    if "Top Story Summary" in df.columns:
        field_options.append("Summary")
    if "Chart Callout" in df.columns:
        field_options.append("Callout")
    if "Entity Sentiment" in df.columns:
        field_options.append("Sentiment")

    if "top_story_report_fields" not in st.session_state:
        st.session_state.top_story_report_fields = field_options.copy()

    preset_col, fields_col = st.columns([0.18, 0.82], gap="small")
    with preset_col:
        bulk_col1, bulk_col2 = st.columns(2, gap="small")
        with bulk_col1:
            if st.button("All", key="top_story_report_select_all", use_container_width=True):
                st.session_state.top_story_report_fields = field_options.copy()
                st.rerun()
        with bulk_col2:
            if st.button("None", key="top_story_report_select_none", use_container_width=True):
                st.session_state.top_story_report_fields = []
                st.rerun()

    with fields_col:
        st.pills(
            "Fields",
            options=field_options,
            selection_mode="multi",
            default=st.session_state.get("top_story_report_fields", field_options),
            key="top_story_report_fields",
            label_visibility="collapsed",
        )

    selected_fields = set(st.session_state.get("top_story_report_fields", []) or [])
    show_mentions = "Mentions" in selected_fields
    show_impressions = "Impressions" in selected_fields
    show_effective_reach = "Effective reach" in selected_fields
    show_top_story_summary = "Summary" in selected_fields
    show_callout = "Callout" in selected_fields
    show_sentiment = "Sentiment" in selected_fields
    
    st.divider()
    
    markdown_content = build_markdown_output(
        df=df,
        show_top_story_summary=show_top_story_summary,
        show_callout=show_callout,
        show_sentiment=show_sentiment,
        show_mentions=show_mentions,
        show_impressions=show_impressions,
        show_effective_reach=show_effective_reach,
    )
    
    st.markdown(markdown_content, unsafe_allow_html=True)
