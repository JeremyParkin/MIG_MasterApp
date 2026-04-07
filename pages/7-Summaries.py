from __future__ import annotations

import warnings

import streamlit as st
from streamlit_tags import st_tags

from processing.top_story_summaries import (
    init_top_story_summary_state,
    seed_entity_names,
    normalize_summary_df,
    build_entity_context,
    build_prompt_preview,
    generate_outputs_for_dataframe,
    build_markdown_output,
    DEFAULT_MODEL,
)

from utils.api_meter import apply_usage_to_session


warnings.filterwarnings("ignore")

st.title("Top Stories Summaries")

if len(st.session_state.get("added_df", [])) == 0:
    st.error("Please select your TOP STORIES before trying this step.")
    st.stop()

init_top_story_summary_state(st.session_state)

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

summary_col1, summary_col2, summary_col3 = st.columns(3, gap="small")

with summary_col1:
    client_name = str(st.session_state.get("client_name", "")).strip()
    seed_entity_names(st.session_state, client_name)

    entity_names = st_tags(
        label="Entity name and aliases",
        text="Primary then aliases",
        maxtags=20,
        value=st.session_state.top_story_entity_names,
        key="top_story_entity_names_tags",
    )
    st.session_state.top_story_entity_names = entity_names

    primary_name = entity_names[0].strip() if entity_names else ""
    alternate_names = [name.strip() for name in entity_names[1:] if name.strip()]

with summary_col2:
    spokespeople = st_tags(
        label="Key spokespeople",
        maxtags=20,
        value=st.session_state.top_story_spokespeople,
        key="top_story_spokespeople_tags",
    )
    st.session_state.top_story_spokespeople = spokespeople

with summary_col3:
    products = st_tags(
        label="Products / sub-brands / initiatives",
        text="Press enter to add more",
        maxtags=20,
        value=st.session_state.top_story_products,
        key="top_story_products_tags",
    )
    st.session_state.top_story_products = products

additional_guidance = st.text_area(
    "**Additional guidance (optional)**",
    value=st.session_state.top_story_guidance,
    height=50,
    help="Optional extra instructions for how the model should interpret or prioritize the entity in coverage.",
    key="top_story_guidance_text",
)
st.session_state.top_story_guidance = additional_guidance

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

    st.session_state.added_df = updated_df.copy()
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
st.subheader("Copy-Paste Top Stories")
st.markdown(":mag: **VIEW OPTIONS**")

show_col1, show_col2, show_col3 = st.columns(3, gap="medium")

with show_col1:
    show_mentions = st.checkbox("Show mentions", value=False)
    show_impressions = st.checkbox("Show impressions", value=False)

with show_col2:
    show_top_story_summary = "Top Story Summary" in df.columns and st.checkbox("Show top story summary", value=True)
    show_callout = "Chart Callout" in df.columns and st.checkbox("Show chart callout", value=True)

with show_col3:
    show_sentiment = "Entity Sentiment" in df.columns and st.checkbox("Show sentiment", value=True)

st.divider()

markdown_content = build_markdown_output(
    df=df,
    show_top_story_summary=show_top_story_summary,
    show_callout=show_callout,
    show_sentiment=show_sentiment,
    show_mentions=show_mentions,
    show_impressions=show_impressions,
)

st.markdown(markdown_content, unsafe_allow_html=True)