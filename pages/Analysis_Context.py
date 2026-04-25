from __future__ import annotations

from datetime import datetime
import importlib

import streamlit as st
from streamlit_tags import st_tags

import processing.analysis_context as analysis_context
from ui.page_help import set_page_help_context

analysis_context = importlib.reload(analysis_context)
set_page_help_context(st.session_state, "Analysis Context")


st.title("Analysis Context")
st.caption("Optionally define the topic, aliases, spokespeople, products, and analytical guidance that should shape AI outputs across the app.")

st.markdown(
    """
    <style>
    div.st-key-analysis_context_draft_exclude_aggregators [data-testid="stCheckbox"] {
        padding-top: 8px;
        padding-bottom: 8px;
    }
    .entity-field-note {
        margin-top: -0.02rem;
        margin-bottom: 1rem;
        color: rgba(250, 250, 250, 0.58);
        font-size: 0.82rem;
        line-height: 1.38;
    }
    .entity-card-gap {
        height: 0.55rem;
    }
    .entity-subfield-gap {
        height: 0.4rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if not st.session_state.get("standard_step", False):
    st.error("Please complete Basic Cleaning before trying this step.")
    st.stop()

analysis_context.init_analysis_context_state(st.session_state)
if "analysis_context_save_message" not in st.session_state:
    st.session_state.analysis_context_save_message = None
if "analysis_context_suggestion_success" not in st.session_state:
    st.session_state.analysis_context_suggestion_success = False
if "analysis_context_tag_widget_version" not in st.session_state:
    st.session_state.analysis_context_tag_widget_version = 0


def _has_meaningful_downstream_work() -> bool:
    list_keys = [
        "author_insights_selected_authors",
        "outlet_insights_selected_outlets",
        "top_stories_validation_confirmed_keys",
    ]
    for key in list_keys:
        if st.session_state.get(key):
            return True

    dict_keys = [
        "author_insights_summaries",
        "outlet_insights_summaries",
        "regions_generated_output",
        "tagging_observation_output",
        "sentiment_observation_output",
    ]
    for key in dict_keys:
        value = st.session_state.get(key)
        if isinstance(value, dict) and len(value) > 0:
            return True

    df_keys = [
        "added_df",
        "df_tagging_unique",
        "df_sentiment_unique",
    ]
    for key in df_keys:
        value = st.session_state.get(key)
        if hasattr(value, "empty") and not value.empty:
            return True

    return False

payload = analysis_context.get_analysis_context_payload(st.session_state)
if "analysis_context_draft_initialized" not in st.session_state:
    st.session_state.analysis_context_draft_initialized = True
    st.session_state.analysis_context_draft_client_name = payload["client_name"]
    st.session_state.analysis_context_draft_primary_name = payload["primary_name"]
    st.session_state.analysis_context_draft_alternate_names = list(payload["alternate_names"])
    st.session_state.analysis_context_draft_spokespeople = list(payload["spokespeople"])
    st.session_state.analysis_context_draft_products = list(payload["products"])
    st.session_state.analysis_context_draft_highlight_keywords = list(payload.get("highlight_keywords", []))
    st.session_state.analysis_context_draft_guidance = payload["guidance"]
    st.session_state.analysis_context_draft_exclude_aggregators = payload["exclude_aggregators_from_outlet_insights"]
    st.session_state.analysis_context_draft_media_type_commentary_mode = payload.get("media_type_commentary_mode", "Auto")
    st.session_state.analysis_context_draft_qualitative_flags = list(payload["qualitative_excluded_flags"])
    st.session_state.analysis_context_draft_dataset_flags = list(payload["dataset_excluded_flags"])
    st.session_state.analysis_context_draft_dataset_date_range = (
        payload.get("dataset_start_date"),
        payload.get("dataset_end_date"),
    )
    st.session_state.analysis_context_draft_dataset_media_types = list(payload.get("dataset_media_types", []))
    st.session_state.analysis_context_draft_qualitative_keep_keys = list(payload.get("qualitative_exclusion_keep_keys", []))
    st.session_state.analysis_context_draft_dataset_keep_keys = list(payload.get("dataset_exclusion_keep_keys", []))

st.session_state.setdefault(
    "analysis_context_draft_highlight_keywords",
    list(payload.get("highlight_keywords", [])),
)
st.session_state.setdefault(
    "analysis_context_draft_media_type_commentary_mode",
    payload.get("media_type_commentary_mode", "Auto"),
)

if _has_meaningful_downstream_work():
        st.warning(
        "You have already done work in downstream workflows. Changes saved here may affect ranked outputs and generated insights in Authors, Outlets, Top Stories, Regions, Sentiment, and Tagging. Revisit and regenerate affected outputs if needed."
        )

pending_suggestions = st.session_state.pop("analysis_context_pending_suggestions", None)
if pending_suggestions:
    client_key = analysis_context._match_key(st.session_state.analysis_context_draft_client_name)
    primary_key = analysis_context._match_key(st.session_state.analysis_context_draft_primary_name)
    alias_names = [
        item["name"]
        for item in analysis_context._clean_suggestion_items(pending_suggestions.get("aliases", []))
        if analysis_context._match_key(item["name"]) not in {client_key, primary_key}
    ]
    spokesperson_names = [
        item["name"]
        for item in analysis_context._clean_suggestion_items(pending_suggestions.get("spokespeople", []))
        if " " in item["name"].strip()
    ]
    product_names = [
        item["name"]
        for item in analysis_context._clean_suggestion_items(pending_suggestions.get("products", []))
        if analysis_context._match_key(item["name"]) not in {client_key, primary_key}
    ]
    st.session_state.analysis_context_draft_alternate_names = analysis_context._clean_list(
        st.session_state.analysis_context_draft_alternate_names + alias_names
    )
    st.session_state.analysis_context_draft_spokespeople = analysis_context._clean_list(
        st.session_state.analysis_context_draft_spokespeople + spokesperson_names
    )
    st.session_state.analysis_context_draft_products = analysis_context._clean_list(
        st.session_state.analysis_context_draft_products + product_names
    )
    st.session_state.analysis_context_suggestion_payload = pending_suggestions
    st.session_state.analysis_context_tag_widget_version += 1
    st.session_state.analysis_context_suggestion_success = True

with st.container(border=True):
    st.subheader("Entity Context")
    st.caption("Define the subject of the analysis, then add reference names and optional prompt-shaping guidance for downstream AI workflows.")

    with st.container(border=True):
        st.markdown("**Core identity**")
        st.caption("Start with the main entity or topic the app should orient around.")
        col1, col2 = st.columns(2, gap="medium")
        with col1:
            client_name = st.text_input(
                "Client name",
                key="analysis_context_draft_client_name",
                help="Usually carried over from Getting Started, but you can adjust it here if needed.",
            )
        with col2:
            primary_name = st.text_input(
                "Primary topic or entity of interest",
                key="analysis_context_draft_primary_name",
                help="This can be the client itself, or a broader topic you want the AI to focus on instead.",
            )

        helper_col1, helper_col2, _ = st.columns([1, 0.95, 1.05], gap="small")
        with helper_col1:
            if st.button("Suggest context items with AI", type="primary", key="analysis_context_ai_suggest", use_container_width=True):
                try:
                    with st.spinner("Generating context suggestions..."):
                        suggestions, _, _ = analysis_context.generate_analysis_context_suggestions(
                            client_name=client_name,
                            primary_name=primary_name,
                            alternate_names=st.session_state.analysis_context_draft_alternate_names,
                            spokespeople=st.session_state.analysis_context_draft_spokespeople,
                            products=st.session_state.analysis_context_draft_products,
                            guidance=st.session_state.analysis_context_draft_guidance,
                            api_key=st.secrets["key"],
                            model=analysis_context.DEFAULT_ANALYSIS_CONTEXT_MODEL,
                        )
                    st.session_state.analysis_context_pending_suggestions = suggestions
                    st.rerun()
                except Exception as e:
                    st.error(f"Could not generate context suggestions: {e}")
        with helper_col2:
            if st.button("Clear entity context below", key="analysis_context_clear_below", use_container_width=True):
                st.session_state.analysis_context_draft_alternate_names = []
                st.session_state.analysis_context_draft_spokespeople = []
                st.session_state.analysis_context_draft_products = []
                st.session_state.analysis_context_draft_highlight_keywords = []
                st.session_state.analysis_context_draft_guidance = ""
                st.session_state.analysis_context_suggestion_payload = None
                st.session_state.analysis_context_tag_widget_version += 1
                st.rerun()

        if st.session_state.get("analysis_context_suggestion_success"):
            st.success("AI context suggestions added to the fields below.")
            st.session_state.analysis_context_suggestion_success = False

    st.markdown('<div class="entity-card-gap"></div>', unsafe_allow_html=True)

    tag_key_suffix = st.session_state.analysis_context_tag_widget_version
    with st.container(border=True):
        st.markdown("**Reference names**")
        st.caption("Add the names and related entities that should help the app recognize relevant coverage.")
        ref_col1, ref_col2 = st.columns(2, gap="medium")
        with ref_col1:
            alternate_names = st_tags(
                label="Alternate names / aliases",
                text="Press enter to add more",
                maxtags=20,
                value=st.session_state.analysis_context_draft_alternate_names,
                key=f"analysis_context_aliases_tags_{tag_key_suffix}",
            )
            st.markdown(
                '<div class="entity-field-note">Include alternate spellings, abbreviations, or commonly used shorthand.</div>',
                unsafe_allow_html=True,
            )
            st.markdown('<div class="entity-subfield-gap"></div>', unsafe_allow_html=True)
            products = st_tags(
                label="Products / sub-brands / initiatives",
                text="Press enter to add more",
                maxtags=20,
                value=st.session_state.analysis_context_draft_products,
                key=f"analysis_context_products_tags_{tag_key_suffix}",
            )
            st.markdown(
                '<div class="entity-field-note">Use this for product names, program names, campaigns, or sub-brands that meaningfully define the story.</div>',
                unsafe_allow_html=True,
            )
        with ref_col2:
            spokespeople = st_tags(
                label="Key spokespeople",
                text="Press enter to add more",
                maxtags=20,
                value=st.session_state.analysis_context_draft_spokespeople,
                key=f"analysis_context_spokespeople_tags_{tag_key_suffix}",
            )
            st.markdown(
                '<div class="entity-field-note">Add people central to the coverage.</div>',
                unsafe_allow_html=True,
            )
            st.markdown('<div class="entity-subfield-gap"></div>', unsafe_allow_html=True)
            highlight_keywords = st_tags(
                label="Other keywords to highlight in Sentiment / Tagging",
                text="Press enter to add more",
                maxtags=30,
                value=st.session_state.analysis_context_draft_highlight_keywords,
                key=f"analysis_context_highlight_keywords_tags_{tag_key_suffix}",
            )
            st.markdown(
                '<div class="entity-field-note">Highlight-only terms for spot checks. These are not added to AI prompt context.</div>',
                unsafe_allow_html=True,
            )

    with st.container(border=True):
        st.markdown("**Prompt shaping**")
        st.caption("Add optional analytical framing that should shape how downstream AI outputs interpret the coverage.")
        guidance = st.text_area(
            "Additional rationale, context, or guidance (optional)",
            key="analysis_context_draft_guidance",
            height=110,
            help="Use this for analytical framing, nuances, or focus that should shape AI-generated summaries and observations.",
        )

    st.session_state.analysis_context_draft_alternate_names = alternate_names
    st.session_state.analysis_context_draft_spokespeople = spokespeople
    st.session_state.analysis_context_draft_products = products
    st.session_state.analysis_context_draft_highlight_keywords = highlight_keywords

    with st.expander("AI suggestion rationale", expanded=False):
        suggestion_payload = st.session_state.get("analysis_context_suggestion_payload")
        if suggestion_payload:
            section_map = [
                ("Alternate names / aliases", suggestion_payload.get("aliases", [])),
                ("Key spokespeople", suggestion_payload.get("spokespeople", [])),
                ("Products / sub-brands / initiatives", suggestion_payload.get("products", [])),
            ]
            for heading, items in section_map:
                if not items:
                    continue
                st.write(f"**{heading}**")
                for item in items:
                    name = str(item.get("name", "") or "").strip()
                    detail = str(item.get("detail", "") or "").strip()
                    if not name:
                        continue
                    st.markdown(f"- `{name}`" + (f": {detail}" if detail else ""))
        else:
            st.info("No AI suggestions have been generated yet.")

with st.container(border=True):
    st.subheader("Analysis Focus")
    st.caption("Set shared rules for what should be excluded from qualitative workflows like Top Stories, Sentiment, Tagging, Authors, Outlets, and Regions.")

    focus_col1, focus_col2 = st.columns(2, gap="medium")
    with focus_col1:
        with st.container(border=True):
            st.markdown("**Aggregator handling**")
            st.caption("Control whether aggregator coverage should count in Outlet metrics and narrative outputs.")
            exclude_aggregators_from_outlet_insights = st.checkbox(
                "Exclude aggregators from Outlet metrics / insights",
                key="analysis_context_draft_exclude_aggregators",
                help="Recommended. Keeps aggregator coverage out of Outlet charts and narrative while leaving the rest of the dataset alone.",
            )

    with focus_col2:
        with st.container(border=True):
            st.markdown("**Media type emphasis**")
            st.caption("Control how much downstream AI outputs should talk about media-type mix when interpreting coverage patterns.")
            media_type_commentary_mode = st.radio(
                "Media type commentary",
                options=analysis_context.MEDIA_TYPE_COMMENTARY_OPTIONS,
                key="analysis_context_draft_media_type_commentary_mode",
                horizontal=True,
                label_visibility="collapsed",
                help="Controls how much downstream qualitative outputs should talk about media-type mix when interpreting coverage patterns.",
            )

    with st.container(border=True):
        st.markdown("**Qualitative exclusions**")
        st.caption("Exclude low-value or non-editorial coverage from qualitative workflows while keeping the main cleaned dataset intact.")
        qualitative_excluded_flags = st.multiselect(
            "Exclude junky coverage flags from qualitative insights",
            options=payload["available_junky_flags"],
            default=st.session_state.analysis_context_draft_qualitative_flags,
            help="Recommended defaults are Press Release and Advertorial. Add others when the dataset warrants it.",
            key="analysis_context_draft_qualitative_flags_widget",
        )
    st.session_state.analysis_context_draft_qualitative_flags = qualitative_excluded_flags

    qualitative_exclusion_keep_keys = list(st.session_state.analysis_context_draft_qualitative_keep_keys)
    if qualitative_excluded_flags:
        qualitative_preview = analysis_context.build_coverage_flag_removal_preview(
            st.session_state.get("df_traditional"),
            qualitative_excluded_flags,
            keep_row_keys=set(st.session_state.analysis_context_draft_qualitative_keep_keys),
        )
        qual_stat1, qual_stat2 = st.columns(2, gap="medium")
        with qual_stat1:
            st.metric("Rows excluded from qualitative insights", f"{qualitative_preview['removed_rows']:,}")
        with qual_stat2:
            st.metric("Mentions excluded from qualitative insights", f"{qualitative_preview['removed_mentions']:,}")

        if not qualitative_preview["counts_df"].empty:
            st.write("**Qualitative exclusion preview by flag**")
            st.dataframe(
                qualitative_preview["counts_df"][["Coverage Flag", "Mentions", "Impressions", "Effective Reach"]],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Mentions": st.column_config.NumberColumn("Mentions", format="%,d"),
                    "Impressions": st.column_config.NumberColumn("Impressions", format="%,d"),
                    "Effective Reach": st.column_config.NumberColumn("Effective Reach", format="%,d"),
                },
            )

        if not qualitative_preview["sample_df"].empty:
            with st.expander("Review excluded rows", expanded=False):
                st.caption("Uncheck `Exclude` to keep a specific row in qualitative workflows even when it matches the selected junky flags.")
                qualitative_editor = st.data_editor(
                    qualitative_preview["sample_df"][["Remove", "Headline", "Outlet", "Coverage Flags", "Link", "Row Key"]],
                    use_container_width=True,
                    hide_index=True,
                    disabled=["Headline", "Outlet", "Coverage Flags", "Link", "Row Key"],
                    column_order=["Remove", "Headline", "Outlet", "Coverage Flags", "Link"],
                    column_config={
                        "Remove": st.column_config.CheckboxColumn("Exclude", width="small"),
                        "Headline": st.column_config.Column("Headline", width="large"),
                        "Outlet": st.column_config.Column("Outlet", width="medium"),
                        "Coverage Flags": st.column_config.Column("Coverage Flags", width="medium"),
                        "Link": st.column_config.LinkColumn("Link", width="small", display_text="open"),
                    },
                    key="analysis_context_qualitative_removal_editor",
                )
                qualitative_exclusion_keep_keys = qualitative_editor.loc[~qualitative_editor["Remove"], "Row Key"].astype(str).tolist()
                st.session_state.analysis_context_draft_qualitative_keep_keys = qualitative_exclusion_keep_keys
                if qualitative_exclusion_keep_keys:
                    st.caption(f"Keeping {len(qualitative_exclusion_keep_keys)} row(s) in qualitative workflows even though they match the selected junky flags.")
        else:
            st.info("No rows with those selected junky flags are present in the current cleaned dataset.")
    else:
        qualitative_exclusion_keep_keys = []
        st.session_state.analysis_context_draft_qualitative_keep_keys = []
        st.info("No junky coverage flags are currently selected for qualitative exclusion.")

with st.container(border=True):
    st.subheader("Data Scope")
    st.caption("Refine the working dataset used by downstream workflows. Anything scoped out here is removed from the app's working view unless you explicitly keep a row below.")

    scope_col1, scope_col2 = st.columns([1.05, 1.25], gap="medium")
    with scope_col1:
        if payload.get("dataset_min_date") and payload.get("dataset_max_date"):
            dataset_date_range = st.date_input(
                "Date range",
                value=st.session_state.analysis_context_draft_dataset_date_range,
                min_value=payload.get("dataset_min_date"),
                max_value=payload.get("dataset_max_date"),
                key="analysis_context_draft_dataset_date_range_widget",
                help="Limit downstream workflows to a specific date range from the cleaned dataset.",
            )
            if isinstance(dataset_date_range, tuple) and len(dataset_date_range) == 2:
                st.session_state.analysis_context_draft_dataset_date_range = dataset_date_range
        else:
            st.info("No usable dates detected in the current cleaned dataset.")

    with scope_col2:
        dataset_media_types = st.multiselect(
            "Traditional media types",
            options=payload["available_media_types"],
            default=st.session_state.analysis_context_draft_dataset_media_types,
            key="analysis_context_draft_dataset_media_types_widget",
            help="Only these media types will remain in the working dataset for downstream workflows.",
        )
        st.session_state.analysis_context_draft_dataset_media_types = dataset_media_types

    dataset_excluded_flags = st.multiselect(
        "Prune flagged coverage from working dataset",
        options=payload["available_junky_flags"],
        default=st.session_state.analysis_context_draft_dataset_flags,
        help="Optional. Remove rows with these flags from the working dataset, with row-level keep overrides below.",
        key="analysis_context_draft_dataset_flags_widget",
    )
    st.session_state.analysis_context_draft_dataset_flags = dataset_excluded_flags

    dataset_exclusion_keep_keys = list(st.session_state.analysis_context_draft_dataset_keep_keys)
    dataset_date_range = st.session_state.analysis_context_draft_dataset_date_range
    dataset_start_date = dataset_date_range[0] if isinstance(dataset_date_range, tuple) and len(dataset_date_range) == 2 else None
    dataset_end_date = dataset_date_range[1] if isinstance(dataset_date_range, tuple) and len(dataset_date_range) == 2 else None

    preview = analysis_context.build_dataset_scope_preview(
        st.session_state.get("df_traditional"),
        start_date=dataset_start_date,
        end_date=dataset_end_date,
        selected_media_types=dataset_media_types,
        excluded_flags=dataset_excluded_flags,
        keep_row_keys=set(st.session_state.analysis_context_draft_dataset_keep_keys),
    )
    if preview["removed_rows"] > 0:
        preview_stat1, preview_stat2 = st.columns(2, gap="medium")
        with preview_stat1:
            st.metric("Rows removed from working dataset", f"{preview['removed_rows']:,}")
        with preview_stat2:
            st.metric("Mentions removed from working dataset", f"{preview['removed_mentions']:,}")

        if not preview["counts_df"].empty:
            st.write("**Removal preview by reason**")
            st.dataframe(
                preview["counts_df"][["Reason", "Rows", "Mentions", "Impressions", "Effective Reach"]],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Rows": st.column_config.NumberColumn("Rows", format="%,d"),
                    "Mentions": st.column_config.NumberColumn("Mentions", format="%,d"),
                    "Impressions": st.column_config.NumberColumn("Impressions", format="%,d"),
                    "Effective Reach": st.column_config.NumberColumn("Effective Reach", format="%,d"),
                },
            )

        if not preview["sample_df"].empty:
            st.write("**Rows removed from working dataset**")
            st.caption("Uncheck `Remove` to keep a specific row in the working dataset even when it matches the selected scope or pruning rules.")
            removal_editor = st.data_editor(
                preview["sample_df"][["Remove", "Headline", "Outlet", "Type", "Coverage Flags", "Removal Reason", "Link", "Row Key"]],
                use_container_width=True,
                hide_index=True,
                disabled=["Headline", "Outlet", "Type", "Coverage Flags", "Removal Reason", "Link", "Row Key"],
                column_order=["Remove", "Headline", "Outlet", "Type", "Coverage Flags", "Removal Reason", "Link"],
                column_config={
                    "Remove": st.column_config.CheckboxColumn("Remove", width="small"),
                    "Headline": st.column_config.Column("Headline", width="large"),
                    "Outlet": st.column_config.Column("Outlet", width="medium"),
                    "Type": st.column_config.Column("Type", width="small"),
                    "Coverage Flags": st.column_config.Column("Coverage Flags", width="medium"),
                    "Removal Reason": st.column_config.Column("Removal Reason", width="medium"),
                    "Link": st.column_config.LinkColumn("Link", width="small", display_text="open"),
                },
                key="analysis_context_dataset_removal_editor",
            )
            dataset_exclusion_keep_keys = removal_editor.loc[~removal_editor["Remove"], "Row Key"].astype(str).tolist()
            st.session_state.analysis_context_draft_dataset_keep_keys = dataset_exclusion_keep_keys
            if dataset_exclusion_keep_keys:
                st.caption(f"Keeping {len(dataset_exclusion_keep_keys)} row(s) even though they match the selected data-scope rules.")
    else:
        st.info("No rows are currently being removed by the selected data-scope rules.")

save_col, save_status_col = st.columns([0.32, 0.68], gap="small")
with save_col:
    if st.button("Save Analysis Context", type="primary", use_container_width=True):
        analysis_context.save_analysis_context(
            st.session_state,
            client_name=client_name,
            primary_name=primary_name,
            alternate_names=alternate_names,
            spokespeople=spokespeople,
            products=products,
            highlight_keywords=highlight_keywords,
            guidance=guidance,
            qualitative_excluded_flags=qualitative_excluded_flags,
            dataset_excluded_flags=dataset_excluded_flags,
            exclude_aggregators_from_outlet_insights=exclude_aggregators_from_outlet_insights,
            media_type_commentary_mode=media_type_commentary_mode,
            dataset_start_date=dataset_start_date,
            dataset_end_date=dataset_end_date,
            dataset_media_types=dataset_media_types,
            qualitative_exclusion_keep_keys=qualitative_exclusion_keep_keys,
            dataset_exclusion_keep_keys=dataset_exclusion_keep_keys,
        )
        saved_at = datetime.now().astimezone().strftime("%b %-d, %Y at %-I:%M %p")
        st.session_state.analysis_context_save_message = f"Saved {saved_at}"

with save_status_col:
    save_message = st.session_state.get("analysis_context_save_message")
    if save_message:
        st.success(save_message)
        st.session_state.analysis_context_save_message = None

with st.expander("Saved shared context preview", expanded=False):
    preview = analysis_context.build_analysis_context_text(st.session_state)
    if preview:
        st.code(preview, language="text")
    else:
        st.info("No analysis context has been saved yet.")
