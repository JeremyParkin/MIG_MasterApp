from __future__ import annotations

import importlib

import streamlit as st
from streamlit_tags import st_tags

import processing.analysis_context as analysis_context

analysis_context = importlib.reload(analysis_context)


st.title("Analysis Context")
st.caption("Optionally define the topic, aliases, spokespeople, products, and analytical guidance that should shape AI outputs across the app.")

if not st.session_state.get("standard_step", False):
    st.error("Please complete Basic Cleaning before trying this step.")
    st.stop()

analysis_context.init_analysis_context_state(st.session_state)
if "analysis_context_save_success" not in st.session_state:
    st.session_state.analysis_context_save_success = False
if "analysis_context_suggestion_success" not in st.session_state:
    st.session_state.analysis_context_suggestion_success = False
if "analysis_context_tag_widget_version" not in st.session_state:
    st.session_state.analysis_context_tag_widget_version = 0
pending_suggestions = st.session_state.pop("analysis_context_pending_suggestions", None)
if pending_suggestions:
    analysis_context.apply_analysis_context_suggestions(st.session_state, pending_suggestions)
    st.session_state.analysis_context_tag_widget_version += 1
    st.session_state.analysis_context_suggestion_success = True

payload = analysis_context.get_analysis_context_payload(st.session_state)

st.subheader("Entity Context")
col1, col2 = st.columns(2, gap="medium")
with col1:
    client_name = st.text_input(
        "Client name",
        value=payload["client_name"],
        help="Usually carried over from Getting Started, but you can adjust it here if needed.",
    )
with col2:
    primary_name = st.text_input(
        "Primary topic or entity of interest",
        value=payload["primary_name"],
        help="This can be the client itself, or a broader topic you want the AI to focus on instead.",
    )

helper_col1, helper_col2, helper_col3 = st.columns([1, 0.55, 1.85], gap="small")
with helper_col1:
    if st.button("Suggest context items with AI", type="primary", key="analysis_context_ai_suggest", use_container_width=True):
        try:
            with st.spinner("Generating context suggestions..."):
                suggestions, _, _ = analysis_context.generate_analysis_context_suggestions(
                    client_name=client_name,
                    primary_name=primary_name,
                    alternate_names=payload["alternate_names"],
                    spokespeople=payload["spokespeople"],
                    products=payload["products"],
                    guidance=payload["guidance"],
                    api_key=st.secrets["key"],
                    model=analysis_context.DEFAULT_ANALYSIS_CONTEXT_MODEL,
                )
            st.session_state.analysis_context_pending_suggestions = suggestions
            st.rerun()
        except Exception as e:
            st.error(f"Could not generate context suggestions: {e}")
with helper_col2:
    if st.button("Clear all below", key="analysis_context_clear_below", use_container_width=True):
        st.session_state.analysis_alternate_names = []
        st.session_state.analysis_spokespeople = []
        st.session_state.analysis_products = []
        st.session_state.analysis_guidance = ""
        st.session_state.ui_alternate_names = []
        st.session_state.ui_spokespeople = []
        st.session_state.ui_products = []
        st.session_state.ui_toning_rationale = ""
        st.session_state.analysis_context_suggestion_payload = None
        st.session_state.analysis_context_tag_widget_version += 1
        st.rerun()

if st.session_state.get("analysis_context_suggestion_success"):
    st.success("AI context suggestions added to the fields above.")
    st.session_state.analysis_context_suggestion_success = False

tag_key_suffix = st.session_state.analysis_context_tag_widget_version
alternate_names = st_tags(
    label="Alternate names / aliases",
    text="Press enter to add more",
    maxtags=20,
    value=payload["alternate_names"],
    key=f"analysis_context_aliases_tags_{tag_key_suffix}",
)
spokespeople = st_tags(
    label="Key spokespeople",
    text="Press enter to add more",
    maxtags=20,
    value=payload["spokespeople"],
    key=f"analysis_context_spokespeople_tags_{tag_key_suffix}",
)
products = st_tags(
    label="Products / sub-brands / initiatives",
    text="Press enter to add more",
    maxtags=20,
    value=payload["products"],
    key=f"analysis_context_products_tags_{tag_key_suffix}",
)
guidance = st.text_area(
    "Additional rationale, context, or guidance (optional)",
    value=payload["guidance"],
    height=110,
    help="Use this for analytical framing, nuances, or focus that should shape AI-generated summaries and observations.",
)

with st.expander("AI suggestion rationale", expanded=False):
    suggestion_payload = st.session_state.get("analysis_context_suggestion_payload")
    if suggestion_payload:
        assessment = str(suggestion_payload.get("assessment", "") or "").strip()
        if assessment:
            st.write("**Assessment**")
            st.write(assessment)

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

st.subheader("Analysis Focus")
st.caption("Set shared rules for what should be excluded from qualitative workflows like Top Stories, Tagging, Sentiment, Authors, Outlets, and Regions.")

exclude_aggregators_from_outlet_insights = st.checkbox(
    "Exclude aggregators from Outlet metrics / insights",
    value=payload["exclude_aggregators_from_outlet_insights"],
    help="Recommended. Keeps aggregator coverage out of Outlet charts and narrative while leaving the rest of the dataset alone.",
)

qualitative_excluded_flags = st.multiselect(
    "Exclude junky coverage flags from qualitative insights",
    options=payload["available_junky_flags"],
    default=payload["qualitative_excluded_flags"],
    help="Recommended defaults are Press Release and Advertorial. Add others when the dataset warrants it.",
)

qualitative_exclusion_keep_keys = list(payload.get("qualitative_exclusion_keep_keys", []))
if qualitative_excluded_flags:
    qualitative_preview = analysis_context.build_coverage_flag_removal_preview(
        st.session_state.get("df_traditional"),
        qualitative_excluded_flags,
        keep_row_keys=set(payload.get("qualitative_exclusion_keep_keys", [])),
    )
    qual_stat1, qual_stat2 = st.columns(2, gap="medium")
    with qual_stat1:
        st.metric("Rows excluded from qualitative insights", f"{qualitative_preview['removed_rows']:,}")
    with qual_stat2:
        st.metric("Mentions excluded from qualitative insights", f"{qualitative_preview['removed_mentions']:,}")

    if not qualitative_preview["counts_df"].empty:
        st.write("**Qualitative exclusion preview by flag**")
        st.dataframe(
            qualitative_preview["counts_df"],
            use_container_width=True,
            hide_index=True,
            column_config={
                "Rows": st.column_config.NumberColumn("Rows", format="%d"),
                "Mentions": st.column_config.NumberColumn("Mentions", format="%,d"),
            },
        )

    if not qualitative_preview["sample_df"].empty:
        st.write("**Rows excluded from qualitative insights**")
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
        if qualitative_exclusion_keep_keys:
            st.caption(f"Keeping {len(qualitative_exclusion_keep_keys)} row(s) in qualitative workflows even though they match the selected junky flags.")
    else:
        st.info("No rows with those selected junky flags are present in the current cleaned dataset.")
else:
    qualitative_exclusion_keep_keys = []
    st.info("No junky coverage flags are currently selected for qualitative exclusion.")

st.subheader("⚠ Delete from dataset")
st.caption("Use with caution. Anything selected here is removed from downstream working views unless you explicitly keep a row below.")

dataset_excluded_flags = st.multiselect(
    "Delete junky coverage flags from dataset",
    options=payload["available_junky_flags"],
    default=payload["dataset_excluded_flags"],
    help="No defaults here. Select only the flags you are comfortable removing from the working dataset.",
)

dataset_exclusion_keep_keys = list(payload.get("dataset_exclusion_keep_keys", []))
if dataset_excluded_flags:
    preview = analysis_context.build_coverage_flag_removal_preview(
        st.session_state.get("df_traditional"),
        dataset_excluded_flags,
        keep_row_keys=set(payload.get("dataset_exclusion_keep_keys", [])),
    )
    preview_stat1, preview_stat2 = st.columns(2, gap="medium")
    with preview_stat1:
        st.metric("Rows that would be removed", f"{preview['removed_rows']:,}")
    with preview_stat2:
        st.metric("Mentions that would be removed", f"{preview['removed_mentions']:,}")

    if not preview["counts_df"].empty:
        st.write("**Deletion preview by flag**")
        st.dataframe(
            preview["counts_df"],
            use_container_width=True,
            hide_index=True,
            column_config={
                "Rows": st.column_config.NumberColumn("Rows", format="%d"),
                "Mentions": st.column_config.NumberColumn("Mentions", format="%,d"),
            },
        )

    if not preview["sample_df"].empty:
        st.write("**Rows flagged for deletion**")
        st.caption("Uncheck `Delete` to keep a specific row even when it matches the selected junky flags.")
        removal_editor = st.data_editor(
            preview["sample_df"][["Remove", "Headline", "Outlet", "Coverage Flags", "Link", "Row Key"]],
            use_container_width=True,
            hide_index=True,
            disabled=["Headline", "Outlet", "Coverage Flags", "Link", "Row Key"],
            column_order=["Remove", "Headline", "Outlet", "Coverage Flags", "Link"],
            column_config={
                "Remove": st.column_config.CheckboxColumn("Delete", width="small"),
                "Headline": st.column_config.Column("Headline", width="large"),
                "Outlet": st.column_config.Column("Outlet", width="medium"),
                "Coverage Flags": st.column_config.Column("Coverage Flags", width="medium"),
                "Link": st.column_config.LinkColumn("Link", width="small", display_text="open"),
            },
            key="analysis_context_dataset_removal_editor",
        )
        dataset_exclusion_keep_keys = removal_editor.loc[~removal_editor["Remove"], "Row Key"].astype(str).tolist()
        if dataset_exclusion_keep_keys:
            st.caption(f"Keeping {len(dataset_exclusion_keep_keys)} row(s) even though they match the selected junky flags.")
    else:
        st.info("No rows with those selected junky flags are present in the current cleaned dataset.")
else:
    dataset_exclusion_keep_keys = []
    st.info("No coverage flags are currently selected for dataset deletion.")

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
            guidance=guidance,
            qualitative_excluded_flags=qualitative_excluded_flags,
            dataset_excluded_flags=dataset_excluded_flags,
            exclude_aggregators_from_outlet_insights=exclude_aggregators_from_outlet_insights,
            qualitative_exclusion_keep_keys=qualitative_exclusion_keep_keys,
            dataset_exclusion_keep_keys=dataset_exclusion_keep_keys,
        )
        st.session_state.analysis_context_save_success = True

with save_status_col:
    if st.session_state.get("analysis_context_save_success"):
        st.success("Analysis context saved.")

with st.expander("Current shared context preview", expanded=False):
    preview = analysis_context.build_analysis_context_text(st.session_state)
    if preview:
        st.code(preview, language="text")
    else:
        st.info("No analysis context has been saved yet.")

with st.expander("AI prompt preview", expanded=False):
    st.code(
        analysis_context.build_analysis_context_discovery_prompt(
            client_name=client_name,
            primary_name=primary_name,
            alternate_names=payload["alternate_names"],
            spokespeople=payload["spokespeople"],
            products=payload["products"],
            guidance=payload["guidance"],
        ),
        language="text",
    )
