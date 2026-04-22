# 5-Translation.py

from __future__ import annotations

import warnings

import streamlit as st
from ui.page_help import set_page_help_context

from processing.translation import (
    init_translation_state,
    ensure_translation_columns,
    count_non_english_records,
    get_non_english_records,
    translate_headlines,
    translate_snippets,
    headline_translation_done,
)

warnings.filterwarnings("ignore")

st.title("Translation")
st.caption("Translate non-English headlines and snippets so the rest of the workflow can be reviewed and summarized in English.")
set_page_help_context(st.session_state, "Translation")


def display_non_english_records(df, title: str) -> None:
    non_english_df = get_non_english_records(df)
    if len(non_english_df) > 0:
        with st.expander(f"{title} - Non-English"):
            st.dataframe(
                non_english_df,
                use_container_width=True,
                hide_index=True,
            )


if not st.session_state.get("standard_step", False):
    st.error("Please complete Basic Cleaning before trying this step.")
    st.stop()

init_translation_state(st.session_state)

st.session_state.df_traditional = ensure_translation_columns(st.session_state.df_traditional)
st.session_state.df_social = ensure_translation_columns(st.session_state.df_social)

trad_non_eng = count_non_english_records(st.session_state.df_traditional)
soc_non_eng = count_non_english_records(st.session_state.df_social)
total_non_eng = trad_non_eng + soc_non_eng

headline_done = headline_translation_done(
    st.session_state.df_traditional,
    st.session_state.df_social,
    st.session_state.translated_headline,
)
snippet_done = st.session_state.translated_snippet

# Status message
if headline_done and snippet_done:
    st.subheader("✓ Translation complete.")
    display_non_english_records(st.session_state.df_traditional, "Traditional")
    display_non_english_records(st.session_state.df_social, "Social")
    st.stop()

elif headline_done:
    st.subheader("✓ Headline translation complete.")
elif snippet_done:
    st.subheader("✓ Snippet translation complete.")

# No work needed at all
if total_non_eng == 0:
    if not (headline_done or snippet_done):
        st.subheader("No translation required")
    st.stop()

# Preview current non-English records
st.write(f"There are {total_non_eng} non-English records in your data.")

display_non_english_records(st.session_state.df_traditional, "Traditional")
display_non_english_records(st.session_state.df_social, "Social")

# Translation options remain visible even after partial completion
st.subheader("Pick columns for translation")

headline_to_english = st.checkbox(
    "Headline",
    value=not headline_done,
    disabled=headline_done,
)

snippet_to_english = st.checkbox(
    "Snippet (full text)",
    value=False,
    disabled=snippet_done,
)

if snippet_to_english and not snippet_done:
    st.warning(
        "WARNING: Snippet translation will overwrite the original text and may cut off the ends of articles longer than ~700 words."
    )

nothing_left_to_translate = headline_done and snippet_done

go_clicked = st.button(
    "Go!",
    type="primary",
    disabled=nothing_left_to_translate,
)

if go_clicked:
    if not headline_to_english and not snippet_to_english:
        st.warning("Please select at least one translation option.")
        st.stop()

    st.warning("Stay on this page until translation is complete")

    if headline_to_english and not headline_done:
        with st.spinner("Translating headlines..."):
            trad, social = translate_headlines(
                st.session_state.df_traditional,
                st.session_state.df_social,
            )
            st.session_state.df_traditional = trad
            st.session_state.df_social = social
            st.session_state.translated_headline = True
            st.success("Done translating headlines!")

    if snippet_to_english and not snippet_done:
        with st.spinner("Translating snippets..."):
            trad, social = translate_snippets(
                st.session_state.df_traditional,
                st.session_state.df_social,
            )
            st.session_state.df_traditional = trad
            st.session_state.df_social = social
            st.session_state.translated_snippet = True
            st.success("Done translating snippets!")

    st.rerun()
