# 11-Download.py

from __future__ import annotations

import warnings

import pandas as pd
import streamlit as st
from ui.page_help import set_page_help_context

from processing.download_exports import (
    build_clean_workbook_bytes,
    build_report_copy_docx_bytes,
    build_scoped_traditional_export_bundle,
)
from processing.notebooklm_exports import build_notebooklm_zip

warnings.filterwarnings("ignore")

st.title("Download")
st.caption("Build and download the cleaned workbook and NotebookLM-ready bundle from the current session state.")
set_page_help_context(st.session_state, "Download")

if not st.session_state.get("standard_step", False):
    st.error("Please complete Basic Cleaning before trying this step.")
    st.stop()

st.divider()
st.subheader("Clean data workbook")

had_clean_workbook = "clean_excel_bytes" in st.session_state

build_xlsx = st.button("Build cleaned data workbook", key="build_clean_workbook")
if build_xlsx:
    try:
        with st.spinner("Building workbook now..."):
            st.session_state.clean_excel_bytes = build_clean_workbook_bytes(st.session_state)
            st.session_state.clean_excel_built_at = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
            action_word = "rebuilt" if had_clean_workbook else "built"
            st.success(f"Cleaned workbook {action_word} at {st.session_state.clean_excel_built_at}")
    except Exception as e:
        st.error(f"Error building Excel workbook: {e}")

if "clean_excel_bytes" in st.session_state:
    export_name = f"{st.session_state.export_name} - clean_data.xlsx"
    st.download_button(
        "Download cleaned data workbook",
        st.session_state.clean_excel_bytes,
        file_name=export_name,
        type="primary",
        key="download_clean_workbook",
    )

    if "clean_excel_built_at" in st.session_state:
        st.caption(f"Current workbook built: {st.session_state.clean_excel_built_at}")

st.divider()
st.subheader("Report copy document")

had_report_copy = "report_copy_docx_bytes" in st.session_state

build_report_copy = st.button(
    "Build report copy document",
    key="build_report_copy_document",
    help="Creates a lightly formatted Word document with the current narrative insights and representative linked examples.",
)
if build_report_copy:
    try:
        with st.spinner("Building report copy document..."):
            st.session_state.report_copy_docx_bytes = build_report_copy_docx_bytes(st.session_state)
            st.session_state.report_copy_built_at = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
            action_word = "rebuilt" if had_report_copy else "built"
            st.success(f"Report copy document {action_word} at {st.session_state.report_copy_built_at}")
    except Exception as e:
        st.error(f"Error building report copy document: {e}")

if "report_copy_docx_bytes" in st.session_state:
    export_name = f"{st.session_state.export_name} - report_copy.docx"
    st.download_button(
        "Download report copy document",
        st.session_state.report_copy_docx_bytes,
        file_name=export_name,
        type="primary",
        key="download_report_copy_document",
    )

    if "report_copy_built_at" in st.session_state:
        st.caption(f"Current report copy document built: {st.session_state.report_copy_built_at}")

st.divider()
st.subheader("NotebookLM bundle")

had_notebooklm_bundle = "notebooklm_zip_bytes" in st.session_state

build_nlm = st.button(
    "Build NotebookLM bundle (zip)",
    key="build_notebooklm_bundle",
    help=(
        "Creates a zip of JSON-formatted text files for NotebookLM. "
        "If the dataset exceeds ~50 files worth of content, "
        "a random sample is taken to stay within upload limits."
    ),
)

if build_nlm:
    try:
        with st.spinner("Building NotebookLM bundle..."):
            scoped_traditional, _excluded_rows, _excluded_counts = build_scoped_traditional_export_bundle(st.session_state)
            nlm_zip_io, nlm_info = build_notebooklm_zip(
                scoped_traditional,
                st.session_state.df_social,
                client_name=st.session_state.client_name,
            )
        st.session_state.notebooklm_zip_bytes = nlm_zip_io.getvalue()
        st.session_state.notebooklm_info = nlm_info
        st.session_state.notebooklm_built_at = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        action_word = "rebuilt" if had_notebooklm_bundle else "built"
        st.success(f"NotebookLM bundle {action_word} at {st.session_state.notebooklm_built_at}")
    except ValueError as e:
        st.error(str(e))
    except Exception as e:
        st.error(f"Error building NotebookLM bundle: {e}")

if "notebooklm_zip_bytes" in st.session_state:
    info = st.session_state.get("notebooklm_info", {})
    client_short = info.get("client_short", "Client")
    zip_filename = f"NLM_prepared_data-{client_short}.zip"

    st.download_button(
        "Download NotebookLM bundle",
        data=st.session_state.notebooklm_zip_bytes,
        file_name=zip_filename,
        type="primary",
    )

    if "notebooklm_built_at" in st.session_state:
        st.caption(f"Current NotebookLM bundle built: {st.session_state.notebooklm_built_at}")

    if info:
        st.caption(
            f"Rows in cleaned dataset: {info.get('total_rows', 0):,} · "
            f"Rows included in bundle: {info.get('rows_included', 0):,} · "
            f"Files: {info.get('files_created', 0)} (max {info.get('max_files', 0)}), "
            f"Rows/file cap: {info.get('max_rows_per_file', 0)}, "
            f"Word limit/file: {info.get('max_words_per_file', 0):,}, "
            f"Size limit/file: {info.get('max_bytes_per_file', 0) // (1024 * 1024)} MB."
        )

st.divider()
client_name = st.session_state.client_name

with st.expander("NotebookLM prompt examples for this client"):
    st.markdown(
        f"""
### Executive Summary
Generate a concise 2-paragraph executive summary of the media coverage of **{client_name}**.  
Present the information as though it is going to be included as a lead into a media briefing for an executive.  
Focus on high-level concepts rather than specific facts and figures.

---

### High-level coverage summary

You are analyzing media coverage for **{client_name}**.  
Summarize the overall coverage tone, key themes, and notable storylines across this dataset.  
Highlight the most influential outlets and authors, and note any spikes in volume, sentiment shifts, or recurring issues.

---

### Coverage Themes
What are the 5–7 key themes in the coverage that would be pertinent to **{client_name}**, and the communications and public relations professionals who work there?  
For each theme, include **5 example headlines** of where that topic was found. For each headline, also include the **date and outlet**.

---

### Competitive Comparison
Are any of **{client_name}**’s competitors mentioned in any of the stories?  
If so, how does the media tend to characterize them? How are they compared to **{client_name}**?

---

### SWOT Analysis (Media Coverage Perspective)

Task  
Analyze the news articles and broadcast transcripts related to **{client_name}** contained in this notebook.  
Generate a media-coverage-driven SWOT analysis that reflects how **{client_name}** is positioned through earned media, based solely on observable patterns, themes, and narratives in the coverage.

Audience & Perspective  
Write for communications and PR professionals responsible for understanding how organizational reputation is shaped in the media.  
Act strictly as a **media analyst**, not as a strategist or advisor.

Do not:
- Recommend actions, tactics, messaging, or strategy  
- Suggest what the organization “should” do  
- Speculate beyond what is supported by coverage evidence  

Analytical Framing Rules  
- Base all points on **coverage patterns**, not assumptions about operations or intent  
- Use neutral analytical phrasing such as:
- “coverage reflects…”
- “media frequently positions…”
- “reporting emphasizes…”  
- Avoid consulting or marketing jargon (e.g., “own the narrative,” “optimize messaging,” “capitalize on”)  
- Where relevant, distinguish between:
- **Story volume**
- **Story type**
- **Audience reach / outlet credibility**
as drivers of reputational impact  

SWOT Structure  

Provide **4–5 concise bullets per quadrant**, written in clear, plain language and consistent in specificity.

**Strengths**  
Positive reputational signals conveyed through coverage (e.g., authority positioning, association with innovation or leadership, visibility in high-credibility outlets, trusted expert commentary).

**Weaknesses**  
Reputational vulnerabilities or negative associations reflected in coverage (e.g., incident-driven reporting, legal or safety narratives, uneven sub-brand perception, contextual linkage to negative events).

**Opportunities**  
Gaps, under-developed narratives, or emerging themes suggesting potential for broader or more balanced reputational positioning — without implying recommended action.

**Threats**  
External or category-level narratives present in coverage that may pose reputational risk (e.g., regulatory scrutiny, cybersecurity issues, political framing, sector instability).

Observations Section  

After the SWOT bullets, write a brief **“SWOT Analysis Observations”** section.  
Include **one short paragraph per quadrant** synthesizing what the coverage patterns suggest at a higher level.

These observations should:
- Explain why the patterns matter from a media-analysis standpoint  
- Not repeat bullets verbatim  
- Not introduce new unsupported claims  

Tone & Constraints  
- Analytical, neutral, evidence-based  
- No prescriptive or advisory language  
- No operational judgments  
- No generalized industry claims unless clearly reflected in the sources  

---

### Issues and Risk Monitoring
Review the coverage of {client_name} and describe any emerging or ongoing issues or risks that could affect the organization’s reputation.
Present your findings as concise, clearly structured text (not tables).
For each issue, include:
- A brief summary of what the issue is about
- An estimate of how common it is in the dataset (frequent / occasional / rare)
- The general sentiment or framing of the coverage (positive / neutral / negative)
- A few representative headlines or short quotes that illustrate the tone or context
- A sentence or two on whether the issue appears to be growing, stable, or fading in prominence

---

### Implications and Recommendations for Comms
Based on the coverage patterns for {client_name}, outline observations and cautious considerations that might inform communications planning — without prescribing strategy.
Write your response as concise, clearly structured text (not tables).
For example, you might note:
- Which narratives appear most influential or persistent
- Any patterns that might warrant closer monitoring or further validation
- Possible areas where proactive engagement or clarification could help manage perceptions
- Observations that may be useful for future reporting or research focus

---

### Misinformation & False Claims
Search the coverage of {client_name} for any mentions of false, misleading, or unverified information about the brand itself.
Exclude stories where {client_name} is the one accused of misleading others.
For each example, describe the nature of the misinformation, its origin (if known), and how the media handled it — corrected, ignored, or amplified it.
Present your findings as structured text sections, not as a table.

---

### Negative Message Discovery
You are analyzing media coverage of {client_name}.
Ignore any existing sentiment labels or numerical scores in the data.
Your goal is to identify and describe any negative messages or narratives about {client_name} that appear across the coverage, from the obvious to the more burried.

For each distinct negative message you find:
- Start a new short section with a clear heading naming the message (e.g., “Concerns about Product Safety”, “Leadership Controversies”).
- Summarize what the message is and how it is expressed in the media (2–3 sentences).
- Explain what aspect of {client_name} it relates to (e.g., reputation, products, financials, operations, ethics, customer experience, etc.).
- Include one or two short quotes or paraphrased examples, with outlet names and dates mentioned inline where available.
- Indicate whether the message appears isolated (few stories) or recurring (multiple outlets, ongoing).

At the end, include a short paragraph summarizing what these negative messages collectively suggest about how {client_name} is being portrayed, without speculating about motives or offering recommendations.
Present your findings as structured text sections, not as a table.
"""
    )
