# 11-Download.py

from __future__ import annotations

import warnings

import pandas as pd
import streamlit as st
from ui.page_help import set_page_help_context
from utils.deployment import rapid_labeling_enabled
from utils.time_display import format_local_timestamp
from utils.download_freshness import content_fingerprint, frame_fingerprint, workbook_fingerprint

from processing.download_exports import (
    build_clean_workbook_bytes,
    build_report_copy_docx_bytes,
    build_scoped_traditional_export_bundle,
    iter_clean_workbook_sheets,
    labeling_audit_available,
    prepare_report_copy_content,
    rapid_analysis_context_matches,
    rapid_report_regeneration_needed,
    stale_regions_export_observations,
)
from processing.notebooklm_exports import (
    compile_notebooklm_bundle, notebooklm_client_short, prepare_notebooklm_rows,
)
from processing.rapid_sampling import rapid_sample_matches_scope
from processing.report_observation_freshness import established_report_regeneration_needed, stale_coverage_themes

warnings.filterwarnings("ignore")

st.title("Download")
st.caption("Build and download the cleaned workbook, Word report, and NotebookLM bundle.")
set_page_help_context(st.session_state, "Download")

if not st.session_state.get("standard_step", False):
    st.error("Please complete Basic Cleaning before trying this step.")
    st.stop()

def render_download(asset, title, filename, *, fingerprint, button_key, prerequisite=False, summary=None):
    data = st.session_state.get(f"{asset}_bytes")
    if data is None:
        return
    earlier = prerequisite or fingerprint is None or st.session_state.get(f"_{asset}_fingerprint") != fingerprint
    if earlier and not prerequisite:
        st.warning(f"The downloadable {title} is an earlier version. Rebuild it before using it.")
    label = f"Download earlier {title}" if earlier else f"Download {title}"
    st.download_button(label, data, file_name=filename, type="primary", key=button_key)
    if built_at := st.session_state.get(f"{asset.removesuffix('_docx').removesuffix('_zip')}_built_at"):
        prefix = "Earlier version built" if earlier else "Built"
        st.caption(f"{prefix}: {built_at}")
    if summary:
        st.caption(summary)


def save_download(asset, data, fingerprint, title):
    rebuilt = f"{asset}_bytes" in st.session_state
    st.session_state[f"{asset}_bytes"] = data
    st.session_state[f"_{asset}_fingerprint"] = fingerprint
    timestamp = format_local_timestamp()
    st.session_state[f"{asset.removesuffix('_docx').removesuffix('_zip')}_built_at"] = timestamp
    st.success(f"{title} {'rebuilt' if rebuilt else 'built'} at {timestamp}")


st.divider()
st.subheader("Clean data workbook")

rapid_enabled = rapid_labeling_enabled()
rapid_scope_matches = not rapid_enabled or rapid_sample_matches_scope(st.session_state)
rapid_context_matches = not rapid_enabled or rapid_analysis_context_matches(st.session_state)
if not rapid_scope_matches:
    st.warning(
        "The saved Rapid Labeling sample does not match the current dataset scope. "
        "A new workbook will omit its Rapid results. An earlier workbook may still contain them. "
        "To include Rapid results in a new workbook, restore a compatible scope "
        "or prepare a new sample and rerun labeling in Rapid Labeling. Your saved work is still in the session."
    )
elif not rapid_context_matches:
    st.warning(
        "A new workbook will omit Rapid results prepared with a different Analysis Context. "
        "Restore that Analysis Context or prepare a new Rapid sample and rerun labeling to include Rapid results."
    )

if stale_regions := stale_regions_export_observations(st.session_state):
    st.warning(
        f"Regenerate {', '.join(stale_regions)} observations in Regions to include them in a new workbook or Word report. "
        "Rebuilding alone will omit those observations."
    )

if labeling_audit_available(st.session_state):
    st.checkbox("Include labeling audit columns", value=False, key="include_labeling_audit_columns")
else:
    st.session_state.include_labeling_audit_columns = False

build_xlsx = st.button("Build cleaned data workbook", key="build_clean_workbook")
if build_xlsx or "clean_excel_bytes" in st.session_state:
    workbook_key = None
    try:
        with st.spinner("Checking workbook inputs..." if not build_xlsx else "Building workbook now..."):
            sheets = iter_clean_workbook_sheets(st.session_state)
            if build_xlsx:
                sheets = list(sheets)
            workbook_key = workbook_fingerprint(sheets, st.session_state.get("original_ave_col") or "AVE")
            if build_xlsx:
                data = build_clean_workbook_bytes(st.session_state, sheets=sheets)
                save_download("clean_excel", data, workbook_key, "Cleaned workbook")
    except Exception as exc:
        st.error(f"Error preparing Excel workbook: {exc}")
    render_download(
        "clean_excel", "cleaned data workbook", f"{st.session_state.export_name} - clean_data.xlsx",
        fingerprint=workbook_key, button_key="download_clean_workbook",
    )

st.divider()
st.subheader("Report copy document")

rapid_observations = tuple(
    family for family in ("sentiment", "tag")
    if rapid_enabled and st.session_state.get(f"rapid_{family}_observation_output")
)
report_actions = []
if rapid_observations:
    if not rapid_scope_matches:
        report_actions.append(
            "Restore a compatible scope or prepare a new Rapid sample, rerun labeling, "
            "and regenerate its observations in Rapid Labeling → Insights."
        )
    elif not rapid_context_matches:
        report_actions.append(
            "Resolve the Analysis Context warning in Rapid Labeling, then regenerate the Rapid observations in Insights."
        )
    elif regenerate := rapid_report_regeneration_needed(st.session_state, rapid_observations):
        report_actions.append(f"Regenerate Rapid {' and '.join(regenerate)} observations in Rapid Labeling → Insights.")

for observation in established_report_regeneration_needed(st.session_state):
    report_actions.append(f"Regenerate {observation}.")

report_content = None
if not report_actions and (
    "report_copy_docx_bytes" in st.session_state
    or st.session_state.get("author_insights_summaries")
    or st.session_state.get("outlet_insights_summaries")
):
    try:
        report_content = prepare_report_copy_content(st.session_state)
        for theme in stale_coverage_themes(
            st.session_state, author_blocks=report_content["author_blocks"], outlet_blocks=report_content["outlet_blocks"],
        ):
            report_actions.append(f"Update the shortlist if needed and regenerate {theme}.")
    except Exception as exc:
        st.error(f"Error checking Word report inputs: {exc}")
        report_actions.append("Check the saved Insights and try again.")

if report_actions:
    earlier_notice = (
        "The downloadable report copy document is an earlier version. "
        if "report_copy_docx_bytes" in st.session_state else ""
    )
    st.warning(f"{earlier_notice}Before rebuilding the Word report: {' '.join(report_actions)}")

build_report_copy = st.button(
    "Build report copy document",
    key="build_report_copy_document",
    help="Creates a Word document with narrative insights and representative linked examples.",
    disabled=bool(report_actions),
)
if build_report_copy or "report_copy_docx_bytes" in st.session_state:
    report_key = None
    if not report_actions:
        try:
            content = report_content if report_content is not None else prepare_report_copy_content(st.session_state)
            report_key = content_fingerprint(content)
            previous_sections = st.session_state.get("_report_copy_sections", set())
            narrative_sections = {
                "sentiment": "Sentiment → Insights",
                "tags": "Tagging → Insights",
                "top_stories": "Top Stories → Top Story Insights",
                "rapid_sentiment": "Rapid Labeling → Insights (sentiment)",
                "rapid_tags": "Rapid Labeling → Insights (tags)",
                "regions_sections": "Regions",
            }
            sections = {key for key in narrative_sections if key != "top_stories" and any(content[key])}
            if content["top_stories"][0]:
                sections.add("top_stories")
            removed = previous_sections - sections
            missing = [name for key, name in narrative_sections.items()
                       if key in removed and (rapid_enabled or not key.startswith("rapid_"))]
            if missing:
                st.info(
                    f"A new Word report will omit observations no longer available from {', '.join(missing)}. "
                    "Generate them again before rebuilding if you want to include them."
                )
            if build_report_copy:
                with st.spinner("Building report copy document..."):
                    data = build_report_copy_docx_bytes(st.session_state, content=content)
                    save_download("report_copy_docx", data, report_key, "Report copy document")
                    st.session_state["_report_copy_sections"] = sections
        except Exception as exc:
            st.error(f"Error preparing Word report: {exc}")
    render_download(
        "report_copy_docx", "report copy document", f"{st.session_state.export_name} - report_copy.docx",
        fingerprint=report_key, button_key="download_report_copy_document", prerequisite=bool(report_actions),
    )

st.divider()
st.subheader("NotebookLM bundle")

build_nlm = st.button(
    "Build NotebookLM bundle (zip)",
    key="build_notebooklm_bundle",
    help=(
        "Creates a zip of JSON-formatted text files for NotebookLM. "
        "If the dataset exceeds ~50 files worth of content, a random sample is taken to stay within upload limits."
    ),
)
if build_nlm or "notebooklm_zip_bytes" in st.session_state:
    notebook_key = None
    try:
        with st.spinner("Building NotebookLM bundle..." if build_nlm else "Checking NotebookLM inputs..."):
            scoped_traditional, _, _ = build_scoped_traditional_export_bundle(st.session_state)
            rows = prepare_notebooklm_rows(scoped_traditional, st.session_state.get("df_social", pd.DataFrame()))
            client_short = notebooklm_client_short(st.session_state.client_name)
            input_key = content_fingerprint((client_short, frame_fingerprint(rows)))
            checked_input, checked_bundle = st.session_state.get("_notebooklm_checked_input", (None, None))
            if build_nlm or input_key != checked_input:
                zip_io, info, notebook_key = compile_notebooklm_bundle(rows, client_short, write_archive=build_nlm)
                st.session_state["_notebooklm_checked_input"] = (input_key, notebook_key)
            else:
                notebook_key = checked_bundle
            if build_nlm:
                save_download("notebooklm_zip", zip_io.getvalue(), notebook_key, "NotebookLM bundle")
                st.session_state.notebooklm_info = info
    except Exception as exc:
        st.error(f"Error preparing NotebookLM bundle: {exc}")

    info = st.session_state.get("notebooklm_info", {})
    summary = (
        f"Rows in cleaned dataset at build: {info.get('total_rows', 0):,} · "
        f"Rows in bundle: {info.get('rows_included', 0):,} · "
        f"Files: {info.get('files_created', 0)} (max {info.get('max_files', 0)}), "
        f"Rows/file cap: {info.get('max_rows_per_file', 0)}, "
        f"Word limit/file: {info.get('max_words_per_file', 0):,}, "
        f"Size limit/file: {info.get('max_bytes_per_file', 0) // (1024 * 1024)} MB."
    ) if info else None
    render_download(
        "notebooklm_zip", "NotebookLM bundle", f"NLM_prepared_data-{info.get('client_short', 'Client')}.zip",
        fingerprint=notebook_key, button_key="download_notebooklm_bundle", summary=summary,
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
