# MIG_MasterApp Project Context

## Purpose

MIG_MasterApp is a Streamlit application for turning an Agility PR media-coverage export into an analyst-reviewed, report-ready media-intelligence package. Its primary users are communications and PR analysts who need to clean coverage data, group syndicated copies into story families, apply client context, label coverage, review AI work, build insight outputs, and export usable workbooks and report copy.

The product is designed around a few durable principles:

- Preserve source data and make working changes traceable and reversible where practical.
- Keep analyst workflows fast, with review effort concentrated on meaningful ambiguity and impact.
- Prefer deterministic, reproducible data transformations for cleaning, grouping, and export joins.
- Treat analyst decisions as authoritative over AI outputs.
- Use canonical grouped-story identity consistently across workflows.
- Keep request/debug plumbing separate from analyst-facing default outputs.

The app is traditional-media focused today, although Basic Cleaning and export support a social split. A separate social-analysis application remains future work rather than an active extension of this navigation.

## Runtime and Dependencies

- Python: `3.13.5` (Pipenv requirement).
- UI: Streamlit `~=1.55` with `streamlit-tags` and `streamlit-extras`.
- Data and grouping: pandas `~=2.3`, NumPy `~=2.0`, SciPy `~=1.13`, scikit-learn `~=1.7`.
- LLM integrations: OpenAI Python SDK `~=2.6`; direct HTTP requests for OpenRouter Decisions API / Jev.
- Exports: openpyxl, XlsxWriter, python-docx, dill, Jinja2.
- Other notable utilities: deep-translator, titlecase, unidecode.

The standard OpenAI credential convention is `st.secrets["key"]`. Rapid Labeling uses `st.secrets["openrouter_key"]`, with `OPENROUTER_API_KEY` as its environment fallback. These are separate integrations. The normal model defaults are currently `gpt-5.6-luna`; Rapid first pass is pinned to `typesafe/jev-1.13` through OpenRouter's Decisions API.

## Environments and Deployment

Development is routinely done locally through Pipenv and Streamlit. GitHub is the primary repository, and Streamlit Community Cloud generally follows pushed GitHub state quickly. An AWS deployment may be managed independently and can lag behind GitHub or Community Cloud. Do not assume the three environments are on the same revision.

## Application Structure

`app.py` configures the wide Streamlit app and delegates navigation to `ui/app_shell.py`. The sidebar routes to these pages:

1. Getting Started
2. Basic Cleaning
3. Analysis Context
4. Authors
5. Outlets
6. Translation
7. Top Stories
8. Regions
9. Sentiment
10. Tagging
11. Rapid Labeling
12. Download
13. Save & Load

The main implementation layers are:

- `pages/`: Streamlit page orchestration and page-local controls.
- `processing/`: data transformations, grouping, AI prompt/result logic, resolution, insights, and exports.
- `ui/`: reusable workflow views, charts, sidebar behavior, page help, and review presentation.
- `utils/`: session snapshots, checkpoints, formatting, timing, IO, and shared batch helpers.
- `tests/`: main unit/regression suite.
- `visual_checker_handoff/drop_in_visual_checker/`: a separate drop-in visual-validation handoff with its own focused test file; it is not part of the primary app navigation.

The application is intentionally session-state driven. Dataframes and workflow settings live in `st.session_state`; processing modules operate on explicit dataframes and return data rather than mutating UI state from background workers.

## Core Data Model and Shared Concepts

### Coverage dataframes

The uploaded file is retained as source-oriented state, while Basic Cleaning creates working dataframes. Important current tables include:

- `df_untouched`: uploaded/raw source data retained for export/reference.
- `df_traditional`: cleaned traditional-media rows.
- `df_social`: cleaned social rows, separated during cleaning.
- `df_dupes`: removed duplicate rows.
- `df_ai_grouped`: traditional rows carrying the canonical grouping and `Prime Example` marker.
- Workflow-specific row and grouped tables, such as `df_sentiment_rows`, `df_sentiment_unique`, `df_tagging_rows`, `df_tagging_unique`, and `df_jev_sentiment_unique` for Rapid Labeling.

`CLEAN TRAD` is a row-level working/export view. Sample and Rapid sheets are generally grouped-story views and are not interchangeable with the row-level dataset.

### Canonical Group ID

`Group ID` is the authoritative story-family identity. It is the only supported key for carrying grouped labels back onto row-level coverage. Do not join labels by dataframe index, headline, URL, or display order.

Canonical grouping considers media type, normalized headline/snippet text, and `SyndicationId` when available. Hybrid grouping can connect text-similar representatives and rows sharing a syndication identifier. The resulting `Grouping Source` identifies `Text Similarity`, `SyndicationId`, or `SyndicationId + Text Similarity`; `Grouping Warning` captures notable grouping conditions such as multiple syndication IDs or unusually large groups.

Text similarity normally uses cosine `0.935`. Representatives with fewer than 20 normalized snippet tokens are weak evidence, so any edge touching one requires cosine `0.95`; nonblank same-`SyndicationId` links remain strong and unchanged. The sparse edge filter runs before connected components, preventing rejected low-evidence edges from bridging canonical groups without another vectorization or all-pairs pass. Singleton groups have blank `Grouping Source` and `Grouping Warning` because no rows were actually grouped.

### Prime Example

Basic Cleaning marks one `Prime Example` per canonical Group ID. Selection favors preferred wire sources, avoids lower-quality flagged coverage where possible, and considers text completeness, impressions, and date. A prime row is the default representative, not a second identity layer and not the only usable source for a story family.

### Grouped versus row-level metrics

Grouped views aggregate `Mentions`, `Impressions`, and `Effective Reach` by canonical Group ID, and retain `Group Count` where needed. Rows may repeat a grouped result in a row-level export; request-level usage metrics must not be summed from those repeated rows.

### Analysis Context and qualitative policy

Analysis Context defines the monitored collective, shared guidance, qualitative coverage exclusions, data scope, and optional reporting/relevance preferences. Qualitative exclusions remove rows from narrative-oriented workflows without inherently deleting them from the working dataset. Dataset scope settings can exclude rows from downstream working data and exports, subject to explicit keep overrides.

## Story Grouping and Top Stories Identity

Canonical grouping is implemented in `processing/story_grouping.py`. It normalizes syndication IDs, prepares text, clusters within media type and date-aware batches, then builds connected components from text and syndication edges. Prime-example selection happens within those canonical groups.

Top Stories now treats one canonical Group ID as one candidate. `build_grouped_story_candidates()` delegates to the canonical, prime-based candidate builder and does not create new active `TOPMERGE::...` identities. The older `consolidate_top_story_candidates()` function remains for compatibility and is still used by Regions for regional top-story context; it is not the active Top Stories selection identity model.

`Source Group IDs` remains a compatibility and source-family field. Current Top Stories entries normally contain their canonical group ID. Legacy saved `TOPMERGE` records and multi-group `Source Group IDs` can still be interpreted so existing saved sessions continue to load and validate.

Representative-source selection is distinct from grouping. During validation, the app can rotate through distinct eligible underlying rows/URLs from the saved story family, keeping the prime row first. Rotating a source updates representative headline, outlet, URL, snippet, type, and date; it does not change canonical identity or aggregated Mentions, Impressions, or Effective Reach.

## Analysis Context

The Analysis Context page stores a primary entity/client and a broader collective definition:

- alternate entity/client names;
- spokespeople acting for the entity;
- products, sub-brands, and programs;
- highlight-only keywords;
- general shared guidance and sentiment-specific guidance;
- qualitative coverage exclusions and data-scope settings.

Multi-value fields normalize at the canonical input/save boundary. Comma- and newline-separated pasted values are split, whitespace is trimmed, empty values are dropped, and values are deduplicated case-insensitively while preserving first-entered display form. Highlight-only terms support review/highlighting without silently expanding AI entity scope.

Context is used in established Sentiment, established Tagging, Rapid Labeling, and multiple narrative/selection workflows. Its entity terms feed prompt construction and review highlighting. The context requirement is enforced before workflows that depend on it.

## Workflow Overview

### Getting Started

Uploads CSV/XLSX coverage data, captures client/reporting information, performs initial quality reporting, and presents high-level basic statistics. It establishes the upload baseline used by Basic Cleaning resets.

### Basic Cleaning

Runs the core transformation pipeline: standardizes source fields and media types, separates social content, applies duplicate handling, calculates effective reach, creates canonical groups, and marks prime examples. It exposes reconciliation and quality checks. Reset Basic Cleaning returns to the completed upload baseline and clears downstream work.

### Translation

Allows headline/snippet translation after cleaning and Analysis Context preparation. It is a working-data enrichment, not a replacement for the raw uploaded export.

### Top Stories

Uses three stages: Selection, Validation, and Insights.

- Selection filters canonical grouped traditional coverage, applies qualitative exclusions, ranks/saves a shortlist, and preserves one story candidate per Group ID.
- Validation lets the analyst inspect and rotate representative sources, remove a saved story without touching source coverage, or confirm the source. Removal clears related validation confirmation/candidate state and invalidates generated top-story observations.
- Insights generates summaries and linked examples for the saved shortlist.

### Authors and Outlets

Authors has Missing, Outlets, Selection, and Insights work. Missing-author review offers candidate names, bulk obvious-match handling, and undo. Routine successful author changes use a transient toast rather than a layout-shifting banner.

The author-to-outlet workflow compares cleaned coverage with the Agility media database and supports confirmation, manual override, skip, undo, caching, and conservative auto-assignment. Database candidate order is deterministic: normalized exact author-name matches first, then coverage-outlet preference, then original API order. Manual fallback remains available when no useful database match exists.

Outlets supports cleanup/review, candidate selection, and insight output. It includes normalized-name cleanup aids, but those are reference cues rather than automatic source-data reassignment.

### Regions

Regions provides regional analysis and observations from cleaned/grouped coverage. It normally uses canonical grouping, but its regional top-story context still calls the legacy Top Stories consolidation helper. That dependency is a known architectural follow-up.

### Download and Save / Load

Download builds the cleaned workbook, a Word report-copy document, and a NotebookLM bundle from current session state. Save & Load serializes session state to a dill-backed `.pkl` snapshot and restores dataframe/date state, canonical tag definitions, timing fields, and compatible workflow state. Generated binary downloads are intentionally omitted because they can be rebuilt.

## Established Sentiment

Established Sentiment is a five-stage workflow: Setup, AI First Pass, AI Second Opinion, Spot Checks, and Insights.

Setup owns the prepared sample, sentiment scale/type, Analysis Context-derived prompt configuration, and model configuration. Step 2's `Reset Processed Rows` clears first-pass, second-opinion, human-review, observation, checkpoint, and recommendation state for the prepared job while preserving the prepared dataset and setup configuration. Full reconfiguration belongs in Setup.

The first pass and second opinion evaluate sentiment toward the configured collective, not broad topic sentiment. A collective member (alias, spokesperson acting for the entity, product, sub-brand, or program) can make a story relevant even when the parent organization name is absent. Brief but genuine in-scope coverage is not automatically `NOT RELEVANT`; lexical overlap inside a different proper name is not proof of a true entity mention.

Lexical entity matching is a QA signal only. A semantic `NOT RELEVANT` result is preserved, and a lexical conflict can prioritize second opinion without forcing a human review when semantic passes agree. Human/assigned sentiment remains authoritative in final selection.

Second-opinion recommendations are stored targets for the current first-pass population. The recommended target seeds the batch-size control but can be manually overridden; it does not repeatedly regenerate simply because eligible rows remain. A reset or materially new first-pass population creates a new target.

## Established Tagging

Established Tagging also uses Setup, AI First Pass, AI Second Opinion, Spot Checks, and Insights. It supports two durable formulations:

- `Single best tag`: one mutually exclusive tag.
- `Multiple applicable tags`: a set of applicable tags.

`tagging_mode` is durable workflow configuration and is intentionally separate from ephemeral Streamlit widget state. `Reset Processed Rows` clears AI/review/insight state while retaining the prepared dataset, tag definitions, sample choice, and selected formulation. Full data/configuration changes belong in Setup.

Spot Checks choose review controls from the configured mode, not from how narrowly an AI response happened to tag a story. Multi-applicable review remains a multi-select task even if one opinion lists fewer tags than another. Final tags are exported independently of Sentiment.

## Rapid Labeling

Rapid Labeling is an experimental but increasingly production-ready parallel workflow. Its five stages are:

1. Prepare & Configure
2. Run Rapid Labeling
3. AI Second Opinion
4. Spot Checks
5. Insights

Preparation reuses canonical grouped stories, sampling/grouping primitives, Analysis Context, and qualitative-exclusion policy. It can reuse the established Sentiment sample for direct comparison. Tag configuration is optional and is used during labeling.

Each first-pass grouped story receives one OpenRouter Decisions API request to Jev. The request shares one story state and asks five independent questions: 3-way sentiment with `NOT RELEVANT`, 5-way sentiment with `NOT RELEVANT`, ordered sentiment score, independent relevance, and sentiment mixture. When configured, the same request also carries best-fit and independent all-applicable tag questions. Bounded thread-based workers process independent stories; completed results are mapped back to Group ID by the main thread. Errors do not block later rows.

The Rapid first pass keeps the raw opinions separate. The resolver computes machine-effective results without pretending a second opinion is automatic authority. An internally contradictory first pass remains unresolved rather than falling back to a raw label. Luna second opinion is QA/provenance evidence and is optional; all successfully usable first-pass stories remain available for second opinion, while priority rules only rank/recommend them.

Rapid human review is formulation-specific:

- Human 3-way sentiment changes only final/effective 3-way sentiment.
- Human 5-way sentiment changes only final/effective 5-way sentiment.
- Human best-fit tag and human applicable-tags decisions are independent.
- Human `NOT RELEVANT` establishes non-relevance for downstream human-reviewed sentiment use and clears the final machine-derived score.
- No human numeric score is fabricated; the score remains machine-derived supporting evidence.

The analyst-facing Step 4 UI uses first/second opinion language rather than provider names. Raw responses remain stored for audit/debugging but are not part of normal analyst review. Step 2 shows results only after at least one first-pass result exists; its normal outputs are results, machine-effective provenance, and distributions rather than raw-response inspection.

Rapid Step 5 reuses mature observation behavior through Rapid-specific adapters built from final/effective resolver outputs. Sentiment and tag observations are separate, retain linked examples, and use input fingerprints so they become stale only when data they actually consume changes. A second-opinion rerun does not by itself invalidate observations unless it changes an effective/final input. Human-resolved rows do not count as unresolved machine-disagreement warnings.

Rapid page help is mapped to vendor-neutral labels: `Prepare & Configure`, `Run Rapid Labeling`, `AI Second Opinion`, `Spot Checks`, and `Insights`.

## Export Architecture

`processing/download_exports.py` builds exports from current state rather than treating earlier display tables as authoritative outputs. Workflow field families are conditional: a workflow with no meaningful completed output does not create empty labeling placeholders. The Download page shows one `Include labeling audit columns` option only when at least one labeling workflow has meaningful output.

### CLEAN TRAD

`CLEAN TRAD` is row-level. Established Sentiment and Tagging cascade their resolved fields by canonical Group ID. Rapid does the same with a one-row-per-Group-ID resolver adapter and a left merge.

Its grouping metadata remains adjacent for inspection: `Group ID`, `SyndicationId`, `Grouping Source`, and `Grouping Warning`.

Default Rapid fields in `CLEAN TRAD` are:

- `Final Rapid Relevance`
- `Final Rapid Sentiment 3-Way`
- `Final Rapid Sentiment 5-Way`
- `Final Rapid Sentiment Score`
- `Final Rapid Best Tag`
- `Final Rapid Tags`

Those fields are derived fresh through the Rapid human-over-machine resolver. Human assignments take precedence only within the formulation reviewed; machine-effective values are used otherwise. An intentionally unresolved state stays blank and is never replaced with a raw first-pass opinion just to fill the export.

With audit enabled, `CLEAN TRAD` can include final sources, machine-effective values/sources, resolution statuses, score distance, analytical first/second-opinion evidence, and human-review evidence. Request-level fields never belong in this repeated row-level export: raw response, provider/model, tokens, cost, and request errors remain in the grouped `RAPID LABELING SAMPLE` sheet.

Legacy and Rapid output families coexist without precedence or overwrite. Legacy `Final Sentiment`/`Final Tag` retain their own semantics; Rapid fields are explicitly named `Final Rapid ...`.

### Grouped and narrative exports

`SENTIMENT SAMPLE`, `TAGGING SAMPLE`, shared samples where appropriate, and `RAPID LABELING SAMPLE` provide workflow-specific inspection. The grouped Rapid sheet is the authoritative source for Jev request metadata, usage/cost, raw response, and errors. The report-copy document supports established and Rapid sentiment/tag observation families with clear provenance headings so simultaneous insight sources are not silently discarded.

## Page Help and UI Conventions

`ui/page_help.py` owns a shared sidebar Page Help system. Pages call `set_page_help_context(session_state, page, step)`; the app shell resets context around navigation and rerenders the sidebar after the selected page sets its specific step. Help entries are keyed by page and workflow step rather than internal provider or module names.

Durable UI conventions include explicit workflow steps, compact progress/status metrics, primary actions for process/run controls, secondary reset controls, and confirmation/review controls that preserve analyst authority. Material icons are supplied through Streamlit page/button conventions where used.

## State, Reset, and Checkpoint Boundaries

Workflow state is session-scoped. Setup/configuration state is intentionally distinct from processing/result state where the workflow supports reruns. Resets should preserve the configured job when labeled as processing resets and clear the job only through the workflow's setup/reconfiguration path.

Large AI workflows can emit checkpoint reminders using `utils/ai_checkpoints.py`. Checkpoints serialize compatible state and avoid generated binary exports. Snapshot loading restores dataframes and date columns, canonicalizes legacy tag definitions, and retains compatibility paths for legacy saved data such as `TOPMERGE`-based Top Stories records.

## Testing and Verification

Tests are organized by behavior in `tests/`, including grouping, Analysis Context, legacy Sentiment/Tagging state, second-opinion batching, Top Stories validation, Authors, Rapid request/parsing/resolution/insights, report-copy coexistence, and export behavior.

`tests/fixtures/agility/` contains the canonical durable Agility fixture family for the phased automated-testing build-out. `agility_golden_corpus.csv` is the primary normal-upload corpus. `agility_malformed_inputs.csv` is the focused bad-input/robustness corpus. `agility_golden_cleaned_workbook.xlsx` is generated from the golden corpus through the app's real normalization, Basic Cleaning, grouping, and clean-workbook export path. `agility_golden_multisheet_upload.xlsx` exercises the existing worksheet-selection path, with `Agility Export` as the intended data sheet. `MANIFEST.md` owns stable case IDs and fixture rationale without adding test-only IDs to production-facing data.

`tests/test_agility_fixture_spine.py` provides fixture-driven headless regression coverage for the deterministic workflow spine. The golden fixture verifies upload normalization, Basic Cleaning reconciliation, duplicate routing, semantic story grouping, Prime Example invariants, and 44 unique traditional stories. Clean workbook generation is verified semantically rather than by binary equality, and cleaned workbook sheets are validated as acceptable headless re-entry inputs. Multi-sheet workbook structure and discovery are covered headlessly, while actual Streamlit worksheet-selection interaction remains an E2E responsibility. Malformed-input coverage verifies warning classification and that deterministic processing terminates without crash or hang. The current full `tests/` suite passes at 233 tests.

Current milestone verification (September 2026):

- `python -m unittest discover -s tests`: 203 passing tests.
- `PYTHONPATH=. pytest tests`: 203 passing tests.
- `PYTHONPATH=visual_checker_handoff/drop_in_visual_checker pytest visual_checker_handoff/drop_in_visual_checker/tests`: 4 passing tests.
- `python -m compileall` over app source, pages, processing, UI, utilities, tests, and the visual-checker handoff: passed.
- `git diff --check`: passed.
- Headless Streamlit startup: passed.

Raw `pytest` currently needs the project root on `PYTHONPATH`; without it, test collection cannot import `processing` and `utils`. This is a runner/configuration limitation, not a known product defect.

## Integrity and Security Boundaries

- Canonical Group ID is the sole story-family join key for downstream grouped results.
- No cross-group label leakage is acceptable.
- Representative-source rotation must stay within the saved story family.
- Human assignments override machine values only within their documented formulation boundaries.
- Processing resets are non-destructive to the prepared configuration unless explicitly described as full setup/reset behavior.
- Final exports derive current final/effective values fresh where appropriate.
- Unresolved labels are preserved as unresolved, not silently reinterpreted.
- Request-level raw/debug/cost information is separated from default analyst-facing row-level exports.

## Known Limitations and Open Decisions

- Regions still uses `consolidate_top_story_candidates()` for regional top-story context. Removing that remaining secondary-consolidation dependency is a follow-up.
- Established Sentiment/Tagging and Rapid Labeling coexist. No replacement/deprecation decision has been made.
- Rapid provides comparative 3-way, 5-way, score, relevance, mixture, and optional tagging evidence. It does not yet choose a preferred formulation, add deterministic reconciliation, or replace established labeling.
- Social data is cleaned and exported but does not yet have a dedicated social-analysis workflow.
- The roadmap contains future product candidates; this document describes current implementation rather than committing the app to those ideas.
