from __future__ import annotations

from typing import Any

import streamlit as st


def set_page_help_context(session_state, page: str, step: str | None = None) -> None:
    session_state.page_help_page = str(page or "").strip()
    session_state.page_help_step = str(step or "").strip()


def _page_help_content() -> dict[tuple[str, str], dict[str, Any]]:
    return {
        ("Analysis Context", ""): {
            "title": "Analysis Context",
            "intro": "Use this page to shape how downstream AI workflows interpret the coverage and what portion of the cleaned dataset stays in scope.",
            "sections": [
                {
                    "heading": "What this page does",
                    "bullets": [
                        "Sets shared entity context such as primary topic, aliases, spokespeople, products, and extra analytical guidance.",
                        "Controls what will be excluded by qualitative insight workflows like Top Stories, Authors, Outlets, Regions, Tagging, and Sentiment.",
                        "Defines Data Scope rules such as date range, media types, and dataset-level pruning with row-level keep overrides.",
                    ],
                },
                {
                    "heading": "What the app does automatically",
                    "bullets": [
                        "Carries the saved context into downstream AI prompts and report-copy generators.",
                        "Builds removal previews for both qualitative exclusions and dataset-wide scope changes before you save.",
                        "Keeps highlight-only keywords separate so they help Spot Checks without changing AI prompt context.",
                    ],
                },
                {
                    "heading": "What to review manually",
                    "bullets": [
                        "Make sure aliases, spokespeople, and product names are actually useful for downstream interpretation.",
                        "Double-check that junky-coverage exclusions are aligned with account priorities, especially if you want to keep some flagged rows.",
                        "Review the removed-row preview when using Data Scope so you do not accidentally cut meaningful coverage.",
                    ],
                },
                {
                    "heading": "Key logic / heuristics",
                    "bullets": [
                        "Media-type commentary only shapes narrative emphasis. It does not filter rows by itself.",
                        "Data Scope rules change the working dataset used by downstream workflows and exports except where you explicitly keep a row.",
                        "Qualitative exclusions hide rows from narrative-oriented workflows but do not remove them from the main dataset unless also selected in Data Scope.",
                    ],
                },
            ],
        },
        ("Basic Cleaning", ""): {
            "title": "Basic Cleaning",
            "intro": "Standardize the uploaded export, split social content, remove duplicates, calculate effective reach, and group similar coverage into unique stories.",
            "sections": [
                {
                    "heading": "What this step does",
                    "bullets": [
                        "Normalizes the uploaded file into the app's working structure before the rest of the workflow begins.",
                        "Separates traditional and social coverage, removes duplicates when selected, and builds grouped-story tables for downstream modules.",
                        "Shows row reconciliation and dataset previews so you can confirm the cleaned output looks sensible.",
                    ],
                },
                {
                    "heading": "What the app does automatically",
                    "bullets": [
                        "Large datasets run in 3 parts for stability, while smaller datasets complete in one pass.",
                        "Broadcast duplicate handling uses separate, more conservative logic than ordinary URL and field-based dedupe.",
                        "Effective reach is calculated after cleaning, using different logic for traditional and social datasets.",
                    ],
                },
                {
                    "heading": "What to review manually",
                    "bullets": [
                        "Check row reconciliation first. Original rows should equal Traditional + Social + Deleted Duplicates.",
                        "Look at the dataset previews to make sure media types, social split, and duplicate removal all feel plausible.",
                        "If something looks wrong here, fix it now. Most later workflows assume Basic Cleaning is trustworthy.",
                    ],
                },
                {
                    "heading": "Key logic / heuristics",
                    "bullets": [
                        "Reset Basic Cleaning returns the app to the completed upload baseline and clears downstream workflow state.",
                        "This is the step that creates the working datasets used by almost everything else in the app.",
                    ],
                },
            ],
        },
        ("Authors", "Missing"): {
            "title": "Authors > Missing",
            "intro": "Resolve missing author names story by story, with shortcuts for obvious repeated matches.",
            "sections": [
                {
                    "heading": "What this step does",
                    "bullets": [
                        "Finds grouped stories where some rows have author names and other rows are missing them.",
                        "Lets you choose from suggested author names or write one in manually.",
                        "Updates the cleaned traditional dataset so later Author steps work from the improved names.",
                    ],
                },
                {
                    "heading": "What the app does automatically",
                    "bullets": [
                        "Gold highlighting is used only for stronger candidate names, not single one-off suggestions.",
                        "Accept obvious applies one-click fixes when a non-blank, name-like suggestion accounts for at least 80% of known author suggestions.",
                        "Undo can reverse the last author update, including a bulk obvious-match acceptance.",
                    ],
                },
                {
                    "heading": "What to review manually",
                    "bullets": [
                        "Use the headline search when a story cluster looks ambiguous or syndicated across many outlets.",
                        "Manually write in a name when the dropdown suggestions look noisy and need fixing.",
                        "Skip headlines where no known author can be confirmed.",
                        "No need to fix every headline. This queue is sorted to surface the biggest missing-author opportunities first, so there are diminishing returns as you work further down the list."
                    ],
                },
                {
                    "heading": "Key logic / heuristics",
                    "bullets": [
                        "Obvious-match detection uses only rows with known author suggestions, not the full story-row denominator.",
                        "Author suggestions with junky characters or poor name quality are not eligible for auto-acceptance.",
                    ],
                },
            ],
        },
        ("Authors", "Outlets"): {
            "title": "Authors > Outlets",
            "intro": "Assign one primary outlet to each author so the shortlist and insights reflect a clean author-to-outlet relationship.",
            "sections": [
                {
                    "heading": "What this step does",
                    "bullets": [
                        "Shows likely outlet assignments for each author based on the cleaned dataset and Agility media database.",
                        "Lets you confirm, override, or skip the primary outlet for each author.",
                        "Includes tools to correct author names when a bad or messy name string is affecting assignment.",
                    ],
                },
                {
                    "heading": "What the app does automatically",
                    "bullets": [
                        "Sorts candidate authors by Mentions, Impressions, or Effective Reach, depending on the selected ranking metric.",
                        "Tallies how often each outlet appears for an author in the cleaned dataset and compares that pattern to the Agility media database.",
                        "Provides visual cues for likely or perfect matches, and can auto-assign perfect matches when you run that action.",
                    ],
                },
                {
                    "heading": "What to review manually",
                    "bullets": [
                        "Watch for author strings that accidentally include outlet names, newsroom labels, or other non-name text.",
                        "Fields listing multiple authors usually cannot be matched cleanly here and should typically be fixed by assigning the lead author first.",
                        "You usually only need to work far enough to support the final top-author shortlist, which is often around the top 10 authors.",
                    ],
                },
                {
                    "heading": "Key logic / heuristics",
                    "bullets": [
                        "Assigned outlets are workflow-level reporting decisions, not raw-source edits to the original outlet field.",
                        "Fix name updates the cleaned dataset and refreshes the outlet-assignment workflow around the corrected author name.",
                        "Ranking metric changes inspector order, shortlist order, and insights order together.",
                        "Qualitative exclusions from Analysis Context can reduce which rows contribute to author-outlet insights.",
                    ],
                },
            ],
        },
        ("Authors", "Selection"): {
            "title": "Authors > Selection",
            "intro": "Curate the final top-author shortlist after missing names and outlet assignments have been cleaned up.",
            "sections": [
                {
                    "heading": "What this step does",
                    "bullets": [
                        "Lets you inspect ranked author candidates and save the final reporting shortlist.",
                        "Shows a current shortlist table that you can trim directly before moving to Insights.",
                    ],
                },
                {
                    "heading": "What the app does automatically",
                    "bullets": [
                        "Sorts the inspector, candidate table, current shortlist, and downstream insights order together based on the selected ranking metric.",
                        "Keeps the shortlist table in sync immediately when you remove saved authors.",
                    ],
                },
                {
                    "heading": "What to review manually",
                    "bullets": [
                        "Save authors who are meaningfully covering the client brand, not just mechanically high-volume or reach.",
                        "Use the inspector examples to separate true key voices from syndicated, low-signal, or overly fragmented names.",
                        "You usually only need a concise final shortlist of ~10, not every plausible author.",
                    ],
                },
                {
                    "heading": "Key logic / heuristics",
                    "bullets": [
                        "The shortlist is a curated output layer, so saving an author is an analyst decision even when ranking is automated.",
                        "Ranking metric changes display order only. It does not change the underlying author data.",
                    ],
                },
            ],
        },
        ("Authors", "Insights"): {
            "title": "Authors > Insights",
            "intro": "Review charts, tables, and generate AI insights for the saved top-author shortlist.",
            "sections": [
                {
                    "heading": "What this step does",
                    "bullets": [
                        "Builds the final author outputs from the saved shortlist only.",
                        "Lets you generate coverage themes and review linked grouped-story examples for each saved author.",
                    ],
                },
                {
                    "heading": "What the app does automatically",
                    "bullets": [
                        "Keeps chart, table, and report-copy order aligned with the selected ranking metric.",
                        "Uses Analysis Context guidance and qualitative exclusions when generating author narrative outputs.",
                    ],
                },
                {
                    "heading": "What to review manually",
                    "bullets": [
                        "Check that the saved shortlist is still the right one before trusting the final copy.",
                        "Use linked grouped-story examples to sanity-check the themes and make sure each author is represented by the right kinds of stories.",
                        "If the narrative feels weak, the better fix is often refining the shortlist rather than editing the wording in isolation.",
                    ],
                },
                {
                    "heading": "Key logic / heuristics",
                    "bullets": [
                        "This step only reflects saved authors. If someone is missing or overrepresented here, go back to Selection and adjust the shortlist.",
                        "Qualitative exclusions can change which rows and examples shape the author insights without changing the raw cleaned dataset.",
                    ],
                },
            ],
        },
        ("Outlets", "Cleanup"): {
            "title": "Outlets > Cleanup",
            "intro": "Clean outlet variants into sensible reporting rollups before building the final outlet shortlist.",
            "sections": [
                {
                    "heading": "What this step does",
                    "bullets": [
                        "Reviews suggested merge clusters such as station variants, network rollups, and similar outlet names.",
                        "Lets you decide which source outlets belong together and what the master reporting name should be.",
                        "Creates cleaner reporting names for this workflow without overwriting the raw outlet field in the cleaned dataset.",
                    ],
                },
                {
                    "heading": "What the app does automatically",
                    "bullets": [
                        "Suggests merge clusters using outlet-name similarity, network rollup logic, and other cleanup heuristics.",
                        "Canonical rollups like CBC, CTV, and Global keep their predefined reporting names when appropriate.",
                        "Merged source outlets drop out of future queue proposals, and explicitly dismissed candidates are remembered.",
                        "Queue position rebuilds against the unresolved list so confirmed work does not keep resurfacing.",
                    ],
                },
                {
                    "heading": "What to review manually",
                    "bullets": [
                        "Uncheck outlets that happen to share a keyword but do not belong in the same reporting rollup.",
                        "Be especially careful with mixed-country or mixed-domain clusters that only look superficially similar.",
                        "Do not feel like you need to merge every possible cluster. Focus on the outlet names that will matter for the final shortlist and insights.",
                    ],
                },
                {
                    "heading": "Key logic / heuristics",
                    "bullets": [
                        "Cleanup does not overwrite raw outlet names in the cleaned dataset. It builds a reporting rollup map used by this workflow and exports.",
                        "These rollups affect Outlets workflow outputs and exports, but they are separate from the raw source names in the cleaned data.",
                    ],
                },
            ],
        },
        ("Outlets", "Selection"): {
            "title": "Outlets > Selection",
            "intro": "Save the final outlet shortlist after cleanup so the rest of the workflow uses the curated reporting names.",
            "sections": [
                {
                    "heading": "What this step does",
                    "bullets": [
                        "Lets you inspect ranked outlet candidates and save the final shortlist.",
                        "Shows a current shortlist table that you can trim directly before moving to Insights.",
                    ],
                },
                {
                    "heading": "What the app does automatically",
                    "bullets": [
                        "Sorts the inspector, candidate table, shortlist order, and downstream insights order together based on the selected ranking metric.",
                        "Uses cleanup rollups in place of raw outlet-name variants when those have already been defined.",
                    ],
                },
                {
                    "heading": "What to review manually",
                    "bullets": [
                        "Save outlets that genuinely matter for the final story, not just every high-volume source.",
                        "Use examples and media-type mix to separate key editorial brands from thin, duplicate, or low-value variants.",
                        "You usually only need a concise final shortlist, not every plausible outlet.",
                    ],
                },
                {
                    "heading": "Key logic / heuristics",
                    "bullets": [
                        "The shortlist is a curated output layer, so saving an outlet is an analyst decision even when ranking is automated.",
                        "Ranking metric changes display order only. It does not change the underlying outlet data.",
                    ],
                },
            ],
        },
        ("Outlets", "Insights"): {
            "title": "Outlets > Insights",
            "intro": "Review charts, tables, and generate AI insights for the saved outlet shortlist.",
            "sections": [
                {
                    "heading": "What this step does",
                    "bullets": [
                        "Builds the final outlet outputs from the saved shortlist only.",
                        "Shows quantitative rankings and AI-generated outlet insights shaped by Analysis Context.",
                    ],
                },
                {
                    "heading": "What the app does automatically",
                    "bullets": [
                        "Keeps chart, table, and insight order aligned with the selected ranking metric.",
                        "Uses cleanup rollups, aggregator exclusions, and qualitative exclusions when building outlet outputs.",
                    ],
                },
                {
                    "heading": "What to review manually",
                    "bullets": [
                        "Check whether your cleanup rollups and saved shortlist are still right before trusting the final insights.",
                        "Use linked grouped-story examples to confirm the strongest outlet patterns and make sure the narrative matches the actual coverage.",
                        "If the insights feel off, the better fix is often refining cleanup or shortlist choices rather than editing the wording in isolation.",
                    ],
                },
                {
                    "heading": "Key logic / heuristics",
                    "bullets": [
                        "This step only reflects saved outlets. If an important outlet is missing or a weak one is overrepresented, go back to Selection and adjust the shortlist.",
                        "Qualitative exclusions can change which rows and examples shape the outlet insights without changing the raw cleaned dataset.",
                    ],
                },
            ],
        },
        ("Top Stories", "Selection"): {
            "title": "Top Stories > Selection",
            "intro": "Build the candidate pool, use recommendation help if needed, and save the final top-story shortlist.",
            "sections": [
                {
                    "heading": "What this step does",
                    "bullets": [
                        "Generates grouped top-story candidates from the cleaned traditional dataset.",
                        "Lets you manually check stories, use blunt metric shortcuts, or use the recommendation engine before saving the final shortlist.",
                        "Shows a current saved list that you can keep refining as you work.",
                    ],
                },
                {
                    "heading": "What the app does automatically",
                    "bullets": [
                        "Recommend Top Stories uses a local heuristic blend of visibility and relevance signals, not an API call.",
                        "The recommendation logic uses headline and snippet relevance, mentions, impressions, effective reach, and outlet or coverage-flag adjustments.",
                        "Saved stories drop out of the candidate table so repeated recommendation passes can surface the next batch.",
                    ],
                },
                {
                    "heading": "What to review manually",
                    "bullets": [
                        "Override recommendations whenever a story is high-visibility but not truly one of the best anchors for the final narrative.",
                        "Pay attention to thin snippets or sparse source data that may understate a story's relevance.",
                        "You usually want a concise final shortlist, not every plausible candidate.",
                    ],
                },
                {
                    "heading": "Key logic / heuristics",
                    "bullets": [
                        "The recommender now normalizes diacritics, so names like Ténéré and Tenere match consistently.",
                        "Recommendation quality can still be affected when a high-visibility row has almost no snippet or body text.",
                        "Saving a story is always an analyst decision, even when recommendation logic is helping with the shortlist.",
                    ],
                },
            ],
        },
        ("Top Stories", "Validation"): {
            "title": "Top Stories > Validation",
            "intro": "Review one saved story at a time, rotate weak links, and confirm the source you want to keep.",
            "sections": [
                {
                    "heading": "What this step does",
                    "bullets": [
                        "Shows one saved top story at a time with its current example link and alternate source options.",
                        "Lets you rotate to the next source when the current link is weak, dead, or low quality.",
                        "Lets you confirm a source and remove that story from the active validation queue.",
                    ],
                },
                {
                    "heading": "What the app does automatically",
                    "bullets": [
                        "Source options prefer alternatives from the same original story group first, then fall back to sibling merged groups.",
                        "If the current story has a real link, alternate candidates are restricted to candidates that also have links.",
                        "Confirm source removes that story from the active validation queue and moves you forward immediately.",
                    ],
                },
                {
                    "heading": "What to review manually",
                    "bullets": [
                        "Use Open current link to confirm the source is live and feels like a strong example for final use.",
                        "Try next source when the current page is dead, low quality, or clearly not the best representative example.",
                        "Confirm a source once you are satisfied so you do not keep seeing the same story in the active queue.",
                    ],
                },
                {
                    "heading": "Key logic / heuristics",
                    "bullets": [
                        "The validation queue rebuilds naturally when Saved Top Stories change, so new stories re-enter and removed stories drop out.",
                        "Validation is about choosing the best example link for an already-saved story, not deciding whether the story belongs in the shortlist at all.",
                    ],
                },
            ],
        },
        ("Top Stories", "Insights"): {
            "title": "Top Stories > Insights",
            "intro": "Generate and review AI insights built from the saved and validated top stories.",
            "sections": [
                {
                    "heading": "What this step does",
                    "bullets": [
                        "Uses the final saved top stories as the basis for summaries and story blocks.",
                        "Keeps the workflow centered on a curated shortlist rather than the full candidate pool.",
                    ],
                },
                {
                    "heading": "What the app does automatically",
                    "bullets": [
                        "Narrative prompts inherit Analysis Context guidance, exclusions, and data-scope settings.",
                        "Validated example links and final saved-story state flow into the final output.",
                    ],
                },
                {
                    "heading": "What to review manually",
                    "bullets": [
                        "Check that the saved shortlist is still the right one before trusting the generated insights.",
                        "If the output feels weak or oddly balanced, the better fix is often refining the shortlist or validation choices rather than editing the wording in isolation.",
                    ],
                },
                {
                    "heading": "Key logic / heuristics",
                    "bullets": [
                        "This step only reflects saved top stories. If an important story is missing or a weak one is overrepresented, go back to Selection or Validation and adjust the underlying set.",
                    ],
                },
            ],
        },
        ("Tagging", "Setup"): {
            "title": "Tagging > Setup",
            "intro": "Prepare the tagging dataset, define tags, and decide how large the tagging run should be.",
            "sections": [
                {
                    "heading": "What this step does",
                    "bullets": [
                        "Builds a tagging sample or uses the full eligible dataset, then regroups it into unique stories for tagging.",
                        "Lets you define the tag list and choose whether tagging is single-best or multiple applicable tags.",
                    ],
                },
                {
                    "heading": "What the app does automatically",
                    "bullets": [
                        "Sampling happens at the row level first, then the sampled rows are regrouped into unique stories.",
                        "Representative and reuse modes help keep cost and stability under control.",
                    ],
                },
                {
                    "heading": "What to review manually",
                    "bullets": [
                        "Make sure the tags are distinct enough that an analyst could apply them consistently.",
                        "Use a full run only when the dataset is small enough to justify the extra cost and time.",
                    ],
                },
                {
                    "heading": "Key logic / heuristics",
                    "bullets": [
                        "Tagging works from grouped stories, then cascades final tags back to sampled row-level rows.",
                    ],
                },
            ],
        },
        ("Tagging", "Run"): {
            "title": "Tagging > Run",
            "intro": "Run the first AI tagging pass across the prepared unique-story set.",
            "sections": [
                {
                    "heading": "What this step does",
                    "bullets": [
                        "Applies the first AI tagging pass across the prepared grouped-story dataset.",
                    ],
                },
                {
                    "heading": "What the app does automatically",
                    "bullets": [
                        "Writes the first AI tag, confidence, and rationale onto the grouped-story table.",
                        "Cascades final/effective tags back to the sampled row-level rows for tables and exports.",
                    ],
                },
                {
                    "heading": "What to review manually",
                    "bullets": [
                        "You do not need to judge quality here yet. The real review happens in AI Pre-Review, Spot Checks, and Insights.",
                    ],
                },
            ],
        },
        ("Tagging", "AI Pre-Review"): {
            "title": "Tagging > AI Pre-Review",
            "intro": "Run a second AI pass on top candidates so exact high-confidence matches can be auto-resolved before human review.",
            "sections": [
                {
                    "heading": "What this step does",
                    "bullets": [
                        "Runs a second AI opinion on the strongest review candidates before you do human spot checks.",
                    ],
                },
                {
                    "heading": "What the app does automatically",
                    "bullets": [
                        "Single-tag mode compares the first and second AI opinion directly.",
                        "Multi-tag mode compares normalized tag sets and only exact high-confidence matches auto-resolve.",
                        "Timestamped completion messages appear right after the batch and do not persist when you leave and return.",
                    ],
                },
                {
                    "heading": "What to review manually",
                    "bullets": [
                        "Use this step to reduce manual work, not to treat all auto-resolved tags as beyond question.",
                    ],
                },
                {
                    "heading": "Key logic / heuristics",
                    "bullets": [
                        "Auto-resolution is intentionally strict. Multi-tag mode only auto-resolves exact high-confidence set matches.",
                    ],
                },
            ],
        },
        ("Tagging", "Spot Checks"): {
            "title": "Tagging > Spot Checks",
            "intro": "Review grouped stories one at a time, compare AI opinions, and finalize tags where needed.",
            "sections": [
                {
                    "heading": "What this step does",
                    "bullets": [
                        "Lets you review flagged, disagreement, unresolved, or all tagged coverage queues.",
                        "Uses button-based final tag assignment for single-tag mode and checkboxes for multi-tag mode.",
                    ],
                },
                {
                    "heading": "What the app does automatically",
                    "bullets": [
                        "Keyword highlighting uses the same safe escaped pipeline as Sentiment and respects highlight-only keywords from Analysis Context.",
                        "All Tagged Coverage lets you audit everything AI tagged, not just the unresolved review pool.",
                    ],
                },
                {
                    "heading": "What to review manually",
                    "bullets": [
                        "Use this step when you want to correct disagreements, audit all AI-tagged coverage, or override edge cases the model handled poorly.",
                        "In multi-tag mode, check only the tags that truly apply. Do not feel pressured to preserve every AI-suggested label.",
                    ],
                },
                {
                    "heading": "Key logic / heuristics",
                    "bullets": [
                        "Single-tag mode uses one final label, while multi-tag mode treats the final assignment as a set of applicable tags.",
                    ],
                },
            ],
        },
        ("Tagging", "Insights"): {
            "title": "Tagging > Insights",
            "intro": "Review the final tag distribution and generate AI insights from the reviewed tag set.",
            "sections": [
                {
                    "heading": "What this step does",
                    "bullets": [
                        "Shows the final tag distribution and lets you generate narrative observations from the effective tags.",
                    ],
                },
                {
                    "heading": "What the app does automatically",
                    "bullets": [
                        "Uses final human-reviewed tags where present and falls back to effective AI tags elsewhere.",
                    ],
                },
                {
                    "heading": "What to review manually",
                    "bullets": [
                        "If the distribution feels off, the better fix is usually in Setup, Spot Checks, or the final tag assignments rather than the wording of the insight text.",
                    ],
                },
                {
                    "heading": "Key logic / heuristics",
                    "bullets": [
                        "In multi-tag mode, tag percentages represent the share of grouped stories carrying that tag, not share of all tag assignments.",
                    ],
                },
            ],
        },
        ("Sentiment", "Setup"): {
            "title": "Sentiment > Setup",
            "intro": "Prepare the sentiment dataset and choose the tone scheme before running AI sentiment.",
            "sections": [
                {
                    "heading": "What this step does",
                    "bullets": [
                        "Builds the sentiment sample or full eligible set, then groups it into unique stories for AI analysis.",
                        "Lets you choose 3-way or 5-way sentiment and define optional Analysis Context guidance.",
                    ],
                },
                {
                    "heading": "What the app does automatically",
                    "bullets": [
                        "Sampling happens at the row level first, then the sampled rows are regrouped into unique stories.",
                        "Representative and reuse modes help keep cost and stability under control.",
                    ],
                },
                {
                    "heading": "What to review manually",
                    "bullets": [
                        "Choose the 3-way or 5-way tone scheme that actually fits how you want to interpret this account.",
                        "Use a full run only when the dataset is small enough to justify the extra cost and time.",
                    ],
                },
            ],
        },
        ("Sentiment", "Run"): {
            "title": "Sentiment > Run",
            "intro": "Run the first AI sentiment pass across the prepared unique stories.",
            "sections": [
                {
                    "heading": "What this step does",
                    "bullets": [
                        "Applies the first AI sentiment pass across the prepared grouped-story dataset.",
                    ],
                },
                {
                    "heading": "What the app does automatically",
                    "bullets": [
                        "Writes first-pass sentiment, rationale, and confidence onto the grouped-story table.",
                        "Cascades effective sentiment back to the sampled row-level dataset used for exports and downstream tables.",
                    ],
                },
                {
                    "heading": "What to review manually",
                    "bullets": [
                        "You do not need to judge quality here yet. The real review happens in AI Pre-Review, Spot Checks, and Insights.",
                    ],
                },
            ],
        },
        ("Sentiment", "AI Pre-Review"): {
            "title": "Sentiment > AI Pre-Review",
            "intro": "Run a second AI pass on the strongest review candidates so high-confidence matches can be auto-resolved before human spot checks.",
            "sections": [
                {
                    "heading": "What this step does",
                    "bullets": [
                        "Runs a second AI opinion on the strongest review candidates before you do human spot checks.",
                    ],
                },
                {
                    "heading": "What the app does automatically",
                    "bullets": [
                        "Compares first and second AI sentiment opinions and auto-resolves exact high-confidence matches.",
                        "Uses a timestamped, non-persistent completion message after each pre-review batch.",
                    ],
                },
                {
                    "heading": "What to review manually",
                    "bullets": [
                        "Use this step to reduce manual review volume, not to assume every auto-resolved call is beyond question.",
                    ],
                },
            ],
        },
        ("Sentiment", "Spot Checks"): {
            "title": "Sentiment > Spot Checks",
            "intro": "Review one grouped story at a time, compare AI opinions, and finalize sentiment where needed.",
            "sections": [
                {
                    "heading": "What this step does",
                    "bullets": [
                        "Lets you review flagged, disagreements-only, unresolved, or all toned coverage queues.",
                        "Highlights entity-context keywords in the story text to help you focus on the relevant parts.",
                    ],
                },
                {
                    "heading": "What the app does automatically",
                    "bullets": [
                        "Keyword highlighting is rendered safely through escaped text plus injected highlight markup.",
                        "All Toned Coverage lets you audit everything the AI labeled, not just the unresolved review pool.",
                    ],
                },
                {
                    "heading": "What to review manually",
                    "bullets": [
                        "Use this step to resolve disagreements, audit all toned coverage, or override cases where the AI read the story incorrectly.",
                        "Focus on whether the final sentiment is genuinely about the client or topic in context, not just whether the story sounds positive or negative in general.",
                    ],
                },
            ],
        },
        ("Sentiment", "Insights"): {
            "title": "Sentiment > Insights",
            "intro": "Review the final sentiment distribution and generate AI insights from the reviewed sentiment set.",
            "sections": [
                {
                    "heading": "What this step does",
                    "bullets": [
                        "Shows the final sentiment distribution and lets you generate narrative observations from the effective sentiment set.",
                    ],
                },
                {
                    "heading": "What the app does automatically",
                    "bullets": [
                        "Uses final human-reviewed sentiment where present and falls back to effective AI sentiment elsewhere.",
                        "Carries Analysis Context guidance and exclusions into the narrative generation step.",
                    ],
                },
                {
                    "heading": "What to review manually",
                    "bullets": [
                        "If the distribution or themes feel off, the better fix is usually in Setup, Spot Checks, or the final sentiment decisions rather than the wording of the insight text.",
                    ],
                },
            ],
        },
        ("Regions", "Setup"): {
            "title": "Regions > Setup",
            "intro": "Review the geography first, decide which levels matter, and prepare the regional analysis view.",
            "sections": [
                {
                    "heading": "What this step does",
                    "bullets": [
                        "Shows country, region, and city previews so you can decide which geographic levels are worth analyzing.",
                        "Lets you choose the ranking metric and see whether the dataset actually has meaningful geographic spread.",
                        "Prepares the regional analysis before you move into Insights.",
                    ],
                },
                {
                    "heading": "What the app does automatically",
                    "bullets": [
                        "Builds country, region, and city rankings from the cleaned dataset using the selected ranking metric.",
                        "Applies the current Analysis Context and qualitative exclusions before building the geography views.",
                    ],
                },
                {
                    "heading": "What to review manually",
                    "bullets": [
                        "Decide whether countries, regions, cities, or only one of those levels are actually worth discussing for this dataset.",
                        "Do not feel obligated to analyze every geographic level if the spread is shallow or the story is effectively concentrated in one place.",
                    ],
                },
                {
                    "heading": "Key logic / heuristics",
                    "bullets": [
                        "This step is about deciding whether geography is analytically useful, not forcing a regional story where there is not one.",
                    ],
                },
            ],
        },
        ("Regions", "Insights"): {
            "title": "Regions > Insights",
            "intro": "Analyze countries, regions, and cities with charts, tables, and AI-generated geographic insights.",
            "sections": [
                {
                    "heading": "What this step does",
                    "bullets": [
                        "Generates observations that explain what is driving the leading places and how attention is distributed geographically.",
                        "Lets you inspect supporting grouped stories for each major place.",
                        "Shows chart, table, and report-copy views for the selected geographic level.",
                    ],
                },
                {
                    "heading": "What the app does automatically",
                    "bullets": [
                        "Analysis Context guidance and media-type commentary settings shape how the narrative is written.",
                        "Uses the selected ranking metric to determine which places lead and what examples are surfaced underneath.",
                    ],
                },
                {
                    "heading": "What to review manually",
                    "bullets": [
                        "Check that the chosen geographic level is actually producing meaningful differences, not just repeating the same storyline across places.",
                        "Use the supporting grouped stories to confirm what is really driving each major place before trusting the narrative.",
                        "If the insights feel forced, the better fix is often analyzing fewer geographic levels or changing the ranking metric.",
                    ],
                },
                {
                    "heading": "Key logic / heuristics",
                    "bullets": [
                        "The module can still be useful for single-country or single-channel datasets, but the geographic narrative should stay proportional to the real spread in the data.",
                        "Media-type commentary can be emphasized or de-emphasized through Analysis Context without changing the underlying geography data.",
                    ],
                },
            ],
        },
    }


def _get_help_entry(page: str, step: str | None = None) -> dict[str, Any] | None:
    page = str(page or "").strip()
    step = str(step or "").strip()
    content = _page_help_content()
    entry = content.get((page, step)) or content.get((page, ""))
    if entry is not None:
        return entry
    for (candidate_page, _candidate_step), candidate_entry in content.items():
        if candidate_page == page:
            return candidate_entry
    return None


@st.dialog("Page Help", width="large")
def _render_page_help_dialog(page: str, step: str | None = None) -> None:
    entry = _get_help_entry(page, step)
    if not entry:
        st.info("No page help has been written for this page yet.")
        return

    st.markdown(
        """
        <style>
        .page-help-intro {
            color: rgba(250, 250, 250, 0.82);
            margin: 0.2rem 0 1rem 0;
            font-size: 1rem;
            line-height: 1.55;
        }
        .page-help-section {
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 14px;
            padding: 0.95rem 1rem 0.8rem 1rem;
            margin: 0 0 0.9rem 0;
            background: rgba(255, 255, 255, 0.02);
        }
        .page-help-heading {
            font-weight: 700;
            margin: 0 0 0.45rem 0;
        }
        .page-help-list {
            margin: 0.15rem 0 0 0;
            padding-left: 1.1rem;
        }
        .page-help-list li {
            margin: 0 0 0.5rem 0;
            line-height: 1.5;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    title = entry.get("title")
    intro = entry.get("intro")
    sections = entry.get("sections", [])

    if title:
        st.subheader(title)
    if intro:
        st.markdown(f'<div class="page-help-intro">{intro}</div>', unsafe_allow_html=True)

    for section in sections:
        heading = str(section.get("heading", "")).strip()
        bullets = section.get("bullets", []) or []
        list_items = "".join(f"<li>{bullet}</li>" for bullet in bullets)
        st.markdown(
            (
                '<div class="page-help-section">'
                + (f'<div class="page-help-heading">{heading}</div>' if heading else "")
                + f'<ul class="page-help-list">{list_items}</ul>'
                + "</div>"
            ),
            unsafe_allow_html=True,
        )


def render_sidebar_page_help(target=None, *, key_suffix: str = "default") -> None:
    target = target or st.sidebar
    page = str(st.session_state.get("page_help_page", "") or "").strip()
    step = str(st.session_state.get("page_help_step", "") or "").strip()
    entry = _get_help_entry(page, step)

    if not page:
        return

    with target:
        label = "Page Help"
        if target.button(
            label,
            key=f"sidebar_page_help_open_{key_suffix}",
            use_container_width=True,
            disabled=entry is None,
        ):
            _render_page_help_dialog(page, step)
        if entry and step:
            st.caption(f"{page} > {step}")
