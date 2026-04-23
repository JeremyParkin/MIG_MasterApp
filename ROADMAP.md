# Roadmap

This file is a lightweight parking lot for product and workflow ideas that are worth revisiting later.

## Near-Term Candidates

### Pubs Check Workflow
- Add a dedicated `Pubs Check` workflow for validating and enriching coverage exports before final reporting.
- This likely belongs earlier in the workflow, before downstream ranking and insight generation, because correcting impression values can materially change:
  - effective reach
  - sorting
  - top authors / outlets
  - top stories and other later-stage calculations
- Core purpose:
  - detect and fix incorrect outlet geography, especially for broadcast
  - enrich TV and Radio coverage with audience reach and AVE using reference data
  - enrich Print coverage using circulation-first logic, with online UMV as a fallback
  - queue ambiguous matches for analyst review instead of forcing low-confidence automation
  - maintain a reusable outlet reference layer that improves over time
- Proposed core flow:
  - upload coverage export
  - map source columns such as outlet, media type, country, state/province, and city
  - review and apply regional fixes for broadcast outlets
  - enrich broadcast metrics for TV and Radio
  - enrich print metrics for Print outlets
  - export the cleaned dataset
- Broadcast enrichment goals:
  - identify unique outlets from the uploaded dataset
  - match them against a maintained reference database
  - reuse existing non-stale reach and AVE values where available
  - refresh stale rows when needed
  - use AI-assisted classification only when required for missing outlets
  - merge resolved `Audience_Reach` and `AVE` back onto the coverage rows
- Print enrichment goals:
  - reuse existing non-stale print reference rows first
  - prefer direct/source-backed circulation values where available
  - if missing, try to find the corresponding online outlet of the same name
  - use online UMV as the fallback basis for print estimates
  - calculate `Audience_Reach` as best circulation value
  - calculate baseline `AVE` from UMV using print-specific formulas
  - send low-confidence or ambiguous matches to analyst review
- Matching strategy:
  - auto-accept strong matches
  - queue borderline matches for analyst confirmation
  - avoid silently applying weak outlet matches
  - capture source/method metadata so analysts can understand how a value was derived
- Outlet impression consistency:
  - detect cases where rows with the same outlet name, domain, and media type still carry inconsistent impression values
  - flag these for analyst review or normalize them when a strong canonical value can be inferred
  - treat this as a data hygiene and reference-data problem, not just a downstream reporting quirk
  - likely surface the issue earlier in the workflow, since inconsistent outlet impressions can distort:
    - effective reach
    - outlet ranking
    - author ranking
    - top stories scoring
  - if normalization is allowed, preserve the original exported impression values for traceability
- Reference data layer should support:
  - broadcast outlets
  - cities / population data
  - regional correction rules
  - print outlet metrics
  - updateable reference records so each run improves future runs

### Top Stories / Example Link Validation
- Expand the new Top Stories validation step based on analyst usage.
- Keep improving source rotation so saved stories can quickly swap to the next-best source when a link is weak, dead, or low quality.
- Consider smarter source ranking for story families, favoring stronger canonical sources over thin syndication pages.

### Cross-Workflow Link Review
- Consider a shared late-stage `Link Review` step near the end of the workflow or before Download.
- Scope would include linked example stories surfaced in:
  - Top Stories
  - Authors
  - Outlets
  - Sentiment
  - Tagging
- Goal: streamline analyst QA of linked examples before client-facing use, without requiring them to manually hunt through every workflow page.

### Media Type Module
- Consider a dedicated `Media Types` module that treats type mix as its own analytical lens rather than only mentioning it incidentally in other workflows.
- Goal:
  - help analysts quickly understand how coverage is distributed across `ONLINE`, `PRINT`, `TV`, `RADIO`, social platforms, and any other relevant media types
  - provide a cleaner place for media-type breakdowns than scattering those observations across Regions, Outlets, or general narrative copy
- Suggested outputs:
  - count and share by media type
  - mentions and share of mentions by media type
  - impressions and effective reach by media type
  - one or more representative example stories per media type
  - AI-generated observations about media-type mix, concentration, and notable differences in story character or outlet mix by type
- Possible workflow shape:
  - overview chart / table with `#` and `%` breakdowns
  - expandable detail by media type
  - example-linked grouped stories for each major type
  - report-ready narrative blurbs similar to Regions / Authors / Outlets
- Important guardrails:
  - avoid over-explaining media type when the dataset is effectively single-channel
  - respect `Analysis Context > Analysis Focus` guidance about whether media-type commentary should be emphasized or de-emphasized
  - keep the module useful both for mixed-media accounts and simpler online-only accounts

### Prominence Normalization
- Consider an early workflow step that normalizes `Prominence...` columns across grouped or syndicated copies of the same story.
- Problem:
  - prominence labels such as `Very High`, `High`, or `Moderate` are often based on whatever snippet text is available in the export
  - syndicated copies of the same article can therefore get inconsistent prominence labels when one row has fuller text and another is artificially truncated
  - this weakens the usefulness of prominence fields for later ranking, QA, and reporting
- Core idea:
  - use the app's story grouping logic to identify rows that appear to represent the same underlying story
  - for each `Prominence...` column, propagate the strongest prominence label found within the grouped story family to the other matched rows
- Important dependency:
  - this only works well if the story grouping is trustworthy enough to represent syndicated variants safely
  - if the stronger headline/date grouping used in Top Stories proves more reliable, consider whether some version of that grouping should move earlier in the workflow
- Likely normalization rule:
  - define a clear rank order such as:
    - `Very High`
    - `High`
    - `Moderate`
    - `Low`
    - `Very Low`
  - apply the highest valid value in the group to all grouped rows for that prominence column
- Possible workflow shapes:
  - simple option:
    - add an optional checkbox in `Basic Cleaning`, such as `Normalize prominence across grouped story variants`
  - analyst-review option:
    - add a lightweight review queue or wizard
    - show one representative story per grouped family rather than every row
    - let the analyst confirm whether the grouping is safe before propagating prominence values
- Suggested first version:
  - start with an optional normalization checkbox for speed and simplicity
  - add a review-based workflow later only if analysts are uneasy about silent propagation
- Important guardrails:
  - preserve the original exported prominence values somewhere for traceability
  - only normalize within high-confidence grouped families
  - do not invent prominence when no valid value exists in the group
  - be explicit when multiple `Prominence...` columns are present and normalize each independently

### Prominence Signal Controls In Analysis Context
- If `Prominence...` columns exist in the dataset, consider adding an `Analysis Context` control that lets the analyst choose which of those columns should count as meaningful signal for this analysis.
- Goal:
  - let the analyst say which prominence dimensions actually matter for this account or report
  - avoid treating every available prominence column as equally important when recommending top stories or prioritizing qualitative review
- Suggested first version:
  - detect available `Prominence...` columns automatically
  - show a multiselect in `Analysis Context` for `Which prominence columns should count as meaningful signal?`
  - optionally add a threshold control later for minimum signal level such as `High / Very High` vs `Moderate and above`
- Likely downstream uses:
  - `Recommend Top Stories`
  - qualitative prioritization and review queues
  - possible later weighting in Authors / Outlets / Regions if analysts find it valuable
- Important guardrails:
  - do not silently change row data; this is a weighting / signal interpretation choice, not normalization
  - keep it optional when no prominence fields exist or when prominence is not useful for the account

## Later Exploration

### Automated / Assisted Link Validation
- Explore whether the app can help validate current example links automatically from a cloud-hosted environment.
- Start conservatively:
  - lightweight URL checks
  - optional validation only
  - fallback to analyst review when verification is uncertain
- Keep in mind that publisher bot protection, redirects, and removed pages may make fully automated validation noisy or brittle.

### Automated Pre-Review for Link QA
- Long-term goal: reduce human review burden by automatically surfacing only the least certain or weakest example links.
- Possible model:
  - app pre-reviews example links and source candidates
  - stronger / more confident examples pass through automatically
  - human analyst reviews only the least certain, weakest, or failed link cases
- This could apply first to Top Stories, then later to Authors / Outlets / Sentiment / Tagging if the pattern proves useful.

### Shared Final QA Layer
- Consider a broader final QA step before Download that helps analysts review:
  - example links
  - representative source choice
  - possibly other output-readiness checks
- This should stay lightweight and optional unless real analyst usage proves it saves time.

### Tagging Validation Layer
- Consider adding a validation step for `Tagging` similar to the existing `Sentiment > Spot Checks` pattern.
- Core idea:
  - before final tagging insights are treated as output-ready, run a lightweight QA layer that combines:
    - AI pre-review or second-opinion checks
    - human spot checks through a review interface
- Goals:
  - catch obvious mis-tags before they shape the narrative
  - surface uncertain or low-confidence tag assignments for review
  - build more analyst trust in the final tag summaries and charts
- Possible workflow shape:
  - AI pre-reviews generated tags and flags rows or story groups that look questionable
  - analyst reviews a bounded sample rather than every tagged row
  - analyst can confirm, correct, or override tags before final insights are generated
- Likely first-pass design:
  - sample one item from large or important tag clusters
  - add a random spot-check sample across the tagged set
  - prioritize:
    - rows with conflicting evidence
    - rows where AI tagging rationale looks weak
    - rows where multiple likely tags compete
- Important guardrails:
  - this should improve trust without turning Tagging into a full manual moderation workflow
  - AI pre-review should help prioritize what the analyst sees, not silently rewrite tags
  - corrections should feed the final tagging insights and exported outputs cleanly

### Two-Pass Narrative Editing For Authors / Outlets
- Consider moving Author and Outlet insight generation to a two-pass model:
  - first pass: generate richer, evidence-grounded per-author / per-outlet summaries
  - second pass: send the whole set through an editorial rewrite pass
- Goal:
  - preserve factual substance and distinctiveness
  - reduce repeated sentence patterns across the full set
  - tighten entries for report use without losing important detail
- Key idea:
  - the first pass should optimize for fidelity to the sampled stories
  - the second pass should optimize for cross-entry variety, brevity, and polish
- Important guardrails:
  - keep the first-pass text available for QA / fallback
  - do not let the editor pass add new claims
  - keep the editor pass in English only
  - preserve differences between entries instead of smoothing them into one voice

### Small AI Helpers For Analyst Workflows
- Explore more narrow, structured AI helpers modeled on the new `Analysis Context` flow:
  - bounded task
  - structured JSON output
  - human-visible rationale
  - editable results rather than silent automation
- Most promising near-term candidates:
  - `Authors > Selection`
    - suggest a top-10 author shortlist for review
  - `Outlets > Selection`
    - suggest a top-10 outlet shortlist for review
  - `Missing Authors`
    - suggest the cleanest likely author name when source variants are messy
  - `Outlet Cleanup`
    - suggest a reporting name and / or flag why a merge looks risky
- Key guardrails:
  - keep the analyst in control
  - avoid hidden row-level mutation
  - prefer acceptance / editing over auto-apply
  - show the AI reasoning when it helps build trust

### Top Stories Recommendation Engine
- Consider a dedicated recommendation layer for `Top Stories` that ranks the strongest candidate stories before analyst review.
- Goal:
  - surface stories that are both highly visible and strongly focused on the client, instead of relying on raw mentions or impressions alone
- Suggested signal mix:
  - visibility signals:
    - mentions
    - impressions
    - effective reach
    - evidence of syndication / broad pickup
  - relevance signals:
    - brand or client mention in headline
    - brand or client mention early in the snippet / text
    - overall prominence of the client within the story
  - qualitative signal:
    - AI second opinion on whether the story is truly about the client and how central the client is to the narrative
- Possible first-pass workflow:
  - start from a candidate pool such as:
    - top 20 stories by mentions
    - top 20 stories by impressions
    - top 20 stories by effective reach
  - combine and deduplicate that pool
  - score stories using a weighted blend of visibility + relevance signals
  - let AI provide a bounded second-pass ranking or tie-break on story focus
  - present the analyst with a recommended shortlist rather than just a raw metric sort
- Important guardrails:
  - keep the scoring transparent enough that analysts can understand why a story ranked highly
  - avoid over-weighting syndication alone when the story is only weakly about the client
  - keep analyst override easy

### Executive Summary / Overall Themes
- Consider a dedicated high-level narrative module that generates an overarching summary of the coverage set, focused on themes, trends, patterns, and the strongest story arcs across the reporting period.
- This could live as:
  - its own standalone module
  - or a late-stage section adjacent to `Top Stories`
- Core idea:
  - assemble a large but bounded payload of the most important grouped stories
  - send that to an LLM with entity context, analysis guidance, and key metrics
  - ask for output such as:
    - executive summary
    - overarching themes
    - recurring product / issue / brand patterns
    - notable momentum shifts or storyline changes
    - coverage-shape observations such as concentration, spread, or major narrative tensions
- Suggested first-pass source set:
  - top ~100 unique grouped stories
  - include fields like:
    - headline
    - date
    - grouped snippet / example text
    - mentions
    - impressions
    - effective reach
    - maybe outlet / media-type context where useful
- Why this could be valuable:
  - lets the app reason across the full important coverage set, not just a handful of saved top stories
  - gives analysts a strong first-draft narrative backbone for reports
  - can surface broad themes that may not be obvious from any single module alone
- Important guardrails:
  - keep the source set bounded enough to stay performant and interpretable
  - make it clear what story pool the summary was based on
  - preserve evidence grounding so the model does not drift into unsupported generalities
  - let analysts edit or regenerate rather than treating the output as final by default

### Client Relevance Review
- Consider a lightweight early-stage QA module that checks whether coverage is actually about the client, rather than just mentioning the client in passing.
- Core use cases:
  - employment-history mentions
  - event listings
  - directory-style pages
  - passing references in broader roundups
  - pages that look more like ads, listings, or weakly related postings than real coverage about the client
- Suggested workflow shape:
  - build a suspect-relevance pool using simple heuristics
  - sample one item from the largest story groups
  - add a random sample of unique stories from the suspect pool
  - let the analyst confirm or uncheck AI-flagged `Not relevant` items
  - surface patterns that may warrant broader follow-up searching in the dataset
- Useful first-pass classification labels to test:
  - `Press Release`
  - `Advertisement / Sponsored Content`
  - `Event Listing / Calendar Notice`
  - `Stock Market Update`
  - `Job Posting`
  - `Incidental Bio Mention`
  - `Legitimate News`
- These labels are useful not because every row needs a final permanent category, but because they help separate:
  - clearly non-editorial or low-value coverage
  - weak / passing brand mentions
  - genuinely relevant client coverage
- Good first-pass heuristic:
  - deprioritize stories where the client or brand is clearly present in the headline or first 125 words of the snippet
  - prioritize stories where the client appears only later, only in metadata-like contexts, or in patterns associated with weak relevance
- Good examples of low-relevance patterns to catch:
  - the client appears only in a person's employment history or biography
  - the client is mentioned only as an event venue, sponsor, or participant in a listing
  - the client appears in promotional, recruitment, or stock-price boilerplate without meaningful editorial discussion
- Important guardrails:
  - this should stay sampled and lightweight, not become a row-by-row moderation queue
  - AI should surface likely non-relevant items, but the analyst should be able to check or uncheck them before any broader exclusion logic is applied
  - findings from this review may justify targeted searching for similar items elsewhere in the coverage

### Downloadable HTML Client Report
- Consider a late-stage module that assembles the app's cleaned data, charts, and AI-generated insights into a polished HTML report.
- Goals:
  - visual, client-ready output without requiring PowerPoint as the only presentation layer
  - downloadable `.html` file that analysts can send directly to clients
  - optional in-app preview, potentially embedded in an iframe-like viewer
  - per-section or tabbed navigation so each report "page" stays focused and readable
- Likely ingredients:
  - reusable charts and tables from Top Stories, Authors, Outlets, Regions, Tagging, and Sentiment
  - narrative blocks already curated in the workflows
  - branding / theme controls
  - optional interactive elements such as section tabs, expandable evidence, or metric toggles
- Important caution:
  - the first version should prioritize clean export and stable layout over highly custom interactivity
  - analysts still need control over what sections are included before generating the final HTML
    - top stories
  - allow region-based filtering for later workflows
