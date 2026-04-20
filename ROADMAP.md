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
- Good first-pass heuristic:
  - deprioritize stories where the client or brand is clearly present in the headline or first 125 words of the snippet
  - prioritize stories where the client appears only later, only in metadata-like contexts, or in patterns associated with weak relevance
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
