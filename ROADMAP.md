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

### Author / Outlet Reconciliation
- Consider a future reconciliation layer between:
  - author-assigned outlet labels
  - outlet reporting-name rollups
- Open question:
  - should author outlet labels remain independent analyst-facing labels
  - or should they optionally align to the outlet reporting-name layer for consistency across workflows and exports
- Important caution:
  - do not tightly couple the Author workflow to Outlet cleanup too early unless the analyst benefit is clear
- Safer future directions may include:
  - a derived mapped-outlet field for authors
  - or a lightweight later-step reconciliation / review pass rather than forcing a workflow reordering

### Geography Insights Module
- Consider a future geography insights workflow or module that helps analysts understand where coverage is concentrated by:
  - country
  - province / state
  - city / market
  - media type / outlet mix by geography
- This depends on stronger outlet geography metadata upstream.
- A `Pubs Check` workflow that fixes incorrect outlet region metadata would likely be an important prerequisite.
- Potential future outputs:
  - market-level coverage tables
  - geography distribution charts
  - narrative observations about where visibility is strongest or weakest
  - region-aware filtering for later insight workflows

### Regions Module
- Consider a dedicated `Regions` workflow for turning reviewed outlet geography into reporting-ready regional analysis.
- This should go beyond raw tables and support the kinds of outputs often used in client decks:
  - top cities
  - top states / provinces / regions
  - regional distribution maps
  - concise narrative observations about what is driving the geographic mix

- Likely dependencies:
  - cleaned coverage rows from Basic Cleaning
  - improved outlet geography from a future `Pubs Check` workflow
  - reliable city / state / province / country normalization
  - optional market reference tables for DMA / metro / region rollups

- Suggested workflow shape:
  - prepare / validate regional fields
  - review top cities
  - review top states / provinces / regions
  - generate regional observations
  - export report-ready tables / copy

- Key analytical views:
  - `Top Cities`
    - ranked by mentions by default
    - optional toggles for impressions / effective reach
    - likely should distinguish between:
      - local-market reporting
      - syndicated pickup
      - national outlets datelined to a city
  - `Top States / Provinces / Regions`
    - ranked view plus map
    - optional rollups for:
      - U.S. states
      - Canadian provinces
      - broader custom regions if needed
  - `Coverage Distribution Map`
    - choropleth or similar regional fill map
    - useful for quickly spotting where coverage is concentrated

- Narrative outputs / observations:
  - generate concise text explaining:
    - where coverage is most concentrated
    - whether leading regions are driven by one dominant storyline or a broader mix
    - where visibility seems local vs syndicated vs nationally amplified
    - whether a region appears because of headquarters/home-market relevance, major events, or broad pickup
  - possible structures:
    - `Overall regional observation`
    - `Top cities observation`
    - `Top states / provinces observation`

- Important methodological choices to think through:
  - whether geography should be attributed from:
    - outlet location
    - dateline / story location
    - both, for different kinds of analysis
  - whether syndicated stories should count fully in every local outlet market or be flagged separately
  - how to handle national outlets with weak or missing regional metadata
  - whether maps and rankings should default to mentions or allow quick metric switching

- Useful downstream integrations:
  - include regional tables and observations in Download
  - optionally feed regional context into:
    - author insights
    - outlet insights
    - top stories
  - allow region-based filtering for later workflows
