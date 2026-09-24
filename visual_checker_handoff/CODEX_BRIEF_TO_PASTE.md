# Codex Brief: Visual Relevance Checker Evolution

I have a standalone Streamlit app that checks whether article URLs contain relevant event/editorial visuals. I do not want this integrated into MIG_MasterApp yet, but I do want this project to evolve in a similar style: multipage Streamlit navigation, a clean app shell, session-state helpers, processing modules, and a download page.

The current standalone app does these things:

- Uploads an `.xlsx` workbook.
- Lets the user choose the sheet and URL column.
- Fetches each URL, collects candidate images from Open Graph/Twitter metadata and inline `<img>` tags.
- Filters obvious junk images.
- Sends candidate images to the OpenAI Responses API with a configurable visual relevance prompt.
- Caches URL-level results in `visual_cache.json`.
- Supports normal batches, retryable error batches, and rechecking `NO` / `NO IMAGE` results.
- Tracks estimated API usage/cost for the current session.
- Writes output columns back into the workbook and downloads an enriched `.xlsx`.

The intended workflow is now:

1. **Upload page**
   - Similar spirit to MIG_MasterApp Getting Started.
   - Upload workbook.
   - Store raw workbook bytes, filename, workbook hash, and sheet names in `st.session_state`.
   - Show basic workbook facts.
   - Include a start-over/reset action.

2. **Setup workflow**
   - This can be one page initially, or later split into sub-pages.
   - User selects sheet and maps source columns, especially URL column.
   - User defines image relevance/tagging criteria.
   - Move the current sidebar processing parameters here:
     - OpenAI API key
     - model
     - batch size
     - retry batch size
     - workers / parallel fetches
     - request timeout
     - straggler timeout
     - max images per URL
     - cache path/name if useful
   - The prompt should be configurable from user-entered criteria, not hard-coded only for PARAISO/Miami Swim Week.
   - Store setup in session state as a dict/dataclass-like structure so Run, QA, and Download read one source of truth.

3. **Run batches page**
   - Show row/URL metrics.
   - Run next uncached batch.
   - Retry transient failures.
   - Recheck negatives.
   - Show a result preview table.
   - Track session usage/cost here or in a restrained sidebar meter.

4. **QA page**
   - Placeholder for now.
   - Future intent: review YES/NO examples, inspect checked image URLs, correct verdicts, maybe sample rows by verdict/source/reason.

5. **Download page**
   - Similar spirit to MIG_MasterApp Download.
   - Build enriched workbook only for this workflow.
   - Download enriched workbook.
   - Show build timestamp and concise output summary.

Style and architecture preferences:

- Keep this as a standalone project for now.
- Use Streamlit multipage navigation via `st.Page` / `st.navigation`, like MIG_MasterApp.
- Keep pages thin and put reusable logic in `processing/`, `utils/`, and `ui/`.
- Avoid putting workflow controls in the sidebar, except maybe branding and a compact session cost meter.
- Keep cache and result-writing behavior compatible with the existing standalone app where practical.
- Preserve the successful parts of the current standalone app: candidate collection, retry logic, timeout handling, usage/cost tracking, and workbook enrichment.
- Use clear page gates: upload required before setup/run/download; setup required before run/download.
- Build incrementally. First target is functional parity with the old standalone app, just reorganized into pages.

I have provided a `drop_in_visual_checker/` scaffold with:

- `app.py`
- `ui/app_shell.py`
- `utils/session.py`
- `utils/time_display.py`
- `processing/visual_relevance.py`
- Streamlit pages for Upload, Setup, Run Batches, QA, and Download
- A small test file for stable utility behavior

Please use it as a starting point, compare it against the current standalone `app.py`, and continue developing in the existing project.
