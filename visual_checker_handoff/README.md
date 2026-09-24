# Visual Checker Handoff

This folder is a handoff package for the standalone visual relevance checker. It is intentionally not integrated into MIG_MasterApp.

Contents:

- `CODEX_BRIEF_TO_PASTE.md`: paste this into the other Codex thread so it understands the evolving intent.
- `drop_in_visual_checker/`: a starter multipage Streamlit scaffold you can copy into the standalone project.

Suggested use:

1. Copy the contents of `drop_in_visual_checker/` into the other PyCharm project.
2. Keep the original standalone `app.py` nearby for comparison.
3. Paste `CODEX_BRIEF_TO_PASTE.md` into that project’s Codex chat.
4. Ask that thread to reconcile the scaffold with the current app and preserve functional parity.

The scaffold is designed to mirror the MIG_MasterApp style at a high level: thin pages, app shell navigation, utility/session modules, processing code separated from UI, and a download page.
