from __future__ import annotations

from pathlib import Path
import runpy


legacy_pages_dir = Path(__file__).resolve().parent.parent / "legacy_pages"
runpy.run_path(str(legacy_pages_dir / "7-Summaries.py"), run_name="__main__")
