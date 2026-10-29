"""Original token extraction for exact differential checks and benchmarks."""
from __future__ import annotations

import importlib.util
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from processing import standard_cleaning, story_grouping


_spec = importlib.util.spec_from_file_location(
    "bounded_tokens_reference",
    Path(__file__).parent / "fixtures" / "bounded_tokens_reference.py",
)
reference = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(reference)


@contextmanager
def original_token_extraction():
    with (
        patch.object(standard_cleaning, "_leading_token_tuple", reference._leading_token_tuple),
        patch.object(story_grouping, "cluster_similar_stories", reference.cluster_similar_stories),
    ):
        yield
