"""Deterministic datasets and a frozen oracle for deduplication verification."""
from __future__ import annotations

import importlib.util
import random
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal


FIXTURES = Path(__file__).parent / "fixtures"
_spec = importlib.util.spec_from_file_location(
    "non_broadcast_dedupe_reference", FIXTURES / "non_broadcast_dedupe_reference.py"
)
reference = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(reference)


def assert_identical_results(original, optimized, frame: pd.DataFrame) -> None:
    before = frame.copy(deep=True)
    expected = original(frame)
    assert_frame_equal(frame, before, check_exact=True)
    actual = optimized(frame)
    for expected_frame, actual_frame in zip(expected, actual, strict=True):
        assert_frame_equal(actual_frame, expected_frame, check_exact=True)
    assert_frame_equal(frame, before, check_exact=True)


def make_dataset(distribution: str, rows: int, *, seed: int = 123, width: int = 40) -> pd.DataFrame:
    """Model different duplicate-group shapes with wide, text-heavy coverage rows."""
    rng = random.Random(seed)
    base = pd.Timestamp("2026-10-01 12:00:00")
    if distribution == "unique":
        groups = list(range(rows))
    elif distribution == "pairs":
        groups = [i // 2 for i in range(rows)]
    elif distribution == "small_groups":
        groups = [i // 10 for i in range(rows)]
    elif distribution == "mixed":
        singles = int(rows * 0.70)
        pair_end = singles + int(rows * 0.20)
        groups = [i if i < singles else singles + (i - singles) // 2 if i < pair_end
                  else rows + (i - pair_end) // 25 for i in range(rows)]
    elif distribution in {"dense", "spread_dates", "undated", "fuzzy"}:
        groups = [0] * rows
    else:
        raise ValueError(distribution)

    dates = [base] * rows
    if distribution == "spread_dates":
        dates = [base + pd.Timedelta(hours=i * 72) for i in range(rows)]
    elif distribution in {"undated", "fuzzy"}:
        dates = [pd.NaT] * rows
    snippets = [f"Coverage of the client organization and its latest announcement number {i}." for i in range(rows)]
    if distribution == "fuzzy":
        snippets = [f"The client organization announced new services in the community today. Update {i % 11}." for i in range(rows)]
    frame = pd.DataFrame({
        "Type": ["ONLINE"] * rows,
        "Outlet": ["Synthetic outlet"] * rows,
        "Headline": [f"Coverage story {group}" for group in groups],
        "Snippet": snippets,
        "Date": pd.Series(dates, dtype="datetime64[ns]"),
        "Author": [rng.choice(["", "Reporter A", "Reporter B"]) for _ in range(rows)],
        "Impressions": [rng.randrange(100_000) for _ in range(rows)],
        "URL": [f"https://example.invalid/article/{i}" for i in range(rows)],
    })
    for i in range(max(0, width - len(frame.columns))):
        frame[f"Source field {i:02d}"] = [f"Additional coverage context {i}: {j % 17}" for j in range(rows)]
    return frame


def randomized_dataset(seed: int) -> pd.DataFrame:
    rng = random.Random(seed)
    rows = rng.randrange(2, 140)
    frame = make_dataset("dense", rows, seed=seed, width=rng.choice([8, 16, 40, 55]))
    group_count = rng.choice([1, 1, 2, 5, rows])
    frame["Headline"] = [f"Story {rng.randrange(group_count)}" for _ in range(rows)]
    frame["Outlet"] = [rng.choice(["Outlet A", "Outlet A", "Outlet B", "", None]) for _ in range(rows)]
    frame["Type"] = [rng.choice(["ONLINE", "ONLINE", "PRINT", "", None]) for _ in range(rows)]
    frame["Author"] = [rng.choice(["", "Same reporter", None, np.nan, pd.NA]) for _ in range(rows)]
    frame["Impressions"] = [rng.choice([0, 100, 100, None, np.nan]) for _ in range(rows)]
    frame["Date"] = [pd.NaT if rng.randrange(3) == 0 else pd.Timestamp("2026-10-01")
                     + pd.Timedelta(hours=rng.choice([0, 24, 48, 49, 72, 120])) for _ in range(rows)]
    snippets = ["Shared leading tokens in a long coverage snippet with a varying ending " + ending
                for ending in ["one", "two", "three", "four"]]
    frame["Snippet"] = [rng.choice(snippets + ["", None, np.nan]) for _ in range(rows)]
    frame.index = pd.Index(rng.sample(range(-100_000, 100_000), rows), name="source_row")
    if seed % 3 == 0:
        frame = frame.iloc[rng.sample(range(rows), rows)]
    if seed % 5 == 0:
        frame["Impressions"] = frame["Impressions"].astype("Int64")
        frame["Author"] = frame["Author"].astype("string")
    if seed % 7 == 0:
        frame["Type"] = frame["Type"].astype("category")
        frame["Outlet"] = frame["Outlet"].astype("category")
        for column in ["Type", "Outlet"]:
            if "" not in frame[column].cat.categories:
                frame[column] = frame[column].cat.add_categories([""])
    return frame
