from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dedupe_support import FIXTURES, assert_identical_results, make_dataset, randomized_dataset, reference
from processing.standard_cleaning import dedupe_non_broadcast_by_fields
from utils.io import normalize_uploaded_dataframe


def compare(frame):
    assert_identical_results(reference.dedupe_non_broadcast_by_fields, dedupe_non_broadcast_by_fields, frame)


@pytest.mark.parametrize("fixture", ["agility_golden_corpus.csv", "agility_malformed_inputs.csv"])
def test_existing_fixtures_match_original_exactly(fixture):
    frame = normalize_uploaded_dataframe(pd.read_csv(FIXTURES / "agility" / fixture))
    frame = frame.loc[~frame["Type"].isin(["TV", "RADIO"])].copy()
    compare(frame)


@pytest.mark.parametrize("seed", range(160))
def test_randomized_inputs_match_original_exactly(seed):
    compare(randomized_dataset(seed))


@pytest.mark.parametrize("rows", [250, 251, 800, 1200])
@pytest.mark.parametrize("distribution", ["dense", "undated", "spread_dates"])
def test_large_groups_and_fuzzy_size_boundary_match_original(distribution, rows):
    compare(make_dataset(distribution, rows))


@pytest.mark.parametrize("distribution", ["unique", "pairs", "small_groups", "mixed", "fuzzy"])
def test_different_duplicate_distributions_match_original(distribution):
    compare(make_dataset(distribution, 100))


@pytest.mark.parametrize("column", ["Type", "Outlet", "Headline", "Date", "Snippet", "Author", "Impressions"])
def test_missing_columns_match_original(column):
    compare(make_dataset("small_groups", 20).drop(columns=[column]))


def test_empty_and_blank_group_keys_match_original():
    compare(pd.DataFrame())
    frame = make_dataset("dense", 12)
    compare(frame.iloc[:0])
    for column in ["Type", "Outlet", "Headline"]:
        frame[column] = [None, np.nan, pd.NA, "", " ", "value"] * 2
    compare(frame)


def test_missing_values_and_tied_rankings_preserve_winners_and_order():
    frame = make_dataset("dense", 32)
    frame["Author"] = [None, np.nan, pd.NA, ""] * 8
    frame["Impressions"] = [None, np.nan, 0, "bad"] * 8
    frame["Snippet"] = [None, np.nan, "", "Same coverage text"] * 8
    frame["Date"] = [pd.NaT, pd.Timestamp("2026-10-01"), pd.Timestamp("2026-10-01"), pd.NaT] * 8
    frame.index = pd.Index(range(900, 868, -1), name="original_index")
    compare(frame)
    frame["Author"] = "Reporter"
    frame["Impressions"] = 100
    frame["Date"] = pd.Timestamp("2026-10-01")
    compare(frame)


def test_time_boundary_transitive_chains_and_missing_date_early_break_match():
    frame = make_dataset("dense", 9)
    start = pd.Timestamp("2026-10-01")
    frame["Date"] = [start, start + pd.Timedelta(hours=48), start + pd.Timedelta(hours=96),
                     start + pd.Timedelta(hours=144), start + pd.Timedelta(hours=192, microseconds=1),
                     pd.NaT, pd.NaT, start + pd.Timedelta(hours=48, nanoseconds=1), "invalid"]
    frame["Snippet"] = ["same exact snippet", "", "", "", "different snippet", "same exact snippet", "", "", ""]
    compare(frame)


@pytest.mark.parametrize("timezone", ["UTC", "America/Toronto"])
def test_timezone_aware_dates_match_original(timezone):
    frame = make_dataset("dense", 15)
    frame["Date"] = pd.date_range("2026-10-31", periods=15, freq="12h", tz=timezone)
    compare(frame)


def test_extension_dtypes_column_order_and_helper_name_collisions_match():
    frame = make_dataset("small_groups", 30)
    frame["Type"] = frame["Type"].astype("category")
    frame["Type"] = frame["Type"].cat.add_categories([""])
    frame["Author"] = frame["Author"].astype("string")
    frame["Snippet"] = frame["Snippet"].astype("string")
    frame["Impressions"] = frame["Impressions"].astype("Int64")
    for column in ["_date_dt", "_snippet_norm", "_snippet_prefix_tokens", "_snippet_len"]:
        frame[column] = "existing helper-name column"
    compare(frame[frame.columns[::-1]])


@pytest.mark.parametrize("rows", [2, 7, 8, 9])
def test_lookup_cache_boundary_keeps_identical_tied_winners(rows):
    frame = make_dataset("dense", rows)
    frame["Author"] = "Reporter"
    frame["Impressions"] = 100
    compare(frame)


@pytest.mark.parametrize("case", ["duplicate_index", "categorical_without_blank", "missing_snippet_scalar"])
def test_preexisting_invalid_input_errors_are_preserved(case):
    frame = make_dataset("dense", 12)
    if case == "duplicate_index":
        frame.index = [0] * len(frame)
    elif case == "categorical_without_blank":
        frame["Type"] = frame["Type"].astype("category")
    else:
        frame["Snippet"] = pd.NA
    with pytest.raises(Exception) as original_error:
        reference.dedupe_non_broadcast_by_fields(frame)
    with pytest.raises(type(original_error.value)) as optimized_error:
        dedupe_non_broadcast_by_fields(frame)
    assert str(optimized_error.value) == str(original_error.value)
