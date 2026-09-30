from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd

from processing.author_outlets import (
    AUTHOR_OUTLET_PREFETCH_BATCH_SIZE,
    AUTHOR_OUTLET_PREFETCH_MAX_WORKERS,
    build_auth_outlet_table,
    get_author_outlet_prefetch_authors,
    get_matched_authors_df,
    make_author_cache_key,
    prefetch_author_outlet_cache_entries,
    should_prefetch_author_outlet_matches,
)


def search_results(*rows: tuple[str, str, str]) -> dict:
    return {
        "results": [
            {
                "firstName": first,
                "lastName": last,
                "primaryEmployment": {"outletName": outlet},
            }
            for first, last, outlet in rows
        ]
    }


class AuthorOutletMatchOrderingTests(unittest.TestCase):
    def test_exact_match_returned_second_moves_to_first_with_matching_choice_order(self) -> None:
        matched, _db_outlets, possibles = get_matched_authors_df(
            search_results(("Jamie", "Smith", "First Outlet"), ("Alex", "Rivera", "Exact Outlet")),
            outlets_in_coverage_list=[],
            author_name="Alex Rivera",
        )

        self.assertEqual(matched["Name"].tolist(), ["Alex Rivera", "Jamie Smith"])
        self.assertEqual(possibles, matched["Outlet"].tolist())
        self.assertEqual(possibles[0], "Exact Outlet")

    def test_multiple_exact_matches_keep_api_order_within_the_exact_bucket(self) -> None:
        matched, _db_outlets, _possibles = get_matched_authors_df(
            search_results(
                ("Jamie", "Smith", "First Outlet"),
                ("Alex", "Rivera", "Exact One"),
                ("Alex", "Rivera", "Exact Two"),
            ),
            outlets_in_coverage_list=[],
            author_name="Alex Rivera",
        )

        self.assertEqual(matched["Outlet"].tolist(), ["Exact One", "Exact Two", "First Outlet"])

    def test_no_exact_match_preserves_api_order_without_coverage_priority(self) -> None:
        matched, _db_outlets, possibles = get_matched_authors_df(
            search_results(("Jamie", "Smith", "First Outlet"), ("Morgan", "Lee", "Second Outlet")),
            outlets_in_coverage_list=[],
            author_name="Alex Rivera",
        )

        self.assertEqual(matched["Outlet"].tolist(), ["First Outlet", "Second Outlet"])
        self.assertEqual(possibles, ["First Outlet", "Second Outlet"])

    def test_coverage_priority_is_retained_within_name_match_buckets(self) -> None:
        matched, _db_outlets, possibles = get_matched_authors_df(
            search_results(
                ("Alex", "Rivera", "Other Outlet"),
                ("Alex", "Rivera", "Coverage Outlet"),
                ("Jamie", "Smith", "Another Outlet"),
            ),
            outlets_in_coverage_list=["Coverage Outlet"],
            author_name="Alex Rivera",
        )

        self.assertEqual(matched["Outlet"].tolist(), ["Coverage Outlet", "Other Outlet", "Another Outlet"])
        self.assertEqual(possibles, matched["Outlet"].tolist())

    def test_missing_database_results_leave_fallback_lists_empty(self) -> None:
        matched, db_outlets, possibles = get_matched_authors_df(
            {"results": []},
            outlets_in_coverage_list=["Coverage Outlet"],
            author_name="Alex Rivera",
        )

        self.assertTrue(matched.empty)
        self.assertEqual(db_outlets, [])
        self.assertEqual(possibles, [])


class AuthorOutletPrefetchTests(unittest.TestCase):
    @staticmethod
    def _author_rows() -> pd.DataFrame:
        return pd.DataFrame(
            {
                "Author": [f"Author {index}" for index in range(1, 13)],
                "Mentions": list(range(12, 0, -1)),
                "Impressions": [40, 120, 10, 110, 20, 100, 30, 90, 50, 80, 60, 70],
                "Effective Reach": [15, 25, 120, 20, 110, 30, 100, 40, 90, 50, 80, 60],
            }
        )

    def _queue(self, metric: str) -> pd.DataFrame:
        return build_auth_outlet_table(self._author_rows(), metric)

    def test_initial_mentions_queue_fetches_first_six_uncached_authors_in_order(self) -> None:
        queue = self._queue("Mentions")

        selected = get_author_outlet_prefetch_authors(queue, 0, {})

        self.assertEqual(selected, [f"Author {index}" for index in range(1, 7)])

    def test_impressions_and_effective_reach_use_their_own_queue_ordering(self) -> None:
        impressions_queue = self._queue("Impressions")
        reach_queue = self._queue("Effective Reach")

        self.assertEqual(
            get_author_outlet_prefetch_authors(impressions_queue, 0, {}),
            impressions_queue["Author"].head(AUTHOR_OUTLET_PREFETCH_BATCH_SIZE).tolist(),
        )
        self.assertEqual(
            get_author_outlet_prefetch_authors(reach_queue, 0, {}),
            reach_queue["Author"].head(AUTHOR_OUTLET_PREFETCH_BATCH_SIZE).tolist(),
        )
        self.assertNotEqual(
            impressions_queue["Author"].head(AUTHOR_OUTLET_PREFETCH_BATCH_SIZE).tolist(),
            reach_queue["Author"].head(AUTHOR_OUTLET_PREFETCH_BATCH_SIZE).tolist(),
        )

    def test_refill_selects_next_six_uncached_authors_from_current_position(self) -> None:
        queue = self._queue("Mentions")
        cache = {make_author_cache_key(f"Author {index}"): {} for index in range(1, 7)}

        selected = get_author_outlet_prefetch_authors(queue, 6, cache)

        self.assertEqual(selected, [f"Author {index}" for index in range(7, 13)])
        self.assertTrue(should_prefetch_author_outlet_matches(queue, 6, cache, rank_changed=False))

    def test_cached_authors_are_skipped_without_reordering_the_queue(self) -> None:
        queue = self._queue("Mentions")
        cache = {
            make_author_cache_key("Author 2"): {},
            make_author_cache_key("Author 4"): {},
        }

        selected = get_author_outlet_prefetch_authors(queue, 0, cache)

        self.assertEqual(selected, ["Author 1", "Author 3", "Author 5", "Author 6", "Author 7", "Author 8"])

    def test_ranking_change_reuses_cached_authors_and_fetches_only_new_queue_authors(self) -> None:
        mentions_queue = self._queue("Mentions")
        impressions_queue = self._queue("Impressions")
        cached_names = mentions_queue["Author"].head(AUTHOR_OUTLET_PREFETCH_BATCH_SIZE).tolist()
        cache = {make_author_cache_key(author_name): {} for author_name in cached_names}

        selected = get_author_outlet_prefetch_authors(impressions_queue, 0, cache)

        self.assertTrue(should_prefetch_author_outlet_matches(impressions_queue, 0, cache, rank_changed=True))
        self.assertTrue(all(make_author_cache_key(author_name) not in cache for author_name in selected))
        self.assertLessEqual(len(selected), AUTHOR_OUTLET_PREFETCH_BATCH_SIZE)
        self.assertEqual(
            selected,
            [
                author_name
                for author_name in impressions_queue["Author"].tolist()
                if make_author_cache_key(author_name) not in cache
            ][:AUTHOR_OUTLET_PREFETCH_BATCH_SIZE],
        )

    def test_cached_active_author_does_not_trigger_another_refill(self) -> None:
        queue = self._queue("Mentions")
        cache = {make_author_cache_key("Author 1"): {}}

        self.assertFalse(should_prefetch_author_outlet_matches(queue, 0, cache, rank_changed=False))

    def test_prefetch_execution_caps_batch_and_worker_count_at_six(self) -> None:
        class ImmediateFuture:
            def __init__(self, result):
                self._result = result

            def result(self):
                return self._result

        class RecordingExecutor:
            worker_counts: list[int] = []
            submitted: list[str] = []

            def __init__(self, max_workers: int):
                self.worker_counts.append(max_workers)

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def submit(self, function, author_name, *_args):
                self.submitted.append(author_name)
                return ImmediateFuture(function(author_name, *_args))

        cache: dict = {}
        author_names = [f"Author {index}" for index in range(1, 13)]

        with (
            patch("processing.author_outlets.ThreadPoolExecutor", RecordingExecutor),
            patch("processing.author_outlets.as_completed", lambda futures: list(futures)),
            patch(
                "processing.author_outlets.build_author_outlet_cache_entry",
                side_effect=lambda author_name, *_args: {"author_name": author_name},
            ),
        ):
            loaded = prefetch_author_outlet_cache_entries(author_names, cache, pd.DataFrame(), {})

        self.assertEqual(loaded, AUTHOR_OUTLET_PREFETCH_BATCH_SIZE)
        self.assertEqual(RecordingExecutor.worker_counts, [AUTHOR_OUTLET_PREFETCH_MAX_WORKERS])
        self.assertEqual(RecordingExecutor.submitted, author_names[:AUTHOR_OUTLET_PREFETCH_BATCH_SIZE])
        self.assertEqual(len(cache), AUTHOR_OUTLET_PREFETCH_BATCH_SIZE)


if __name__ == "__main__":
    unittest.main()
