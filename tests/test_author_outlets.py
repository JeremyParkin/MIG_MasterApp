from __future__ import annotations

import unittest

from processing.author_outlets import get_matched_authors_df


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


if __name__ == "__main__":
    unittest.main()
