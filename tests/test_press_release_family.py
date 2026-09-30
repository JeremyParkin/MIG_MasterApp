from __future__ import annotations

import unittest

import pandas as pd

from processing.analysis_context import apply_coverage_flag_policy
from processing.coverage_flags import (
    STORY_FAMILY_FLAGS_COL,
    STORY_FAMILY_PRESS_RELEASE_EVIDENCE_COL,
    add_coverage_flags,
    apply_story_family_press_release_flags,
)
from processing.download_exports import (
    order_clean_trad_press_release_columns,
    remove_inactive_workflow_columns,
)
from processing.regions import filter_regions_df
from processing.sentiment_config import apply_coverage_flag_exclusions as apply_sentiment_exclusions
from processing.standard_cleaning import run_standard_cleaning
from processing.story_grouping import cluster_by_media_type, mark_prime_examples
from processing.tagging_config import apply_coverage_flag_exclusions as apply_tagging_exclusions
from processing.top_stories import apply_filters


class PressReleaseStoryFamilyTests(unittest.TestCase):
    @staticmethod
    def _rows(**overrides) -> pd.DataFrame:
        row_count = len(overrides.get("Headline", ["College Board announcement", "College Board announcement"]))
        values = {
            "Date": pd.to_datetime(["2026-09-01"] * row_count),
            "Type": ["ONLINE"] * row_count,
            "Headline": ["College Board announcement"] * row_count,
            "Snippet": ["A release about new student opportunities."] * row_count,
            "Outlet": ["Editorial Daily"] * row_count,
            "URL": ["https://daily.example/story"] * row_count,
            "Author": ["Staff Reporter"] * row_count,
            "SyndicationId": ["family-a"] * row_count,
            "Mentions": [1] * row_count,
            "Impressions": [100] * row_count,
            "Effective Reach": [100] * row_count,
        }
        if row_count == 2 and "Outlet" not in overrides:
            values["Outlet"] = ["Distribution Wire", "Editorial Daily"]
        if row_count == 2 and "URL" not in overrides:
            values["URL"] = ["https://wire.example/news-release/college-board", "https://daily.example/story"]
        if row_count == 2 and "Author" not in overrides:
            values["Author"] = ["Newswire", "Staff Reporter"]
        if row_count == 2 and "Impressions" not in overrides:
            values["Impressions"] = [100, 1000]
            values["Effective Reach"] = [100, 1000]
        values.update(overrides)
        return pd.DataFrame(values)

    def _group_and_classify(self, rows: pd.DataFrame) -> pd.DataFrame:
        flagged = add_coverage_flags(rows)
        grouped = cluster_by_media_type(flagged, similarity_threshold=0.99)
        return apply_story_family_press_release_flags(grouped)

    def test_source_press_release_is_preserved_when_online_types_merge(self) -> None:
        cleaned = run_standard_cleaning(
            pd.DataFrame(
                {
                    "Type": ["PRESS RELEASE"],
                    "Headline": ["Announcement"],
                    "Snippet": ["Plain source-provided release."],
                    "Outlet": ["Client newsroom"],
                    "URL": ["https://example.com/news"],
                }
            ),
            merge_online=True,
        )["df_traditional"]

        self.assertEqual(cleaned.loc[0, "Original Type"], "PRESS RELEASE")
        self.assertEqual(cleaned.loc[0, "Type"], "ONLINE")
        self.assertIn("Press Release", cleaned.loc[0, "Coverage Flags"])

    def test_source_press_release_remains_press_release_without_merge(self) -> None:
        cleaned = run_standard_cleaning(
            pd.DataFrame(
                {
                    "Type": ["PRESS RELEASE"],
                    "Headline": ["Announcement"],
                    "Snippet": ["Plain source-provided release."],
                    "Outlet": ["Client newsroom"],
                    "URL": ["https://example.com/news"],
                }
            ),
            merge_online=False,
        )["df_traditional"]

        self.assertEqual(cleaned.loc[0, "Original Type"], "PRESS RELEASE")
        self.assertEqual(cleaned.loc[0, "Type"], "PRESS RELEASE")

    def test_source_press_release_coexists_with_direct_heuristic_flags_after_online_merge(self) -> None:
        cases = [
            (
                "Market Report Spam",
                {
                    "Headline": "Global market forecast 2026",
                    "Outlet": "Editorial Desk",
                    "URL": "https://example.com/market",
                },
            ),
            (
                "Advertorial",
                {
                    "Headline": "A sponsored release announcement",
                    "Outlet": "Editorial Desk",
                    "URL": "https://example.com/sponsored-content",
                },
            ),
            (
                "Financial Outlet",
                {
                    "Headline": "A financial release announcement",
                    "Outlet": "Nasdaq",
                    "URL": "https://example.com/financial",
                },
            ),
            (
                "Aggregator",
                {
                    "Headline": "An aggregator release announcement",
                    "Outlet": "Yahoo",
                    "URL": "https://example.com/aggregator",
                },
            ),
            (
                "User-Generated",
                {
                    "Headline": "A user-generated release announcement",
                    "Outlet": "Editorial Desk",
                    "URL": "https://medium.com/release-announcement",
                },
            ),
        ]

        for coexisting_flag, fields in cases:
            with self.subTest(coexisting_flag=coexisting_flag):
                cleaned = run_standard_cleaning(
                    pd.DataFrame(
                        {
                            "Type": ["PRESS RELEASE"],
                            "Headline": [fields["Headline"]],
                            "Snippet": ["Plain source-provided release."],
                            "Outlet": [fields["Outlet"]],
                            "URL": [fields["URL"]],
                            "Author": ["Staff Reporter"],
                        }
                    ),
                    merge_online=True,
                )["df_traditional"]
                classified = apply_story_family_press_release_flags(
                    cluster_by_media_type(cleaned, similarity_threshold=0.99)
                )

                self.assertEqual(classified.loc[0, "Original Type"], "PRESS RELEASE")
                self.assertEqual(classified.loc[0, "Type"], "ONLINE")
                self.assertIn("Press Release", classified.loc[0, "Coverage Flags"])
                self.assertIn(coexisting_flag, classified.loc[0, "Coverage Flags"])
                self.assertEqual(classified.loc[0, STORY_FAMILY_FLAGS_COL], "Press Release")
                self.assertEqual(
                    classified.loc[0, STORY_FAMILY_PRESS_RELEASE_EVIDENCE_COL],
                    "Source type",
                )

    def test_heuristic_press_release_remains_suppressed_by_market_report_spam(self) -> None:
        flagged = add_coverage_flags(
            pd.DataFrame(
                {
                    "Type": ["ONLINE"],
                    "Headline": ["Global market forecast 2026"],
                    "Snippet": ["Newswire coverage without a source-declared release type."],
                    "Outlet": ["Editorial Desk"],
                    "URL": ["https://example.com/market"],
                    "Author": ["Staff Reporter"],
                }
            )
        )

        self.assertIn("Market Report Spam", flagged.loc[0, "Coverage Flags"])
        self.assertNotIn("Press Release", flagged.loc[0, "Coverage Flags"])

    def test_ordinary_online_row_stays_unflagged(self) -> None:
        flagged = add_coverage_flags(
            self._rows(
                Type=["ONLINE"], Headline=["Independent reporting"], Snippet=["Reported coverage."],
                Outlet=["Editorial Daily"], URL=["https://daily.example/story"], Author=["Reporter"],
                SyndicationId=["ordinary"], Mentions=[1], Impressions=[100], Effective_Reach=[100],
            )
        )
        self.assertEqual(flagged.loc[0, "Coverage Flags"], "")

    def test_strong_evidence_propagates_family_flag_without_rewriting_direct_flags(self) -> None:
        grouped = self._group_and_classify(self._rows())

        self.assertEqual(grouped["Group ID"].nunique(), 1)
        self.assertIn("Press Release", grouped.loc[0, "Coverage Flags"])
        self.assertEqual(grouped.loc[1, "Coverage Flags"], "")
        self.assertEqual(grouped[STORY_FAMILY_FLAGS_COL].tolist(), ["Press Release", "Press Release"])
        self.assertTrue(grouped[STORY_FAMILY_PRESS_RELEASE_EVIDENCE_COL].str.contains("Press-release URL").all())

    def test_author_and_distributor_outlet_are_independently_strong_evidence(self) -> None:
        cases = [
            ("Newswire", "Editorial One", "Press-release author"),
            ("Reporter", "Business Wire", "Distribution outlet"),
        ]
        for author, outlet, evidence in cases:
            with self.subTest(evidence=evidence):
                grouped = self._group_and_classify(
                    self._rows(
                        Author=[author, "Reporter"],
                        Outlet=[outlet, "Editorial Two"],
                        URL=["https://one.example/story", "https://two.example/story"],
                    )
                )
                self.assertTrue(grouped[STORY_FAMILY_FLAGS_COL].eq("Press Release").all())
                self.assertTrue(grouped[STORY_FAMILY_PRESS_RELEASE_EVIDENCE_COL].str.contains(evidence).all())

    def test_snippet_only_press_release_signal_does_not_propagate(self) -> None:
        rows = self._rows(
            Outlet=["Editorial One", "Editorial Two"],
            URL=["https://one.example/story", "https://two.example/story"],
            Author=["Reporter", "Reporter"],
            Snippet=["Newswire distributed this announcement.", "Independent coverage of the announcement."],
        )
        grouped = self._group_and_classify(rows)

        self.assertIn("Press Release", grouped.loc[0, "Coverage Flags"])
        self.assertEqual(grouped[STORY_FAMILY_FLAGS_COL].tolist(), ["", ""])

    def test_family_flag_does_not_leak_across_groups(self) -> None:
        rows = pd.concat(
            [
                self._rows(),
                self._rows(
                    Headline=["Separate editorial story"], Snippet=["Independent reporting."],
                    Outlet=["Editorial Three"], URL=["https://three.example/story"], Author=["Reporter"],
                    SyndicationId=["family-b"], Mentions=[1], Impressions=[20], Effective_Reach=[20],
                ),
            ],
            ignore_index=True,
        )
        grouped = self._group_and_classify(rows)

        family_a = grouped[grouped["SyndicationId"].eq("family-a")]
        family_b = grouped[grouped["SyndicationId"].eq("family-b")]
        self.assertTrue(family_a[STORY_FAMILY_FLAGS_COL].eq("Press Release").all())
        self.assertTrue(family_b[STORY_FAMILY_FLAGS_COL].eq("").all())

    def test_prime_example_uses_direct_flags_not_family_flags(self) -> None:
        grouped = mark_prime_examples(self._group_and_classify(self._rows()))
        prime = grouped.loc[grouped["Prime Example"].eq(1)].iloc[0]

        self.assertEqual(prime["Outlet"], "Editorial Daily")
        self.assertEqual(prime[STORY_FAMILY_FLAGS_COL], "Press Release")

    def test_effective_filters_exclude_an_entire_story_family(self) -> None:
        grouped = self._group_and_classify(self._rows())

        self.assertTrue(apply_coverage_flag_policy(grouped, ["Press Release"]).empty)
        self.assertTrue(apply_sentiment_exclusions(grouped, ["Press Release"]).empty)
        self.assertTrue(apply_tagging_exclusions(grouped, ["Press Release"]).empty)
        self.assertTrue(
            apply_filters(grouped, None, None, [], ["Press Release"], []).empty
        )
        region_rows = grouped.assign(Country="Canada", **{"Prov/State": "ON", "City": "Toronto"})
        self.assertTrue(filter_regions_df(region_rows, ["Press Release"]).empty)

    def test_reclassification_resets_stale_family_columns(self) -> None:
        grouped = self._group_and_classify(self._rows())
        grouped["Original Type"] = "ONLINE"
        grouped["URL"] = "https://editorial.example/story"
        grouped["Author"] = "Reporter"
        grouped["Outlet"] = "Editorial Daily"

        refreshed = apply_story_family_press_release_flags(grouped)
        self.assertTrue(refreshed[STORY_FAMILY_FLAGS_COL].eq("").all())
        self.assertTrue(refreshed[STORY_FAMILY_PRESS_RELEASE_EVIDENCE_COL].eq("").all())

    def test_clean_trad_order_and_audit_evidence_visibility(self) -> None:
        df = pd.DataFrame(
            {
                "Headline": ["Story"],
                "Original Type": ["PRESS RELEASE"],
                "Type": ["ONLINE"],
                "Coverage Flags": ["Press Release"],
                "Story Family Flags": ["Press Release"],
                "Story Family Press Release Evidence": ["Source type"],
            }
        )

        ordered = order_clean_trad_press_release_columns(df)
        self.assertEqual(
            ordered.columns.tolist(),
            ["Headline", "Original Type", "Type", "Coverage Flags", "Story Family Flags", "Story Family Press Release Evidence"],
        )
        self.assertNotIn(
            "Story Family Press Release Evidence",
            remove_inactive_workflow_columns(df, {}, include_labeling_audit_columns=False).columns,
        )
        self.assertIn(
            "Story Family Press Release Evidence",
            remove_inactive_workflow_columns(df, {}, include_labeling_audit_columns=True).columns,
        )

    def test_legacy_rows_without_story_family_columns_remain_usable(self) -> None:
        legacy = pd.DataFrame({"Coverage Flags": [""], "Headline": ["Story"]})
        self.assertEqual(len(apply_coverage_flag_policy(legacy, ["Press Release"])), 1)


if __name__ == "__main__":
    unittest.main()
