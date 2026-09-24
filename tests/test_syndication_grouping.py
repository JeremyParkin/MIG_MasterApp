from __future__ import annotations

import unittest
from types import SimpleNamespace

import pandas as pd

from processing.data_quality import build_data_quality_warnings
from processing.effective_reach import apply_effective_reach_traditional
from processing.story_grouping import build_unique_story_table, cluster_by_media_type, mark_prime_examples
from processing.ai_tagging import call_ai_tagging
from processing.sentiment_config import prepare_sentiment_datasets
from processing.tagging_config import prepare_tagging_datasets
from utils.io import build_upload_quality_report, normalize_uploaded_dataframe


class SyndicationGroupingTests(unittest.TestCase):
    def test_tagging_accepts_modern_tool_call_response(self) -> None:
        response = SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=12, completion_tokens=8),
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        tool_calls=[
                            SimpleNamespace(
                                function=SimpleNamespace(
                                    arguments='{"tag":"Innovation","confidence":90,"explanation":"Relevant."}'
                                )
                            )
                        ]
                    )
                )
            ],
        )

        class FakeCompletions:
            def create(self, **kwargs):
                self.kwargs = kwargs
                return response

        completions = FakeCompletions()
        client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        result, input_tokens, output_tokens = call_ai_tagging(
            client,
            pd.Series({"Headline": "A story", "Snippet": "Story text"}),
            {"Innovation": "New technology", "Other": "No substantive tag applies."},
            "Multiple applicable tags",
            "gpt-5.6-luna",
        )

        self.assertEqual(result["tag"], "Innovation")
        self.assertEqual((input_tokens, output_tokens), (12, 8))
        self.assertEqual(completions.kwargs["tool_choice"]["function"]["name"], "apply_multiple_tags")
        self.assertEqual(completions.kwargs["reasoning_effort"], "low")

    def test_upload_normalizes_spaced_syndication_id_header(self) -> None:
        raw = pd.DataFrame(
            {
                "Published Date": ["09/24/2026"],
                "Published Time": ["07:24:04 AM"],
                "Media Type": ["ONLINE_NEWS"],
                "Coverage Snippet": ["Story text"],
                "Image URLs": ["https://example.com/image.jpg"],
                "Syndication ID": ["abc123"],
            }
        )

        normalized = normalize_uploaded_dataframe(raw)

        self.assertIn("SyndicationId", normalized.columns)
        self.assertEqual(normalized.loc[0, "SyndicationId"], "abc123")
        self.assertIn("Image URLs", normalized.columns)
        self.assertEqual(normalized.loc[0, "Image URLs"], "https://example.com/image.jpg")

    def test_podcast_is_canonical_but_reports_unavailable_effective_reach(self) -> None:
        raw = pd.DataFrame(
            {
                "Date": ["2026-09-24"],
                "Media Type": ["PODCAST"],
                "Headline": ["Podcast story"],
                "Snippet": ["Transcript text"],
                "Impressions": [1000],
            }
        )

        normalized = normalize_uploaded_dataframe(raw)
        report = build_upload_quality_report(raw, normalized)
        reach = apply_effective_reach_traditional(normalized)
        warnings = build_data_quality_warnings(normalized)

        self.assertEqual(report["unrecognized_media_type_values"], [])
        self.assertEqual(report["podcast_row_count"], 1)
        self.assertTrue(any(w["title"] == "Podcast coverage has no Effective Reach model" for w in report["warnings"]))
        self.assertTrue(pd.isna(reach.loc[0, "Effective Reach"]))
        self.assertTrue(any("Podcast coverage is present" in warning for warning in warnings))

    def test_same_syndication_id_groups_across_dates_and_media_types(self) -> None:
        df = pd.DataFrame(
            {
                "Date": pd.to_datetime(["2026-09-01", "2026-09-05"]),
                "Type": ["ONLINE", "PRINT"],
                "Headline": ["Original school story", "Print edition school story"],
                "Snippet": [
                    "A full version of the school recognition story with many details.",
                    "A clipped print version with fewer shared details.",
                ],
                "Outlet": ["Daily Example", "Daily Example Print"],
                "SyndicationId": ["shared-id", "shared-id"],
            }
        )

        grouped = cluster_by_media_type(df, similarity_threshold=0.99)

        self.assertEqual(grouped["Group ID"].nunique(), 1)
        self.assertEqual(set(grouped["Grouping Source"]), {"SyndicationId"})

    def test_text_similarity_still_merges_different_syndication_ids(self) -> None:
        df = pd.DataFrame(
            {
                "Date": pd.to_datetime(["2026-09-01", "2026-09-01"]),
                "Type": ["ONLINE", "ONLINE"],
                "Headline": [
                    "Students recognized by national program",
                    "Students recognized by national program",
                ],
                "Snippet": [
                    "Students were recognized by the national program for academic achievement and college readiness.",
                    "Students were recognized by the national program for academic achievement and college readiness.",
                ],
                "Outlet": ["Example A", "Example B"],
                "SyndicationId": ["platform-a", "platform-b"],
            }
        )

        grouped = cluster_by_media_type(df, similarity_threshold=0.99)

        self.assertEqual(grouped["Group ID"].nunique(), 1)
        self.assertEqual(set(grouped["Grouping Source"]), {"SyndicationId + Text Similarity"})
        self.assertEqual(set(grouped["Grouping Warning"]), {"Multiple SyndicationIds in group"})

    def test_prime_example_prefers_fuller_snippet_for_ai_representative(self) -> None:
        full_snippet = " ".join(["full article text"] * 80)
        clipped_snippet = "clipped story text"
        df = pd.DataFrame(
            {
                "Date": pd.to_datetime(["2026-09-01", "2026-09-04"]),
                "Type": ["ONLINE", "ONLINE"],
                "Headline": ["Students recognized", "Students recognized"],
                "Snippet": [clipped_snippet, full_snippet],
                "Outlet": ["High Reach Outlet", "Lower Reach Outlet"],
                "Impressions": [1_000_000, 1_000],
                "Mentions": [1, 1],
                "Effective Reach": [1_000_000, 1_000],
                "Coverage Flags": ["", ""],
                "SyndicationId": ["shared-story", "shared-story"],
            }
        )

        grouped = mark_prime_examples(cluster_by_media_type(df, similarity_threshold=0.99))
        unique = build_unique_story_table(grouped)

        self.assertEqual(grouped["Group ID"].nunique(), 1)
        self.assertEqual(int(grouped["Prime Example"].sum()), 1)
        self.assertEqual(unique.loc[0, "Snippet"], full_snippet)
        self.assertEqual(int(unique.loc[0, "Group Count"]), 2)

    def test_tagging_dataset_uses_prime_row_as_sole_ai_representative(self) -> None:
        full_snippet = " ".join(["complete representative story"] * 60)
        clipped_snippet = "short"
        grouped = pd.DataFrame(
            {
                "Date": pd.to_datetime(["2026-09-01", "2026-09-02"]),
                "Type": ["ONLINE", "ONLINE"],
                "Headline": ["Story headline", "Story headline"],
                "Snippet": [clipped_snippet, full_snippet],
                "Outlet": ["Outlet A", "Outlet B"],
                "Impressions": [1000, 500],
                "Mentions": [1, 1],
                "Effective Reach": [1000, 500],
                "Coverage Flags": ["", ""],
                "Group ID": [7, 7],
                "Prime Example": [0, 1],
            }
        )

        prepared = prepare_tagging_datasets(
            grouped,
            sample_mode="full",
            excluded_flags=[],
            full_override=True,
        )

        unique = prepared["df_tagging_unique"]
        self.assertEqual(len(unique), 1)
        self.assertEqual(unique.loc[0, "Snippet"], full_snippet)
        self.assertEqual(int(unique.loc[0, "Group Count"]), 2)

    def test_sentiment_dataset_uses_prime_row_as_sole_ai_representative(self) -> None:
        full_snippet = " ".join(["complete sentiment story"] * 60)
        clipped_snippet = "short"
        grouped = pd.DataFrame(
            {
                "Date": pd.to_datetime(["2026-09-01", "2026-09-02"]),
                "Type": ["ONLINE", "ONLINE"],
                "Headline": ["Story headline", "Story headline"],
                "Snippet": [clipped_snippet, full_snippet],
                "Outlet": ["Outlet A", "Outlet B"],
                "Impressions": [1000, 500],
                "Mentions": [1, 1],
                "Effective Reach": [1000, 500],
                "Coverage Flags": ["", ""],
                "Group ID": [7, 7],
                "Prime Example": [0, 1],
            }
        )

        prepared = prepare_sentiment_datasets(
            grouped,
            sample_mode="full",
            excluded_flags=[],
            full_override=True,
        )

        unique = prepared["df_sentiment_unique"]
        self.assertEqual(len(unique), 1)
        self.assertEqual(unique.loc[0, "Snippet"], full_snippet)
        self.assertEqual(int(unique.loc[0, "Group Count"]), 2)


if __name__ == "__main__":
    unittest.main()
