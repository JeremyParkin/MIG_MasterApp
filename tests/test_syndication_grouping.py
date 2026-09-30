from __future__ import annotations

import unittest
from types import SimpleNamespace

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from processing.data_quality import build_data_quality_warnings
from processing.download_exports import order_clean_trad_grouping_columns
from processing.effective_reach import apply_effective_reach_traditional
from processing.story_grouping import build_unique_story_table, cluster_by_media_type, mark_prime_examples
from processing.ai_sentiment import call_ai_sentiment as call_ai_sentiment_first_pass
from processing.ai_tagging import call_ai_tagging
from processing.sentiment_config import prepare_sentiment_datasets
from processing.tagging_config import prepare_tagging_datasets
from utils.io import build_upload_quality_report, normalize_uploaded_dataframe


class SyndicationGroupingTests(unittest.TestCase):
    @staticmethod
    def _grouping_frame(
        headlines: list[str],
        snippets: list[str],
        syndication_ids: list[str],
        media_types: list[str] | None = None,
    ) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "Date": pd.to_datetime(["2026-09-01"] * len(headlines)),
                "Type": media_types or ["ONLINE"] * len(headlines),
                "Headline": headlines,
                "Snippet": snippets,
                "Outlet": [f"Example {index}" for index in range(len(headlines))],
                "SyndicationId": syndication_ids,
            }
        )

    @staticmethod
    def _pairwise_cosine(left: str, right: str) -> float:
        matrix = TfidfVectorizer().fit_transform([left, right])
        return float((matrix * matrix.T).toarray()[0, 1])

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
        self.assertEqual(completions.kwargs["reasoning_effort"], "none")

    def test_sentiment_tool_call_uses_chat_completions_compatible_reasoning_effort(self) -> None:
        response = SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        tool_calls=[
                            SimpleNamespace(
                                function=SimpleNamespace(
                                    arguments='{"sentiment":"NEUTRAL","confidence":88,"explanation":"Factual mention."}'
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
        result, input_tokens, output_tokens = call_ai_sentiment_first_pass(
            client,
            "Headline: College Board announces dates",
            "gpt-5.6-luna",
            [
                {
                    "name": "analyze_sentiment",
                    "description": "Analyze sentiment.",
                    "parameters": {"type": "object", "properties": {}},
                }
            ],
            "3-way",
        )

        self.assertEqual(result["sentiment"], "NEUTRAL")
        self.assertEqual((input_tokens, output_tokens), (10, 5))
        self.assertEqual(completions.kwargs["tool_choice"]["function"]["name"], "analyze_sentiment")
        self.assertEqual(completions.kwargs["reasoning_effort"], "none")

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

    def test_singleton_group_has_no_grouping_source_or_warning(self) -> None:
        df = pd.DataFrame(
            {
                "Date": pd.to_datetime(["2026-09-01"]),
                "Type": ["ONLINE"],
                "Headline": ["Standalone school story"],
                "Snippet": ["A distinct report with no matching coverage."],
                "Outlet": ["Example A"],
                "SyndicationId": ["standalone-id"],
            }
        )

        grouped = cluster_by_media_type(df, similarity_threshold=0.99)

        self.assertEqual(grouped["Group ID"].nunique(), 1)
        self.assertEqual(grouped.loc[0, "Grouping Source"], "")
        self.assertEqual(grouped.loc[0, "Grouping Warning"], "")

    def test_text_similarity_group_retains_source_without_syndication_id(self) -> None:
        df = pd.DataFrame(
            {
                "Date": pd.to_datetime(["2026-09-01", "2026-09-01"]),
                "Type": ["ONLINE", "ONLINE"],
                "Headline": ["Students recognized by national program", "Students recognized by national program"],
                "Snippet": [
                    "Students were recognized by the national program for academic achievement and college readiness.",
                    "Students were recognized by the national program for academic achievement and college readiness.",
                ],
                "Outlet": ["Example A", "Example B"],
                "SyndicationId": ["", ""],
            }
        )

        grouped = cluster_by_media_type(df, similarity_threshold=0.99)

        self.assertEqual(grouped["Group ID"].nunique(), 1)
        self.assertEqual(set(grouped["Grouping Source"]), {"Text Similarity"})
        self.assertEqual(set(grouped["Grouping Warning"]), {""})

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

    def test_rich_text_pair_between_normal_and_weak_thresholds_still_groups(self) -> None:
        shared_details = " ".join(f"detail{number}" for number in range(34))
        snippets = [f"{shared_details} alpha", f"{shared_details} beta"]
        combined = [f"Shared report {snippet}" for snippet in snippets]

        similarity = self._pairwise_cosine(*combined)
        grouped = cluster_by_media_type(
            self._grouping_frame(
                ["Shared report", "Shared report"],
                snippets,
                ["platform-a", "platform-b"],
            )
        )

        self.assertGreaterEqual(similarity, 0.935)
        self.assertLess(similarity, 0.95)
        self.assertEqual(grouped["Group ID"].nunique(), 1)
        self.assertEqual(set(grouped["Grouping Source"]), {"SyndicationId + Text Similarity"})

    def test_weak_pair_between_normal_and_weak_thresholds_does_not_group(self) -> None:
        headline = "Regional public school officials announce expanded support program for students and families across the district"
        snippets = [
            "following detailed review at the regular board meeting with information posted online for families next week.",
            "following detailed review at the special board meeting with information posted online for families next week.",
        ]
        combined = [f"{headline} {snippet}" for snippet in snippets]

        similarity = self._pairwise_cosine(*combined)
        grouped = cluster_by_media_type(
            self._grouping_frame(
                [headline, headline],
                snippets,
                ["platform-a", "platform-b"],
            )
        )

        self.assertGreaterEqual(similarity, 0.935)
        self.assertLess(similarity, 0.95)
        self.assertEqual(grouped["Group ID"].nunique(), 2)
        self.assertEqual(set(grouped["Grouping Source"]), {""})
        self.assertEqual(set(grouped["Grouping Warning"]), {""})

    def test_exact_normalized_sparse_duplicate_still_groups(self) -> None:
        headline = "College Board announces expanded AP computer science course"
        grouped = cluster_by_media_type(
            self._grouping_frame(
                [headline, headline],
                [
                    ">> Students can register online through their schools beginning next week.",
                    "  Students   can register online through their schools beginning next week.  ",
                ],
                ["platform-a", "platform-b"],
            )
        )

        self.assertEqual(grouped["Group ID"].nunique(), 1)
        self.assertEqual(set(grouped["Grouping Source"]), {"SyndicationId + Text Similarity"})
        self.assertEqual(set(grouped["Grouping Warning"]), {"Multiple SyndicationIds in group"})

    def test_blank_snippet_is_weak_evidence(self) -> None:
        shared_headline = " ".join(f"headline{number}" for number in range(34))
        headlines = [f"{shared_headline} alpha", f"{shared_headline} beta"]

        similarity = self._pairwise_cosine(*headlines)
        grouped = cluster_by_media_type(
            self._grouping_frame(headlines, ["", ""], ["platform-a", "platform-b"])
        )

        self.assertGreaterEqual(similarity, 0.935)
        self.assertLess(similarity, 0.95)
        self.assertEqual(grouped["Group ID"].nunique(), 2)

    def test_rejected_weak_edges_cannot_form_a_transitive_bridge(self) -> None:
        headline = " ".join(f"headline{number}" for number in range(45))
        snippets = ["alpha", "beta", "gamma"]
        matrix = TfidfVectorizer().fit_transform([f"{headline} {snippet}" for snippet in snippets])
        similarities = (matrix * matrix.T).toarray()
        grouped = cluster_by_media_type(
            self._grouping_frame(
                [headline, headline, headline],
                snippets,
                ["platform-a", "platform-b", "platform-c"],
            )
        )

        self.assertTrue((similarities[0, 1:] >= 0.935).all())
        self.assertTrue((similarities[0, 1:] < 0.95).all())
        self.assertEqual(grouped["Group ID"].nunique(), 3)

    def test_text_similarity_does_not_cross_media_type_boundaries(self) -> None:
        headline = "College Board announces expanded AP computer science course"
        snippet = "Students can register online through their schools beginning next week."
        grouped = cluster_by_media_type(
            self._grouping_frame(
                [headline, headline],
                [snippet, snippet],
                ["platform-a", "platform-b"],
                media_types=["ONLINE", "TV"],
            )
        )

        self.assertEqual(grouped["Group ID"].nunique(), 2)

    def test_clean_trad_grouping_columns_are_adjacent(self) -> None:
        df = pd.DataFrame(
            {
                "Headline": ["Story"],
                "Group ID": [1],
                "Outlet": ["Example"],
                "Grouping Warning": [""],
                "SyndicationId": ["shared-id"],
                "Grouping Source": ["SyndicationId"],
                "Impressions": [100],
            }
        )

        ordered = order_clean_trad_grouping_columns(df)

        self.assertEqual(
            list(ordered.columns),
            [
                "Headline",
                "Group ID",
                "SyndicationId",
                "Grouping Source",
                "Grouping Warning",
                "Outlet",
                "Impressions",
            ],
        )

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
