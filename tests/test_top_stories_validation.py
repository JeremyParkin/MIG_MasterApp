import unittest

import pandas as pd

from processing.top_stories import (
    build_grouped_story_candidates,
    build_prime_grouped_story_candidates,
    build_story_identity_key,
    build_validation_source_candidate_table,
    cleanup_top_story_validation_state_after_removal,
    remove_saved_top_story,
    rotate_saved_story_source,
)


class TopStoriesValidationTests(unittest.TestCase):
    def _saved_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {"Group ID": "A", "Headline": "First", "Source Group IDs": "A", "Example URL": "https://a.test"},
                {"Group ID": "B", "Headline": "Second", "Source Group IDs": "B | C", "Example URL": "https://b.test"},
                {"Group ID": "D", "Headline": "Third", "Source Group IDs": "D", "Example URL": "https://d.test"},
            ]
        )

    def test_remove_first_validation_story_from_saved_top_stories(self) -> None:
        result = remove_saved_top_story(self._saved_df(), "A", fallback_group_id="A")

        self.assertEqual(result["Group ID"].tolist(), ["B", "D"])

    def test_remove_middle_validation_story_by_source_identity(self) -> None:
        result = remove_saved_top_story(self._saved_df(), "B | C", fallback_group_id="B")

        self.assertEqual(result["Group ID"].tolist(), ["A", "D"])

    def test_remove_last_validation_story_from_saved_top_stories(self) -> None:
        result = remove_saved_top_story(self._saved_df(), "D", fallback_group_id="D")

        self.assertEqual(result["Group ID"].tolist(), ["A", "B"])

    def test_removal_cleanup_drops_confirmation_and_invalidates_observation(self) -> None:
        story_key = build_story_identity_key("B | C", fallback_group_id="B")
        state = {
            "top_stories_validation_confirmed_keys": [story_key, "SRC::Z"],
            "top_story_observation_output": {"overall_observation": "Old copy"},
            "top_stories_validation_saved_signature": ("SRC::A", story_key, "SRC::D"),
            "top_stories_validation_index": 2,
        }

        cleanup_top_story_validation_state_after_removal(
            state,
            story_key,
            current_index=2,
            remaining_queue_count=2,
        )

        self.assertEqual(state["top_stories_validation_confirmed_keys"], ["SRC::Z"])
        self.assertIsNone(state["top_story_observation_output"])
        self.assertNotIn("top_stories_validation_saved_signature", state)
        self.assertEqual(state["top_stories_validation_index"], 1)

    def test_removal_does_not_mutate_underlying_source_data(self) -> None:
        saved = self._saved_df()
        source = pd.DataFrame(
            [
                {"Group ID": "A", "Headline": "First", "Mentions": 1},
                {"Group ID": "B", "Headline": "Second", "Mentions": 1},
            ]
        )
        source_before = source.copy(deep=True)

        _ = remove_saved_top_story(saved, "A", fallback_group_id="A")

        pd.testing.assert_frame_equal(source, source_before)

    def test_removed_story_can_be_readded_from_selection_source_row(self) -> None:
        saved = self._saved_df()
        removed = remove_saved_top_story(saved, "A", fallback_group_id="A")
        source_row = saved[saved["Group ID"].eq("A")]
        readded = pd.concat([removed, source_row], ignore_index=True)

        self.assertIn("A", readded["Group ID"].tolist())

    def test_selection_returns_one_candidate_per_canonical_group_without_topmerge(self) -> None:
        source = pd.DataFrame(
            [
                {
                    "Group ID": "A",
                    "Headline": "Same headline",
                    "Date": "2026-09-01",
                    "Mentions": 1,
                    "Impressions": 100,
                    "Effective Reach": 50,
                    "Outlet": "Outlet 1",
                    "URL": "https://a.test",
                    "Type": "ONLINE",
                    "Snippet": "Same story text long enough for a fingerprint.",
                    "Prime Example": 1,
                },
                {
                    "Group ID": "B",
                    "Headline": "Same headline",
                    "Date": "2026-09-01",
                    "Mentions": 1,
                    "Impressions": 200,
                    "Effective Reach": 75,
                    "Outlet": "Outlet 2",
                    "URL": "https://b.test",
                    "Type": "ONLINE",
                    "Snippet": "Same story text long enough for a fingerprint.",
                    "Prime Example": 1,
                },
            ]
        )

        candidates = build_grouped_story_candidates(source)

        self.assertEqual(candidates["Group ID"].astype(str).tolist(), ["A", "B"])
        self.assertFalse(candidates["Group ID"].astype(str).str.startswith("TOPMERGE::").any())

    def test_canonical_group_metrics_are_preserved(self) -> None:
        source = pd.DataFrame(
            [
                {
                    "Group ID": "A",
                    "Headline": "Grouped story",
                    "Date": "2026-09-01",
                    "Mentions": 4,
                    "Impressions": 100,
                    "Effective Reach": 50,
                    "Outlet": "Outlet 1",
                    "URL": "https://a.test",
                    "Type": "ONLINE",
                    "Snippet": "Prime",
                    "Prime Example": 1,
                },
                {
                    "Group ID": "A",
                    "Headline": "Grouped story",
                    "Date": "2026-09-01",
                    "Mentions": 5,
                    "Impressions": 200,
                    "Effective Reach": 75,
                    "Outlet": "Outlet 2",
                    "URL": "https://a2.test",
                    "Type": "ONLINE",
                    "Snippet": "Other",
                    "Prime Example": 0,
                },
            ]
        )

        candidate = build_grouped_story_candidates(source).iloc[0]

        self.assertEqual(candidate["Group ID"], "A")
        self.assertEqual(int(candidate["Mentions"]), 9)
        self.assertEqual(int(candidate["Impressions"]), 300)
        self.assertEqual(int(candidate["Effective Reach"]), 125)

    def test_validation_exposes_multiple_distinct_urls_within_one_group(self) -> None:
        source = pd.DataFrame(
            [
                {
                    "Group ID": "A",
                    "Headline": "One grouped story",
                    "Date": "2026-09-01",
                    "Mentions": 4,
                    "Impressions": 100,
                    "Effective Reach": 50,
                    "Outlet": "Outlet 1",
                    "URL": "https://a.test",
                    "Type": "ONLINE",
                    "Snippet": "Story text",
                    "Prime Example": 1,
                },
                {
                    "Group ID": "A",
                    "Headline": "One grouped story",
                    "Date": "2026-09-01",
                    "Mentions": 5,
                    "Impressions": 200,
                    "Effective Reach": 75,
                    "Outlet": "Outlet 2",
                    "URL": "https://a2.test",
                    "Type": "ONLINE",
                    "Snippet": "Duplicate mention",
                    "Prime Example": 0,
                },
            ]
        )

        candidates = build_prime_grouped_story_candidates(source)
        source_options = build_validation_source_candidate_table(
            source,
            source_group_ids="A",
            fallback_group_id="A",
            current_source=candidates.iloc[0].to_dict(),
        )

        self.assertEqual(int(candidates.iloc[0]["Mentions"]), 9)
        self.assertEqual(len(source_options), 2)
        self.assertEqual(source_options["Example URL"].tolist(), ["https://a.test", "https://a2.test"])

    def test_validation_source_candidates_dedupe_urls_and_do_not_leak_groups(self) -> None:
        source = pd.DataFrame(
            [
                {
                    "Group ID": "A",
                    "Headline": "Story A",
                    "Date": "2026-09-01",
                    "Mentions": 1,
                    "Impressions": 100,
                    "Effective Reach": 50,
                    "Outlet": "Outlet 1",
                    "URL": "https://same.test",
                    "Type": "ONLINE",
                    "Snippet": "Short",
                    "Prime Example": 1,
                },
                {
                    "Group ID": "A",
                    "Headline": "Story A",
                    "Date": "2026-09-01",
                    "Mentions": 1,
                    "Impressions": 500,
                    "Effective Reach": 250,
                    "Outlet": "Outlet 2",
                    "URL": "https://same.test",
                    "Type": "ONLINE",
                    "Snippet": "Longer source text should not create a duplicate URL.",
                    "Prime Example": 0,
                },
                {
                    "Group ID": "B",
                    "Headline": "Story B",
                    "Date": "2026-09-01",
                    "Mentions": 1,
                    "Impressions": 900,
                    "Effective Reach": 450,
                    "Outlet": "Outlet 3",
                    "URL": "https://b.test",
                    "Type": "ONLINE",
                    "Snippet": "Wrong group",
                    "Prime Example": 1,
                },
            ]
        )

        source_options = build_validation_source_candidate_table(
            source,
            source_group_ids="A",
            fallback_group_id="A",
        )

        self.assertEqual(source_options["Group ID"].astype(str).tolist(), ["A"])
        self.assertEqual(source_options["Example URL"].tolist(), ["https://same.test"])

    def test_rotation_changes_representative_source_but_preserves_metrics(self) -> None:
        saved = pd.DataFrame(
            [
                {
                    "Group ID": "A",
                    "Headline": "Prime headline",
                    "Date": "2026-09-01",
                    "Mentions": 9,
                    "Impressions": 300,
                    "Effective Reach": 125,
                    "Example Outlet": "Outlet 1",
                    "Example URL": "https://a.test",
                    "Example Type": "ONLINE",
                    "Example Snippet": "Prime snippet",
                    "Source Group IDs": "A",
                    "Top Story Summary": "Old summary",
                    "Chart Callout": "Old callout",
                    "Entity Sentiment": "Old sentiment",
                }
            ]
        )
        source = pd.DataFrame(
            [
                {
                    "Group ID": "A",
                    "Headline": "Prime headline",
                    "Date": "2026-09-01",
                    "Mentions": 4,
                    "Impressions": 100,
                    "Effective Reach": 50,
                    "Outlet": "Outlet 1",
                    "URL": "https://a.test",
                    "Type": "ONLINE",
                    "Snippet": "Prime snippet",
                    "Prime Example": 1,
                },
                {
                    "Group ID": "A",
                    "Headline": "Alternate headline",
                    "Date": "2026-09-02",
                    "Mentions": 5,
                    "Impressions": 200,
                    "Effective Reach": 75,
                    "Outlet": "Outlet 2",
                    "URL": "https://a2.test",
                    "Type": "ONLINE",
                    "Snippet": "Alternate snippet",
                    "Prime Example": 0,
                },
            ]
        )

        rotated = rotate_saved_story_source(saved, source, "A", step=1)
        row = rotated.iloc[0]

        self.assertEqual(row["Group ID"], "A")
        self.assertEqual(int(row["Mentions"]), 9)
        self.assertEqual(int(row["Impressions"]), 300)
        self.assertEqual(int(row["Effective Reach"]), 125)
        self.assertEqual(row["Headline"], "Alternate headline")
        self.assertEqual(row["Example Outlet"], "Outlet 2")
        self.assertEqual(row["Example URL"], "https://a2.test")
        self.assertEqual(row["Example Snippet"], "Alternate snippet")
        self.assertEqual(row["Top Story Summary"], "")
        self.assertEqual(row["Chart Callout"], "")
        self.assertEqual(row["Entity Sentiment"], "")

    def test_legacy_topmerge_source_ids_remain_readable_for_validation(self) -> None:
        source = pd.DataFrame(
            [
                {
                    "Group ID": "A",
                    "Headline": "Story A",
                    "Date": "2026-09-01",
                    "Mentions": 1,
                    "Impressions": 100,
                    "Effective Reach": 50,
                    "Outlet": "Outlet 1",
                    "URL": "https://a.test",
                    "Type": "ONLINE",
                    "Snippet": "A",
                    "Prime Example": 1,
                },
                {
                    "Group ID": "B",
                    "Headline": "Story B",
                    "Date": "2026-09-01",
                    "Mentions": 1,
                    "Impressions": 200,
                    "Effective Reach": 75,
                    "Outlet": "Outlet 2",
                    "URL": "https://b.test",
                    "Type": "ONLINE",
                    "Snippet": "B",
                    "Prime Example": 1,
                },
            ]
        )

        source_options = build_validation_source_candidate_table(
            source,
            source_group_ids="A | B",
            fallback_group_id="TOPMERGE::legacy",
        )

        self.assertEqual(source_options["Group ID"].astype(str).tolist(), ["B", "A"])


if __name__ == "__main__":
    unittest.main()
