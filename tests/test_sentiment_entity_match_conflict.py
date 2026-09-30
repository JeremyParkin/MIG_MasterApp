from __future__ import annotations

from types import SimpleNamespace
import unittest

import pandas as pd

from processing.ai_sentiment import (
    annotate_entity_match_conflict,
    apply_sentiment_result_to_unique_df,
    build_final_sentiment_series,
    build_sentiment_distribution,
    initialize_sentiment_workflow_columns,
)
from processing.sentiment_config import build_sentiment_configuration
from processing.spot_checks import (
    apply_review_flags_to_group,
    auto_assign_resolved_match_to_group,
    compute_candidates,
    ensure_review_columns,
    write_review_opinion_to_group,
)


class SentimentEntityMatchConflictTests(unittest.TestCase):
    def test_lexical_overlap_preserves_not_relevant_and_flags_conflict(self) -> None:
        result = {
            "sentiment": "NOT RELEVANT",
            "confidence": 91,
            "explanation": "The story concerns a different Illinois organization.",
        }

        annotated = annotate_entity_match_conflict(
            result,
            headline="More people try to get into prison college classes",
            snippet="The Illinois Community College Board is mentioned in the program background.",
            entity_terms=["College Board"],
        )

        self.assertEqual(annotated["sentiment"], "NOT RELEVANT")
        self.assertEqual(annotated["confidence"], 91)
        self.assertEqual(annotated["explanation"], result["explanation"])
        self.assertEqual(annotated["entity_match_conflict"], "Yes")
        self.assertIn("different organization", annotated["entity_match_conflict_reason"])

    def test_genuine_direct_mention_preserves_not_relevant_and_flags_conflict(self) -> None:
        result = {
            "sentiment": "NOT RELEVANT",
            "confidence": 76,
            "explanation": "The model judged the mention irrelevant.",
        }

        annotated = annotate_entity_match_conflict(
            result,
            headline="College Board announces testing dates",
            snippet="The College Board published an updated calendar.",
            entity_terms=["College Board"],
        )

        self.assertEqual(annotated["sentiment"], "NOT RELEVANT")
        self.assertEqual(annotated["confidence"], 76)
        self.assertEqual(annotated["entity_match_conflict"], "Yes")

    def test_first_pass_conflict_is_written_and_prioritized_for_second_opinion(self) -> None:
        unique = initialize_sentiment_workflow_columns(
            pd.DataFrame(
                {
                    "Group ID": [1, 2],
                    "Headline": ["Conflict", "High reach negative"],
                    "Group Count": [1, 50],
                    "Mentions": [1, 500],
                    "Impressions": [1, 100000],
                    "Effective Reach": [1, 100000],
                }
            )
        )
        grouped = unique.copy()

        conflict_result = annotate_entity_match_conflict(
            {
                "sentiment": "NOT RELEVANT",
                "confidence": 95,
                "explanation": "Different organization.",
            },
            headline="",
            snippet="Illinois Community College Board",
            entity_terms=["College Board"],
        )
        unique = apply_sentiment_result_to_unique_df(unique, 0, conflict_result)
        unique = apply_sentiment_result_to_unique_df(
            unique,
            1,
            {"sentiment": "NEGATIVE", "confidence": 95, "explanation": "Criticism."},
        )

        candidates = compute_candidates(unique, grouped, sentiment_type="3-way")

        self.assertEqual(unique.loc[0, "AI Sentiment"], "NOT RELEVANT")
        self.assertEqual(unique.loc[0, "AI Entity Match Conflict"], "Yes")
        self.assertEqual(candidates.loc[0, "Group ID"], 1)

    def test_matching_second_opinion_not_relevant_can_resolve_without_human_review(self) -> None:
        unique = initialize_sentiment_workflow_columns(
            pd.DataFrame({"Group ID": [1], "AI Sentiment": ["NOT RELEVANT"], "AI Sentiment Confidence": [92]})
        )
        grouped = unique.copy()
        unique, grouped = ensure_review_columns(unique, grouped)

        review_result = annotate_entity_match_conflict(
            {
                "sentiment": "NOT RELEVANT",
                "confidence": 88,
                "explanation": "The story is about another organization.",
            },
            headline="",
            snippet="Illinois Community College Board",
            entity_terms=["College Board"],
        )

        unique, grouped = write_review_opinion_to_group(
            unique,
            grouped,
            1,
            review_result["sentiment"],
            review_result["confidence"],
            review_result["explanation"],
            review_result["entity_match_conflict"],
            review_result["entity_match_conflict_reason"],
        )
        unique, grouped = apply_review_flags_to_group(
            unique,
            grouped,
            1,
            ai_label="NOT RELEVANT",
            ai_confidence=92,
            review_label="NOT RELEVANT",
            review_confidence=88,
            low_conf_threshold=65,
        )
        unique, grouped, auto_resolved = auto_assign_resolved_match_to_group(
            unique,
            grouped,
            1,
            ai_label="NOT RELEVANT",
            review_label="NOT RELEVANT",
            review_confidence=88,
            confidence_threshold=65,
        )

        self.assertEqual(unique.loc[0, "Review Entity Match Conflict"], "Yes")
        self.assertEqual(unique.loc[0, "Needs Human Review"], "No")
        self.assertTrue(auto_resolved)
        self.assertEqual(unique.loc[0, "Assigned Sentiment"], "NOT RELEVANT")

    def test_semantic_disagreement_still_needs_human_review(self) -> None:
        unique = initialize_sentiment_workflow_columns(
            pd.DataFrame({"Group ID": [1], "AI Sentiment": ["NOT RELEVANT"], "AI Sentiment Confidence": [92]})
        )
        grouped = unique.copy()
        unique, grouped = apply_review_flags_to_group(
            unique,
            grouped,
            1,
            ai_label="NOT RELEVANT",
            ai_confidence=92,
            review_label="NEUTRAL",
            review_confidence=90,
            low_conf_threshold=65,
        )

        self.assertEqual(unique.loc[0, "AI Agreement"], "Disagree")
        self.assertEqual(unique.loc[0, "Needs Human Review"], "Yes")

    def test_not_relevant_final_sentiment_is_tolerated_by_distribution(self) -> None:
        unique = initialize_sentiment_workflow_columns(pd.DataFrame({"Group ID": [1, 2], "Group Count": [1, 2]}))
        unique["AI Sentiment"] = ["NOT RELEVANT", "NEUTRAL"]
        unique["AI Sentiment Confidence"] = [90, 80]

        final = build_final_sentiment_series(unique)
        dist = build_sentiment_distribution(unique, "3-way", final_series=final)

        self.assertIn("NOT RELEVANT", dist["Sentiment"].tolist())
        self.assertEqual(dist.loc[dist["Sentiment"] == "NOT RELEVANT", "Count"].iloc[0], 1)

    def test_prompt_clarifies_lexical_overlap_inside_different_names(self) -> None:
        state = SimpleNamespace()

        build_sentiment_configuration(
            state,
            primary_names=["College Board"],
            alternate_names=[],
            spokespeople=[],
            products=[],
            highlight_keywords=[],
            shared_guidance="",
            sentiment_guidance="",
            sentiment_type="3-way",
            model="test-model",
        )

        self.assertIn("different organization, product, or proper name", state.post_prompt)
        self.assertIn("different organization, product, or proper name", state.sentiment_instruction)

    def test_prompt_clarifies_configured_products_are_relevant_without_parent_name(self) -> None:
        state = SimpleNamespace()

        build_sentiment_configuration(
            state,
            primary_names=["Example Parent"],
            alternate_names=["Example Alias"],
            spokespeople=["Example Spokesperson"],
            products=["Example Program"],
            highlight_keywords=[],
            shared_guidance="",
            sentiment_guidance="",
            sentiment_type="3-way",
            model="test-model",
        )

        self.assertIn("parent organization does not need to be explicitly named", state.post_prompt)
        self.assertIn("substantive coverage of any configured alias", state.post_prompt)
        self.assertIn("parent organization does not need to be explicitly named", state.sentiment_instruction)
        self.assertIn("product, sub-brand, or program is in scope", state.sentiment_instruction)


if __name__ == "__main__":
    unittest.main()
