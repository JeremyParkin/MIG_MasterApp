from __future__ import annotations

import io
import unittest
from pathlib import Path

import dill
import pandas as pd

from processing.ai_tagging import RESERVED_OTHER_TAG
from processing.rapid_second_opinion import (
    apply_rapid_review_result_to_df,
    build_rapid_review_candidates,
    build_rapid_review_prompt,
    build_rapid_review_schema,
    derive_five_way_from_score,
    derive_three_way_from_score,
    ensure_rapid_review_columns,
    normalize_applicable_tags,
    normalize_first_pass_sentiment,
    normalize_sentiment_outcome,
    parse_rapid_review_response,
    resolve_rapid_review_batch_size,
    run_rapid_review_batch,
)
from utils.session_snapshot import build_serializable_session_payload, load_session_state_from_file


class State(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


class RapidSecondOpinionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.analysis_payload = {
            "primary_name": "College Board",
            "alternate_names": ["The College Board"],
            "spokespeople": ["Example Person"],
            "products": ["SAT", "AP"],
            "general_guidance": "Separate the entity from the broader testing topic.",
            "sentiment_guidance": "Student recognition may be favorable when credited to the entity.",
        }
        self.tags = {
            "Programs": "Coverage about College Board programs.",
            "Policy": "Coverage about policy or institutional decisions.",
        }

    def make_row(self, **overrides) -> dict:
        base = {
            "Group ID": 10,
            "Headline": "Students recognized by College Board",
            "Snippet": "The College Board recognized students through its programs.",
            "Group Count": 1,
            "Effective Reach": 100,
            "Mentions": 1,
            "Impressions": 100,
            "Jev Sentiment": "POSITIVE",
            "Jev Selected Probability": 0.82,
            "Jev Probability POSITIVE": 0.82,
            "Jev Probability NEUTRAL": 0.12,
            "Jev Probability NEGATIVE": 0.03,
            "Jev Probability NOT RELEVANT": 0.03,
            "Jev 5-Way Sentiment": "SOMEWHAT POSITIVE",
            "Jev 5-Way Selected Probability": 0.74,
            "Jev 5-Way Probability VERY POSITIVE": 0.10,
            "Jev 5-Way Probability SOMEWHAT POSITIVE": 0.74,
            "Jev 5-Way Probability NEUTRAL": 0.12,
            "Jev 5-Way Probability SOMEWHAT NEGATIVE": 0.02,
            "Jev 5-Way Probability VERY NEGATIVE": 0.01,
            "Jev 5-Way Probability NOT RELEVANT": 0.01,
            "Jev Sentiment Score": 3.0,
            "Jev Score Confidence": 0.75,
            "Jev Score Relevant Probability": 0.95,
            "Jev Mixture Label": "VERY CONSISTENT",
            "Jev Mixture Score": 0.0,
            "Jev Best Tag": "Programs",
            "Jev Best Tag Selected Probability": 0.80,
            "Jev Tags": "Programs",
            "Jev Tag Count": 1,
            "Jev Tag [Programs]": "YES",
            "Jev Tag Probability [Programs]": 0.86,
            "Jev Tag Confidence [Programs]": 0.82,
            "Jev Tag [Policy]": "NO",
            "Jev Tag Probability [Policy]": 0.20,
            "Jev Tag Confidence [Policy]": 0.75,
            "Jev Error": "",
        }
        base.update(overrides)
        return base

    def test_anchor_score_mappings(self) -> None:
        expected = {
            -7: ("NEGATIVE", "VERY NEGATIVE"),
            -5: ("NEGATIVE", "VERY NEGATIVE"),
            -3: ("NEGATIVE", "SOMEWHAT NEGATIVE"),
            -1: ("NEUTRAL", "NEUTRAL"),
            0: ("NEUTRAL", "NEUTRAL"),
            1: ("NEUTRAL", "NEUTRAL"),
            3: ("POSITIVE", "SOMEWHAT POSITIVE"),
            5: ("POSITIVE", "VERY POSITIVE"),
            7: ("POSITIVE", "VERY POSITIVE"),
        }
        for score, labels in expected.items():
            self.assertEqual(derive_three_way_from_score(score), labels[0])
            self.assertEqual(derive_five_way_from_score(score), labels[1])

    def test_fractional_midpoint_boundaries(self) -> None:
        self.assertEqual(derive_three_way_from_score(-2), "NEGATIVE")
        self.assertEqual(derive_three_way_from_score(-1.99), "NEUTRAL")
        self.assertEqual(derive_three_way_from_score(1.99), "NEUTRAL")
        self.assertEqual(derive_three_way_from_score(2), "POSITIVE")

        self.assertEqual(derive_five_way_from_score(-4), "VERY NEGATIVE")
        self.assertEqual(derive_five_way_from_score(-3.99), "SOMEWHAT NEGATIVE")
        self.assertEqual(derive_five_way_from_score(-2), "SOMEWHAT NEGATIVE")
        self.assertEqual(derive_five_way_from_score(-1.99), "NEUTRAL")
        self.assertEqual(derive_five_way_from_score(2), "SOMEWHAT POSITIVE")
        self.assertEqual(derive_five_way_from_score(4), "VERY POSITIVE")

    def test_first_pass_not_relevant_nulls_comparable_score(self) -> None:
        row = self.make_row(
            **{
                "Jev Sentiment": "NOT RELEVANT",
                "Jev 5-Way Sentiment": "NOT RELEVANT",
                "Jev Sentiment Score": -7,
                "Jev Score Relevant Probability": 0.95,
            }
        )
        normalized = normalize_first_pass_sentiment(row)
        self.assertEqual(normalized["outcome"], "NOT_RELEVANT")
        self.assertIsNone(normalized["score"])
        self.assertEqual(normalized["derived_3_way"], "NOT RELEVANT")

    def test_first_pass_relevance_probability_does_not_decide_canonical_relevance(self) -> None:
        row = self.make_row(**{"Jev Score Relevant Probability": 0.01})
        normalized = normalize_first_pass_sentiment(row)
        self.assertEqual(normalized["outcome"], "SCORE")
        self.assertEqual(normalized["score"], 3.0)

    def test_mixed_not_relevant_categoricals_mark_contradiction(self) -> None:
        row = self.make_row(**{"Jev Sentiment": "NOT RELEVANT", "Jev 5-Way Sentiment": "NEUTRAL"})
        normalized = normalize_first_pass_sentiment(row)
        self.assertEqual(normalized["outcome"], "CONTRADICTION")
        self.assertTrue(normalized["internal_contradiction"])

    def test_sentiment_outcome_validation(self) -> None:
        self.assertEqual(normalize_sentiment_outcome("NOT_RELEVANT"), ("NOT_RELEVANT", None))
        self.assertEqual(normalize_sentiment_outcome("+5"), ("5", 5.0))
        with self.assertRaises(ValueError):
            normalize_sentiment_outcome("2")

    def test_schema_includes_anchored_outcome_and_tags(self) -> None:
        schema = build_rapid_review_schema(self.tags)[0]["parameters"]
        self.assertIn("NOT_RELEVANT", schema["properties"]["sentiment_outcome"]["enum"])
        self.assertIn("-7", schema["properties"]["sentiment_outcome"]["enum"])
        self.assertIn("Programs", schema["properties"]["best_tag"]["enum"])
        self.assertIn(RESERVED_OTHER_TAG, schema["properties"]["best_tag"]["enum"])

    def test_prompt_clarifies_configured_products_are_relevant_without_parent_name(self) -> None:
        prompt = build_rapid_review_prompt(
            self.make_row(**{"Headline": "SAT changes draw student attention"}),
            self.analysis_payload,
            tag_definitions=self.tags,
        )

        self.assertIn("parent organization does not need to be explicitly named", prompt)
        self.assertIn("substantive coverage of any configured member", prompt)
        self.assertIn("passing or incidental mention", prompt)
        self.assertIn("Use NOT_RELEVANT only when", prompt)

    def test_parse_review_response_normalizes_other_and_tags(self) -> None:
        parsed = parse_rapid_review_response(
            {
                "sentiment_outcome": "NOT_RELEVANT",
                "sentiment_confidence": 84,
                "sentiment_rationale": "No monitored entity.",
                "best_tag": "Other",
                "applicable_tags": [],
                "tag_confidence": 75,
                "tag_rationale": "No explicit tag applies.",
            },
            tag_definitions=self.tags,
            tagging_mode="Multiple applicable tags",
        )
        self.assertEqual(parsed["sentiment_score"], None)
        self.assertEqual(parsed["review_3_way"], "NOT RELEVANT")
        self.assertEqual(parsed["official_tags"], [RESERVED_OTHER_TAG])

    def test_invalid_tag_rejected_and_other_plus_explicit_normalized(self) -> None:
        with self.assertRaises(ValueError):
            parse_rapid_review_response(
                {
                    "sentiment_outcome": "3",
                    "sentiment_confidence": 84,
                    "sentiment_rationale": "Moderately favorable.",
                    "best_tag": "Unknown",
                    "applicable_tags": ["Programs"],
                    "tag_confidence": 75,
                    "tag_rationale": "Program coverage.",
                },
                tag_definitions=self.tags,
            )
        allowed = {**self.tags, RESERVED_OTHER_TAG: "Fallback"}
        self.assertEqual(normalize_applicable_tags(["Other", "Programs"], allowed), ["Programs"])

    def test_candidate_selection_prioritizes_contradiction_and_negative_alone_not_special(self) -> None:
        df = pd.DataFrame(
            [
                self.make_row(**{"Group ID": 1, "Jev Sentiment": "NEGATIVE", "Jev 5-Way Sentiment": "SOMEWHAT NEGATIVE", "Jev Sentiment Score": -3}),
                self.make_row(**{"Group ID": 2, "Jev Sentiment": "POSITIVE", "Jev 5-Way Sentiment": "SOMEWHAT POSITIVE", "Jev Sentiment Score": -3}),
            ]
        )
        candidates = build_rapid_review_candidates(df, tag_definitions=self.tags)
        self.assertEqual(int(candidates.iloc[0]["Group ID"]), 2)
        neg_reason = candidates.loc[candidates["Group ID"] == 1, "Rapid Review Priority Reason"].iloc[0]
        self.assertNotIn("negative +", neg_reason)

    def test_negative_plus_high_impact_exposes_contextual_reason(self) -> None:
        df = pd.DataFrame(
            [
                self.make_row(**{"Group ID": 1, "Jev Sentiment": "NEGATIVE", "Jev 5-Way Sentiment": "SOMEWHAT NEGATIVE", "Jev Sentiment Score": -3, "Group Count": 1}),
                self.make_row(**{"Group ID": 2, "Jev Sentiment": "NEGATIVE", "Jev 5-Way Sentiment": "SOMEWHAT NEGATIVE", "Jev Sentiment Score": -3, "Group Count": 20}),
            ]
        )
        candidates = build_rapid_review_candidates(df, tag_definitions=self.tags)
        reason = candidates.loc[candidates["Group ID"] == 2, "Rapid Review Priority Reason"].iloc[0]
        self.assertIn("high Group Count", reason)
        self.assertIn("negative + high impact", reason)

    def test_already_reviewed_excluded(self) -> None:
        df = pd.DataFrame(
            [
                self.make_row(**{"Group ID": 1, "Rapid Review Status": "Completed"}),
                self.make_row(**{"Group ID": 2}),
            ]
        )
        candidates = build_rapid_review_candidates(df, tag_definitions=self.tags)
        self.assertEqual(candidates["Group ID"].tolist(), [2])

    def test_every_valid_first_pass_story_enters_available_pool(self) -> None:
        df = pd.DataFrame(
            [
                self.make_row(**{"Group ID": 1}),
                self.make_row(**{"Group ID": 2, "Jev Sentiment": "POSITIVE", "Jev 5-Way Sentiment": "SOMEWHAT POSITIVE", "Jev Sentiment Score": -3}),
                self.make_row(**{"Group ID": 3, "Group Count": 20}),
            ]
        )
        candidates = build_rapid_review_candidates(df, tag_definitions=self.tags)
        self.assertEqual(set(candidates["Group ID"].tolist()), {1, 2, 3})

    def test_routine_no_trigger_story_remains_available(self) -> None:
        df = pd.DataFrame([self.make_row(**{"Group ID": 1})])
        candidates = build_rapid_review_candidates(df, tag_definitions=self.tags)
        self.assertEqual(candidates["Group ID"].tolist(), [1])
        self.assertEqual(candidates.loc[0, "Rapid Review Priority Tier"], "Tier 5 - Routine")
        self.assertEqual(candidates.loc[0, "Rapid Review Priority Reason"], "routine processed story")

    def test_priority_reasons_affect_ordering_not_eligibility(self) -> None:
        df = pd.DataFrame(
            [
                self.make_row(**{"Group ID": 1}),
                self.make_row(**{"Group ID": 2, "Jev Sentiment": "POSITIVE", "Jev 5-Way Sentiment": "SOMEWHAT POSITIVE", "Jev Sentiment Score": -3}),
            ]
        )
        candidates = build_rapid_review_candidates(df, tag_definitions=self.tags)
        self.assertEqual(candidates["Group ID"].tolist(), [2, 1])
        self.assertEqual(len(candidates), 2)

    def test_failed_and_unusable_first_pass_rows_excluded(self) -> None:
        df = pd.DataFrame(
            [
                self.make_row(**{"Group ID": 1}),
                self.make_row(**{"Group ID": 2, "Jev Error": "OpenRouter failed"}),
                self.make_row(**{"Group ID": 3, "Jev Sentiment Score": ""}),
            ]
        )
        candidates = build_rapid_review_candidates(df, tag_definitions=self.tags)
        self.assertEqual(candidates["Group ID"].tolist(), [1])

    def test_apply_review_result_computes_agreements_and_distance(self) -> None:
        df = ensure_rapid_review_columns(pd.DataFrame([self.make_row()]))
        result = {
            "parsed": {
                "sentiment_outcome": "3",
                "sentiment_score": 3.0,
                "sentiment_confidence": 0.8,
                "sentiment_rationale": "Favorable.",
                "review_3_way": "POSITIVE",
                "review_5_way": "SOMEWHAT POSITIVE",
                "best_tag": "Programs",
                "official_tags": ["Programs"],
                "tag_confidence": 0.7,
                "tag_rationale": "Program coverage.",
            },
            "input_tokens": 100,
            "output_tokens": 20,
            "cost_usd": 0.0001,
            "raw_response": "{}",
            "model": "gpt-5.6-luna",
        }
        updated = apply_rapid_review_result_to_df(df, 0, result, tag_definitions=self.tags)
        self.assertEqual(updated.loc[0, "Rapid Review Status"], "Completed")
        self.assertEqual(updated.loc[0, "Rapid Sentiment 3-Way Agreement"], "Match")
        self.assertEqual(updated.loc[0, "Rapid Sentiment Score Distance"], 0.0)
        self.assertEqual(updated.loc[0, "Rapid Tag Best-Fit Agreement"], "Match")

    def test_run_batch_one_call_per_candidate_and_mixed_error(self) -> None:
        df = pd.DataFrame(
            [
                self.make_row(**{"Group ID": 1}),
                self.make_row(**{"Group ID": 2, "Headline": "Bad row"}),
            ]
        )
        candidates = build_rapid_review_candidates(df, tag_definitions=self.tags)
        calls = []

        def fake_call(prompt, api_key, *, model, tag_definitions):
            calls.append(prompt)
            if "Bad row" in prompt:
                raise ValueError("boom")
            return (
                {
                    "sentiment_outcome": "3",
                    "sentiment_confidence": 80,
                    "sentiment_rationale": "Favorable.",
                    "best_tag": "Programs",
                    "applicable_tags": ["Programs"],
                    "tag_confidence": 70,
                    "tag_rationale": "Program coverage.",
                },
                100,
                20,
                "{}",
            )

        updated, summary = run_rapid_review_batch(
            df,
            candidates,
            self.analysis_payload,
            "test-key",
            tag_definitions=self.tags,
            limit=2,
            max_workers=2,
            call_fn=fake_call,
        )
        self.assertEqual(len(calls), 4)  # one success + failing row initial attempt and 2 retries
        self.assertEqual(summary["successful"], 1)
        self.assertEqual(len(summary["errors"]), 1)
        self.assertEqual(updated.loc[0, "Rapid Review Status"], "Completed")
        self.assertEqual(updated.loc[1, "Rapid Review Status"], "Error")

    def test_failed_second_opinion_row_remains_retryable(self) -> None:
        df = pd.DataFrame(
            [
                self.make_row(**{"Group ID": 1}),
                self.make_row(**{"Group ID": 2, "Headline": "Bad row"}),
            ]
        )
        candidates = build_rapid_review_candidates(df, tag_definitions=self.tags)

        def fake_call(prompt, api_key, *, model, tag_definitions):
            if "Bad row" in prompt:
                raise ValueError("boom")
            return (
                {
                    "sentiment_outcome": "3",
                    "sentiment_confidence": 80,
                    "sentiment_rationale": "Favorable.",
                    "best_tag": "Programs",
                    "applicable_tags": ["Programs"],
                    "tag_confidence": 70,
                    "tag_rationale": "Program coverage.",
                },
                100,
                20,
                "{}",
            )

        updated, _ = run_rapid_review_batch(
            df,
            candidates,
            self.analysis_payload,
            "test-key",
            tag_definitions=self.tags,
            limit=2,
            max_workers=1,
            call_fn=fake_call,
        )
        retry_candidates = build_rapid_review_candidates(updated, tag_definitions=self.tags)
        self.assertEqual(retry_candidates["Group ID"].tolist(), [2])

    def test_successive_batches_exhaust_the_full_priority_sorted_pool(self) -> None:
        df = pd.DataFrame(
            [
                self.make_row(**{"Group ID": 1, "Headline": "Routine one"}),
                self.make_row(
                    **{
                        "Group ID": 2,
                        "Headline": "Priority two",
                        "Jev Sentiment": "POSITIVE",
                        "Jev 5-Way Sentiment": "SOMEWHAT POSITIVE",
                        "Jev Sentiment Score": -3,
                    }
                ),
                self.make_row(**{"Group ID": 3, "Headline": "Routine three"}),
            ]
        )
        calls: list[str] = []

        def fake_call(prompt, api_key, *, model, tag_definitions):
            calls.append(prompt)
            return (
                {
                    "sentiment_outcome": "3",
                    "sentiment_confidence": 80,
                    "sentiment_rationale": "Favorable.",
                    "best_tag": "Programs",
                    "applicable_tags": ["Programs"],
                    "tag_confidence": 70,
                    "tag_rationale": "Program coverage.",
                },
                100,
                20,
                "{}",
            )

        first_candidates = build_rapid_review_candidates(df, tag_definitions=self.tags)
        self.assertEqual(first_candidates["Group ID"].tolist(), [2, 1, 3])
        updated, first_summary = run_rapid_review_batch(
            df,
            first_candidates,
            self.analysis_payload,
            "test-key",
            tag_definitions=self.tags,
            limit=2,
            max_workers=1,
            call_fn=fake_call,
        )
        self.assertEqual(first_summary["successful"], 2)
        self.assertIn("Priority two", calls[0])
        self.assertEqual(build_rapid_review_candidates(updated, tag_definitions=self.tags)["Group ID"].tolist(), [3])

        updated, second_summary = run_rapid_review_batch(
            updated,
            build_rapid_review_candidates(updated, tag_definitions=self.tags),
            self.analysis_payload,
            "test-key",
            tag_definitions=self.tags,
            limit=2,
            max_workers=1,
            call_fn=fake_call,
        )
        self.assertEqual(second_summary["successful"], 1)
        self.assertTrue(build_rapid_review_candidates(updated, tag_definitions=self.tags).empty)

    def test_batch_size_defaults_to_fifty_and_clamps_to_remaining(self) -> None:
        self.assertEqual(resolve_rapid_review_batch_size(None, 120), 50)
        self.assertEqual(resolve_rapid_review_batch_size(50, 12), 12)
        self.assertEqual(resolve_rapid_review_batch_size(100, 75), 75)
        self.assertEqual(resolve_rapid_review_batch_size(0, 75), 50)
        self.assertEqual(resolve_rapid_review_batch_size(100, 0), 0)

    def test_manual_batch_size_persists_through_snapshot_round_trip(self) -> None:
        state = State(rapid_review_batch_size=100)
        payload, skipped = build_serializable_session_payload(state)
        self.assertEqual(skipped, [])
        restored = State()
        load_session_state_from_file(restored, io.BytesIO(dill.dumps(payload)))
        self.assertEqual(restored.rapid_review_batch_size, 100)
        self.assertEqual(resolve_rapid_review_batch_size(restored.rapid_review_batch_size, 120), 100)

    def test_recommendation_helpers_and_ui_are_removed(self) -> None:
        import processing.rapid_second_opinion as rapid_review

        self.assertFalse(hasattr(rapid_review, "recommend_rapid_review_batch_size"))
        self.assertFalse(hasattr(rapid_review, "resolve_rapid_review_recommendation"))
        page_source = (Path(__file__).resolve().parents[1] / "pages" / "Jev_Sentiment_Experimental.py").read_text()
        self.assertNotIn("Recommended remaining batch", page_source)


if __name__ == "__main__":
    unittest.main()
