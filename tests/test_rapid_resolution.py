from __future__ import annotations

import unittest
import math

import pandas as pd

from processing.rapid_resolution import (
    HUMAN_SENTIMENT_SCALE_3_WAY,
    HUMAN_SENTIMENT_SCALE_5_WAY,
    HUMAN_STATE_ACCEPTED,
    HUMAN_STATE_ASSIGNED,
    RELEVANCE_CONFLICT,
    RELEVANCE_NOT_RELEVANT,
    RELEVANCE_RELEVANT,
    RAPID_TAG_REVIEW_MODE_APPLICABLE,
    RAPID_TAG_REVIEW_MODE_BEST,
    SOURCE_HUMAN_ACCEPTED_MACHINE,
    SOURCE_HUMAN_ASSIGNED,
    SOURCE_RAPID_FIRST_PASS,
    STATUS_AGREEMENT,
    STATUS_DISAGREEMENT,
    STATUS_FIRST_PASS_ONLY,
    STATUS_INTERNAL_CONFLICT,
    STATUS_INVALID,
    STATUS_PARTIAL_AGREEMENT,
    STATUS_SECOND_OPINION_ERROR,
    apply_rapid_sentiment_human_selection,
    apply_rapid_tag_human_selection,
    build_effective_rapid_sentiment_frame,
    build_effective_rapid_tag_frame,
    build_final_rapid_sentiment_frame,
    build_final_rapid_tag_frame,
    build_rapid_highlight_keywords,
    ensure_rapid_human_review_columns,
    format_rapid_numeric,
    resolve_final_rapid_sentiment_row,
    resolve_final_rapid_tag_row,
    resolve_rapid_sentiment_row,
    resolve_rapid_tag_row,
    summarize_rapid_label_provenance,
)


class RapidResolutionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tags = {
            "AP": "Advanced Placement coverage.",
            "SAT": "SAT or PSAT coverage.",
            "Career": "Career planning or readiness coverage.",
        }

    def row(self, **overrides) -> dict:
        base = {
            "Group ID": 1,
            "Jev Sentiment": "NEUTRAL",
            "Jev 5-Way Sentiment": "NEUTRAL",
            "Jev Sentiment Score": 0.2,
            "Jev Best Tag": "SAT",
            "Jev Tags": "SAT",
            "Rapid Review Status": "",
            "Rapid Review Error": "",
        }
        base.update(overrides)
        return base

    def test_both_jev_categories_not_relevant_nulls_effective_score(self) -> None:
        resolved = resolve_rapid_sentiment_row(
            self.row(
                **{
                    "Jev Sentiment": "NOT RELEVANT",
                    "Jev 5-Way Sentiment": "NOT RELEVANT",
                    "Jev Sentiment Score": -7,
                }
            )
        )
        self.assertEqual(resolved["Effective Rapid Relevance"], RELEVANCE_NOT_RELEVANT)
        self.assertEqual(resolved["Effective Rapid Sentiment 3-Way"], "NOT RELEVANT")
        self.assertTrue(pd.isna(resolved["Effective Rapid Sentiment Score"]))
        self.assertEqual(resolved["Rapid Sentiment Resolution Status"], STATUS_FIRST_PASS_ONLY)

    def test_neither_jev_category_not_relevant_retains_raw_categories(self) -> None:
        resolved = resolve_rapid_sentiment_row(
            self.row(
                **{
                    "Jev Sentiment": "POSITIVE",
                    "Jev 5-Way Sentiment": "SOMEWHAT POSITIVE",
                    "Jev Sentiment Score": 0.9,
                }
            )
        )
        self.assertEqual(resolved["Effective Rapid Relevance"], RELEVANCE_RELEVANT)
        self.assertEqual(resolved["Effective Rapid Sentiment 3-Way"], "POSITIVE")
        self.assertEqual(resolved["Effective Rapid Sentiment 5-Way"], "SOMEWHAT POSITIVE")
        self.assertEqual(resolved["Effective Rapid Sentiment Score"], 0.9)
        self.assertEqual(resolved["Effective Rapid Sentiment Source"], SOURCE_RAPID_FIRST_PASS)

    def test_one_not_relevant_one_sentiment_bearing_has_no_effective_sentiment(self) -> None:
        resolved = resolve_rapid_sentiment_row(
            self.row(
                **{
                    "Jev Sentiment": "NOT RELEVANT",
                    "Jev 5-Way Sentiment": "NEUTRAL",
                    "Jev Sentiment Score": 0.0,
                    "Rapid Review Status": "Completed",
                    "Rapid Review Sentiment Outcome": "0",
                    "Rapid Review 3-Way Sentiment": "NEUTRAL",
                    "Rapid Review 5-Way Sentiment": "NEUTRAL",
                }
            )
        )
        self.assertEqual(resolved["Effective Rapid Relevance"], RELEVANCE_CONFLICT)
        self.assertEqual(resolved["Effective Rapid Sentiment 3-Way"], "")
        self.assertEqual(resolved["Effective Rapid Sentiment 5-Way"], "")
        self.assertTrue(pd.isna(resolved["Effective Rapid Sentiment Score"]))
        self.assertEqual(resolved["Rapid Sentiment Resolution Status"], STATUS_INTERNAL_CONFLICT)

    def test_score_derived_disagreement_does_not_replace_raw_categories(self) -> None:
        resolved = resolve_rapid_sentiment_row(
            self.row(
                **{
                    "Jev Sentiment": "NEUTRAL",
                    "Jev 5-Way Sentiment": "NEUTRAL",
                    "Jev Sentiment Score": 2.9,
                }
            )
        )
        self.assertEqual(resolved["Effective Rapid Sentiment 3-Way"], "NEUTRAL")
        self.assertEqual(resolved["Effective Rapid Sentiment 5-Way"], "NEUTRAL")
        self.assertEqual(resolved["Effective Rapid Sentiment Score"], 2.9)

    def test_missing_required_jev_outputs_is_invalid(self) -> None:
        resolved = resolve_rapid_sentiment_row(self.row(**{"Jev Sentiment": ""}))
        self.assertEqual(resolved["Rapid Sentiment Resolution Status"], STATUS_INVALID)
        self.assertEqual(resolved["Effective Rapid Sentiment 3-Way"], "")

    def test_jev_only_status_first_pass_only(self) -> None:
        resolved = resolve_rapid_sentiment_row(self.row())
        self.assertEqual(resolved["Rapid Sentiment Resolution Status"], STATUS_FIRST_PASS_ONLY)

    def test_luna_full_agreement_status(self) -> None:
        resolved = resolve_rapid_sentiment_row(
            self.row(
                **{
                    "Rapid Review Status": "Completed",
                    "Rapid Review Sentiment Outcome": "0",
                    "Rapid Review Sentiment Score": 0,
                    "Rapid Review 3-Way Sentiment": "NEUTRAL",
                    "Rapid Review 5-Way Sentiment": "NEUTRAL",
                }
            )
        )
        self.assertEqual(resolved["Rapid Sentiment Resolution Status"], STATUS_AGREEMENT)
        self.assertEqual(resolved["Rapid Sentiment Score Distance"], 0.2)

    def test_luna_partial_categorical_agreement_status(self) -> None:
        resolved = resolve_rapid_sentiment_row(
            self.row(
                **{
                    "Jev Sentiment": "POSITIVE",
                    "Jev 5-Way Sentiment": "SOMEWHAT POSITIVE",
                    "Jev Sentiment Score": 2.3,
                    "Rapid Review Status": "Completed",
                    "Rapid Review Sentiment Outcome": "1",
                    "Rapid Review Sentiment Score": 1,
                    "Rapid Review 3-Way Sentiment": "NEUTRAL",
                    "Rapid Review 5-Way Sentiment": "SOMEWHAT POSITIVE",
                }
            )
        )
        self.assertEqual(resolved["Rapid Sentiment Resolution Status"], STATUS_PARTIAL_AGREEMENT)
        self.assertEqual(resolved["Effective Rapid Sentiment 3-Way"], "POSITIVE")

    def test_luna_relevance_disagreement_status(self) -> None:
        resolved = resolve_rapid_sentiment_row(
            self.row(
                **{
                    "Rapid Review Status": "Completed",
                    "Rapid Review Sentiment Outcome": "NOT_RELEVANT",
                    "Rapid Review 3-Way Sentiment": "NOT RELEVANT",
                    "Rapid Review 5-Way Sentiment": "NOT RELEVANT",
                }
            )
        )
        self.assertEqual(resolved["Rapid Sentiment Resolution Status"], STATUS_DISAGREEMENT)
        self.assertEqual(resolved["Effective Rapid Relevance"], RELEVANCE_RELEVANT)

    def test_luna_error_keeps_jev_effective_values(self) -> None:
        resolved = resolve_rapid_sentiment_row(
            self.row(**{"Rapid Review Status": "Error", "Rapid Review Error": "boom"})
        )
        self.assertEqual(resolved["Rapid Sentiment Resolution Status"], STATUS_SECOND_OPINION_ERROR)
        self.assertEqual(resolved["Effective Rapid Sentiment 3-Way"], "NEUTRAL")

    def test_tag_jev_only_uses_effective_jev_tags(self) -> None:
        resolved = resolve_rapid_tag_row(self.row(), tag_definitions=self.tags)
        self.assertEqual(resolved["Effective Rapid Best Tag"], "SAT")
        self.assertEqual(resolved["Effective Rapid Tags"], "SAT")
        self.assertEqual(resolved["Rapid Tag Resolution Status"], STATUS_FIRST_PASS_ONLY)

    def test_tag_order_normalization_agrees(self) -> None:
        resolved = resolve_rapid_tag_row(
            self.row(
                **{
                    "Jev Best Tag": "AP",
                    "Jev Tags": "AP; SAT",
                    "Rapid Review Status": "Completed",
                    "Rapid Review Best Tag": "AP",
                    "Rapid Review Tags": "SAT; AP",
                }
            ),
            tag_definitions=self.tags,
        )
        self.assertEqual(resolved["Rapid Tag Resolution Status"], STATUS_AGREEMENT)

    def test_tag_best_fit_only_agreement_is_partial(self) -> None:
        resolved = resolve_rapid_tag_row(
            self.row(
                **{
                    "Rapid Review Status": "Completed",
                    "Rapid Review Best Tag": "SAT",
                    "Rapid Review Tags": "SAT; AP",
                }
            ),
            tag_definitions=self.tags,
        )
        self.assertEqual(resolved["Rapid Tag Resolution Status"], STATUS_PARTIAL_AGREEMENT)

    def test_tag_set_only_agreement_is_partial(self) -> None:
        resolved = resolve_rapid_tag_row(
            self.row(
                **{
                    "Rapid Review Status": "Completed",
                    "Rapid Review Best Tag": "AP",
                    "Rapid Review Tags": "SAT",
                }
            ),
            tag_definitions=self.tags,
        )
        self.assertEqual(resolved["Rapid Tag Resolution Status"], STATUS_PARTIAL_AGREEMENT)

    def test_tag_both_disagree_status(self) -> None:
        resolved = resolve_rapid_tag_row(
            self.row(
                **{
                    "Rapid Review Status": "Completed",
                    "Rapid Review Best Tag": "AP",
                    "Rapid Review Tags": "AP",
                }
            ),
            tag_definitions=self.tags,
        )
        self.assertEqual(resolved["Rapid Tag Resolution Status"], STATUS_DISAGREEMENT)
        self.assertEqual(resolved["Effective Rapid Best Tag"], "SAT")

    def test_best_fit_not_in_applicable_set_is_internal_conflict(self) -> None:
        resolved = resolve_rapid_tag_row(
            self.row(**{"Jev Best Tag": "AP", "Jev Tags": "SAT"}),
            tag_definitions=self.tags,
        )
        self.assertEqual(resolved["Rapid Tag Resolution Status"], STATUS_INTERNAL_CONFLICT)
        self.assertEqual(resolved["Effective Rapid Best Tag"], "")

    def test_other_plus_explicit_tags_is_internal_conflict(self) -> None:
        resolved = resolve_rapid_tag_row(
            self.row(**{"Jev Best Tag": "AP", "Jev Tags": "Other; AP"}),
            tag_definitions=self.tags,
        )
        self.assertEqual(resolved["Rapid Tag Resolution Status"], STATUS_INTERNAL_CONFLICT)

    def test_tag_luna_error_preserves_jev_effective_tags(self) -> None:
        resolved = resolve_rapid_tag_row(
            self.row(**{"Rapid Review Status": "Error", "Rapid Review Error": "boom"}),
            tag_definitions=self.tags,
        )
        self.assertEqual(resolved["Rapid Tag Resolution Status"], STATUS_SECOND_OPINION_ERROR)
        self.assertEqual(resolved["Effective Rapid Best Tag"], "SAT")

    def test_mixed_rows_resolve_independently(self) -> None:
        df = pd.DataFrame(
            [
                self.row(**{"Group ID": 1}),
                self.row(**{"Group ID": 2, "Jev Sentiment": ""}),
                self.row(**{"Group ID": 3, "Jev Sentiment": "NOT RELEVANT", "Jev 5-Way Sentiment": "NOT RELEVANT"}),
            ]
        )
        resolved = build_effective_rapid_sentiment_frame(df)
        self.assertEqual(resolved.loc[0, "Rapid Sentiment Resolution Status"], STATUS_FIRST_PASS_ONLY)
        self.assertEqual(resolved.loc[1, "Rapid Sentiment Resolution Status"], STATUS_INVALID)
        self.assertEqual(resolved.loc[2, "Effective Rapid Relevance"], RELEVANCE_NOT_RELEVANT)

    def test_provenance_summary_counts_statuses(self) -> None:
        df = pd.DataFrame(
            [
                self.row(**{"Group ID": 1}),
                self.row(
                    **{
                        "Group ID": 2,
                        "Rapid Review Status": "Completed",
                        "Rapid Review Sentiment Outcome": "0",
                        "Rapid Review Sentiment Score": 0,
                        "Rapid Review 3-Way Sentiment": "NEUTRAL",
                        "Rapid Review 5-Way Sentiment": "NEUTRAL",
                        "Rapid Review Best Tag": "SAT",
                        "Rapid Review Tags": "SAT",
                    }
                ),
                self.row(
                    **{
                        "Group ID": 3,
                        "Jev Sentiment": "NOT RELEVANT",
                        "Jev 5-Way Sentiment": "NEUTRAL",
                    }
                ),
            ]
        )
        summary = summarize_rapid_label_provenance(df, tag_definitions=self.tags)
        self.assertEqual(summary["rapid_labeled_stories"], 3)
        self.assertEqual(summary["effective_sentiment_available"], 2)
        self.assertGreaterEqual(summary["first_pass_only"], 1)
        self.assertGreaterEqual(summary["cross_model_agreement"], 2)
        self.assertGreaterEqual(summary["first_pass_internal_conflict"], 1)

    def test_tag_frame_helper(self) -> None:
        df = pd.DataFrame([self.row()])
        resolved = build_effective_rapid_tag_frame(df, tag_definitions=self.tags)
        self.assertEqual(resolved.loc[0, "Effective Rapid Best Tag"], "SAT")

    def test_nullable_jev_tags_values_do_not_crash(self) -> None:
        for value in [pd.NA, None, math.nan, ""]:
            with self.subTest(value=repr(value)):
                resolved = resolve_rapid_tag_row(
                    self.row(**{"Jev Tags": value}),
                    tag_definitions=self.tags,
                )
                self.assertEqual(resolved["Rapid Tag Resolution Status"], STATUS_INVALID)
                self.assertEqual(resolved["Effective Rapid Tags"], "")

    def test_valid_single_and_multi_tag_values_still_parse(self) -> None:
        single = resolve_rapid_tag_row(
            self.row(**{"Jev Best Tag": "SAT", "Jev Tags": "SAT"}),
            tag_definitions=self.tags,
        )
        multi = resolve_rapid_tag_row(
            self.row(**{"Jev Best Tag": "AP", "Jev Tags": "AP; SAT"}),
            tag_definitions=self.tags,
        )
        self.assertEqual(single["Rapid Tag Resolution Status"], STATUS_FIRST_PASS_ONLY)
        self.assertEqual(single["Effective Rapid Tags"], "SAT")
        self.assertEqual(multi["Rapid Tag Resolution Status"], STATUS_FIRST_PASS_ONLY)
        self.assertEqual(multi["Effective Rapid Tags"], "AP; SAT")

    def test_nullable_tag_fields_do_not_crash_frame_helpers_used_by_step_4(self) -> None:
        df = pd.DataFrame(
            [
                self.row(**{"Group ID": 1, "Jev Tags": pd.NA}),
                self.row(**{"Group ID": 2, "Rapid Accepted Machine Tags": pd.NA}),
            ]
        )
        effective = build_effective_rapid_tag_frame(df, tag_definitions=self.tags)
        final = build_final_rapid_tag_frame(df, tag_definitions=self.tags)
        self.assertEqual(effective.loc[0, "Rapid Tag Resolution Status"], STATUS_INVALID)
        self.assertEqual(final.loc[0, "Final Rapid Tags"], "")
        self.assertEqual(final.loc[1, "Final Rapid Tags"], "SAT")

    def test_human_3_way_assignment_overrides_only_3_way_scale(self) -> None:
        resolved = resolve_final_rapid_sentiment_row(
            self.row(
                **{
                    "Jev Sentiment": "NEGATIVE",
                    "Jev 5-Way Sentiment": "SOMEWHAT NEGATIVE",
                    "Rapid Human Sentiment Review State": HUMAN_STATE_ASSIGNED,
                    "Rapid Human Sentiment Scale": HUMAN_SENTIMENT_SCALE_3_WAY,
                    "Rapid Human Sentiment": "NEUTRAL",
                }
            )
        )
        self.assertEqual(resolved["Final Rapid Relevance"], RELEVANCE_RELEVANT)
        self.assertEqual(resolved["Final Rapid Sentiment 3-Way"], "NEUTRAL")
        self.assertEqual(resolved["Final Rapid Sentiment 5-Way"], "SOMEWHAT NEGATIVE")
        self.assertEqual(resolved["Final Rapid Sentiment Source"], SOURCE_HUMAN_ASSIGNED)

    def test_human_5_way_assignment_overrides_only_5_way_scale(self) -> None:
        resolved = resolve_final_rapid_sentiment_row(
            self.row(
                **{
                    "Jev Sentiment": "NEUTRAL",
                    "Jev 5-Way Sentiment": "NEUTRAL",
                    "Rapid Human Sentiment Review State": HUMAN_STATE_ASSIGNED,
                    "Rapid Human Sentiment Scale": HUMAN_SENTIMENT_SCALE_5_WAY,
                    "Rapid Human Sentiment": "SOMEWHAT POSITIVE",
                }
            )
        )
        self.assertEqual(resolved["Final Rapid Sentiment 3-Way"], "NEUTRAL")
        self.assertEqual(resolved["Final Rapid Sentiment 5-Way"], "SOMEWHAT POSITIVE")

    def test_human_not_relevant_overrides_relevance_without_numeric_score(self) -> None:
        resolved = resolve_final_rapid_sentiment_row(
            self.row(
                **{
                    "Jev Sentiment": "POSITIVE",
                    "Jev 5-Way Sentiment": "VERY POSITIVE",
                    "Jev Sentiment Score": 6.1,
                    "Rapid Human Sentiment Review State": HUMAN_STATE_ASSIGNED,
                    "Rapid Human Sentiment Scale": HUMAN_SENTIMENT_SCALE_5_WAY,
                    "Rapid Human Sentiment": "NOT RELEVANT",
                }
            )
        )
        self.assertEqual(resolved["Final Rapid Relevance"], RELEVANCE_NOT_RELEVANT)
        self.assertEqual(resolved["Final Rapid Sentiment 3-Way"], "NOT RELEVANT")
        self.assertEqual(resolved["Final Rapid Sentiment 5-Way"], "NOT RELEVANT")
        self.assertTrue(pd.isna(resolved["Final Rapid Sentiment Score"]))

    def test_accepted_machine_sentiment_uses_snapshot(self) -> None:
        resolved = resolve_final_rapid_sentiment_row(
            self.row(
                **{
                    "Jev Sentiment": "NEGATIVE",
                    "Jev 5-Way Sentiment": "SOMEWHAT NEGATIVE",
                    "Rapid Human Sentiment Review State": HUMAN_STATE_ACCEPTED,
                    "Rapid Accepted Machine Relevance": RELEVANCE_RELEVANT,
                    "Rapid Accepted Machine Sentiment 3-Way": "POSITIVE",
                    "Rapid Accepted Machine Sentiment 5-Way": "SOMEWHAT POSITIVE",
                    "Rapid Accepted Machine Sentiment Score": 2.0,
                }
            )
        )
        self.assertEqual(resolved["Final Rapid Sentiment 3-Way"], "POSITIVE")
        self.assertEqual(resolved["Final Rapid Sentiment 5-Way"], "SOMEWHAT POSITIVE")
        self.assertEqual(resolved["Final Rapid Sentiment Source"], SOURCE_HUMAN_ACCEPTED_MACHINE)

    def test_clicking_current_3_way_machine_label_records_accepted_machine(self) -> None:
        df = pd.DataFrame([self.row(**{"Jev Sentiment": "NEGATIVE", "Jev 5-Way Sentiment": "SOMEWHAT NEGATIVE", "Jev Sentiment Score": -2.0})])
        updated = apply_rapid_sentiment_human_selection(
            df,
            1,
            scale=HUMAN_SENTIMENT_SCALE_3_WAY,
            label="NEGATIVE",
        )
        self.assertEqual(updated.loc[0, "Rapid Human Sentiment Review State"], HUMAN_STATE_ACCEPTED)
        self.assertEqual(updated.loc[0, "Rapid Accepted Machine Sentiment 3-Way"], "NEGATIVE")
        self.assertEqual(updated.loc[0, "Rapid Accepted Machine Sentiment 5-Way"], "SOMEWHAT NEGATIVE")

    def test_clicking_different_3_way_label_records_assigned(self) -> None:
        df = pd.DataFrame([self.row(**{"Jev Sentiment": "NEGATIVE", "Jev 5-Way Sentiment": "SOMEWHAT NEGATIVE"})])
        updated = apply_rapid_sentiment_human_selection(
            df,
            1,
            scale=HUMAN_SENTIMENT_SCALE_3_WAY,
            label="NEUTRAL",
        )
        self.assertEqual(updated.loc[0, "Rapid Human Sentiment Review State"], HUMAN_STATE_ASSIGNED)
        self.assertEqual(updated.loc[0, "Rapid Human Sentiment Scale"], HUMAN_SENTIMENT_SCALE_3_WAY)
        self.assertEqual(updated.loc[0, "Rapid Human Sentiment"], "NEUTRAL")
        self.assertTrue(pd.isna(updated.loc[0, "Rapid Accepted Machine Sentiment 3-Way"]))

    def test_clicking_current_5_way_label_compares_against_5_way_machine_label(self) -> None:
        df = pd.DataFrame([self.row(**{"Jev Sentiment": "NEUTRAL", "Jev 5-Way Sentiment": "SOMEWHAT POSITIVE"})])
        updated = apply_rapid_sentiment_human_selection(
            df,
            1,
            scale=HUMAN_SENTIMENT_SCALE_5_WAY,
            label="SOMEWHAT POSITIVE",
        )
        self.assertEqual(updated.loc[0, "Rapid Human Sentiment Review State"], HUMAN_STATE_ACCEPTED)

    def test_clicking_label_without_machine_effective_label_records_assigned(self) -> None:
        df = pd.DataFrame([self.row(**{"Jev Sentiment": "NOT RELEVANT", "Jev 5-Way Sentiment": "NEUTRAL"})])
        updated = apply_rapid_sentiment_human_selection(
            df,
            1,
            scale=HUMAN_SENTIMENT_SCALE_3_WAY,
            label="NEUTRAL",
        )
        self.assertEqual(updated.loc[0, "Rapid Human Sentiment Review State"], HUMAN_STATE_ASSIGNED)
        self.assertEqual(updated.loc[0, "Rapid Human Sentiment"], "NEUTRAL")

    def test_unreviewed_has_no_skipped_state_and_uses_machine(self) -> None:
        row = ensure_rapid_human_review_columns(pd.DataFrame([self.row()])).iloc[0]
        self.assertTrue(pd.isna(row.get("Rapid Human Sentiment Review State")))
        resolved = resolve_final_rapid_sentiment_row(row)
        self.assertEqual(resolved["Final Rapid Sentiment 3-Way"], "NEUTRAL")
        self.assertEqual(resolved["Final Rapid Sentiment Source"], SOURCE_RAPID_FIRST_PASS)

    def test_legacy_human_best_fit_assignment_overrides_only_final_best_fit(self) -> None:
        resolved = resolve_final_rapid_tag_row(
            self.row(
                **{
                    "Rapid Human Tag Review State": HUMAN_STATE_ASSIGNED,
                    "Rapid Human Tag Assignment": "AP",
                    "Rapid Human Tagging Mode": "Single best tag",
                }
            ),
            tag_definitions=self.tags,
        )
        self.assertEqual(resolved["Final Rapid Best Tag"], "AP")
        self.assertEqual(resolved["Final Rapid Tags"], "SAT")
        self.assertEqual(resolved["Final Rapid Tag Source"], f"Best fit: {SOURCE_HUMAN_ASSIGNED}")

    def test_single_best_tag_selection_matching_machine_records_accepted(self) -> None:
        df = pd.DataFrame([self.row(**{"Jev Best Tag": "SAT", "Jev Tags": "SAT"})])
        updated = apply_rapid_tag_human_selection(
            df,
            1,
            assignment="SAT",
            tagging_mode=RAPID_TAG_REVIEW_MODE_BEST,
            tag_definitions=self.tags,
        )
        self.assertEqual(updated.loc[0, "Rapid Human Best Tag Review State"], HUMAN_STATE_ACCEPTED)
        self.assertTrue(pd.isna(updated.loc[0, "Rapid Human Applicable Tags Review State"]))
        self.assertEqual(updated.loc[0, "Rapid Accepted Machine Best Tag"], "SAT")
        self.assertTrue(pd.isna(updated.loc[0, "Rapid Accepted Machine Tags"]))

    def test_single_best_tag_selection_different_from_machine_records_assigned(self) -> None:
        df = pd.DataFrame([self.row(**{"Jev Best Tag": "SAT", "Jev Tags": "SAT"})])
        updated = apply_rapid_tag_human_selection(
            df,
            1,
            assignment="AP",
            tagging_mode=RAPID_TAG_REVIEW_MODE_BEST,
            tag_definitions=self.tags,
        )
        self.assertEqual(updated.loc[0, "Rapid Human Best Tag Review State"], HUMAN_STATE_ASSIGNED)
        self.assertEqual(updated.loc[0, "Rapid Human Best Tag Assignment"], "AP")
        self.assertTrue(pd.isna(updated.loc[0, "Rapid Accepted Machine Best Tag"]))

    def test_single_best_tag_selection_without_machine_effective_tag_records_assigned(self) -> None:
        df = pd.DataFrame([self.row(**{"Jev Best Tag": "", "Jev Tags": ""})])
        updated = apply_rapid_tag_human_selection(
            df,
            1,
            assignment="AP",
            tagging_mode=RAPID_TAG_REVIEW_MODE_BEST,
            tag_definitions=self.tags,
        )
        self.assertEqual(updated.loc[0, "Rapid Human Best Tag Review State"], HUMAN_STATE_ASSIGNED)
        self.assertEqual(updated.loc[0, "Rapid Human Best Tag Assignment"], "AP")

    def test_multiple_applicable_tag_selection_matching_machine_set_records_accepted(self) -> None:
        df = pd.DataFrame([self.row(**{"Jev Best Tag": "AP", "Jev Tags": "AP; SAT"})])
        updated = apply_rapid_tag_human_selection(
            df,
            1,
            assignment="SAT, AP",
            tagging_mode=RAPID_TAG_REVIEW_MODE_APPLICABLE,
            tag_definitions=self.tags,
        )
        self.assertEqual(updated.loc[0, "Rapid Human Applicable Tags Review State"], HUMAN_STATE_ACCEPTED)
        self.assertTrue(pd.isna(updated.loc[0, "Rapid Human Best Tag Review State"]))
        self.assertTrue(pd.isna(updated.loc[0, "Rapid Accepted Machine Best Tag"]))
        self.assertEqual(updated.loc[0, "Rapid Accepted Machine Tags"], "AP; SAT")

    def test_multiple_applicable_tag_selection_different_from_machine_set_records_assigned(self) -> None:
        df = pd.DataFrame([self.row(**{"Jev Best Tag": "AP", "Jev Tags": "AP; SAT"})])
        updated = apply_rapid_tag_human_selection(
            df,
            1,
            assignment="AP",
            tagging_mode=RAPID_TAG_REVIEW_MODE_APPLICABLE,
            tag_definitions=self.tags,
        )
        self.assertEqual(updated.loc[0, "Rapid Human Applicable Tags Review State"], HUMAN_STATE_ASSIGNED)
        self.assertEqual(updated.loc[0, "Rapid Human Applicable Tags Assignment"], "AP")

    def test_multiple_applicable_other_selection_records_normalized_assignment(self) -> None:
        df = pd.DataFrame([self.row(**{"Jev Best Tag": "", "Jev Tags": ""})])
        updated = apply_rapid_tag_human_selection(
            df,
            1,
            assignment="Other, AP",
            tagging_mode=RAPID_TAG_REVIEW_MODE_APPLICABLE,
            tag_definitions=self.tags,
        )
        self.assertEqual(updated.loc[0, "Rapid Human Applicable Tags Review State"], HUMAN_STATE_ASSIGNED)
        self.assertEqual(updated.loc[0, "Rapid Human Applicable Tags Assignment"], "AP")

    def test_multiple_applicable_without_machine_effective_tags_records_assigned(self) -> None:
        df = pd.DataFrame([self.row(**{"Jev Best Tag": "", "Jev Tags": ""})])
        updated = apply_rapid_tag_human_selection(
            df,
            1,
            assignment="AP, SAT",
            tagging_mode=RAPID_TAG_REVIEW_MODE_APPLICABLE,
            tag_definitions=self.tags,
        )
        self.assertEqual(updated.loc[0, "Rapid Human Applicable Tags Review State"], HUMAN_STATE_ASSIGNED)
        self.assertEqual(updated.loc[0, "Rapid Human Applicable Tags Assignment"], "AP; SAT")

    def test_best_fit_and_applicable_human_reviews_are_independent(self) -> None:
        df = pd.DataFrame([self.row(**{"Jev Best Tag": "SAT", "Jev Tags": "SAT"})])
        updated = apply_rapid_tag_human_selection(
            df,
            1,
            assignment="AP",
            tagging_mode=RAPID_TAG_REVIEW_MODE_BEST,
            tag_definitions=self.tags,
        )
        updated = apply_rapid_tag_human_selection(
            updated,
            1,
            assignment="AP, Career",
            tagging_mode=RAPID_TAG_REVIEW_MODE_APPLICABLE,
            tag_definitions=self.tags,
        )
        resolved = resolve_final_rapid_tag_row(updated.iloc[0], tag_definitions=self.tags)
        self.assertEqual(resolved["Final Rapid Best Tag"], "AP")
        self.assertEqual(resolved["Final Rapid Tags"], "AP; Career")
        self.assertEqual(
            resolved["Final Rapid Tag Source"],
            f"Best fit: {SOURCE_HUMAN_ASSIGNED}; Applicable tags: {SOURCE_HUMAN_ASSIGNED}",
        )

    def test_applicable_human_review_survives_later_best_fit_review(self) -> None:
        df = pd.DataFrame([self.row(**{"Jev Best Tag": "SAT", "Jev Tags": "SAT"})])
        updated = apply_rapid_tag_human_selection(
            df,
            1,
            assignment="AP, Career",
            tagging_mode=RAPID_TAG_REVIEW_MODE_APPLICABLE,
            tag_definitions=self.tags,
        )
        updated = apply_rapid_tag_human_selection(
            updated,
            1,
            assignment="AP",
            tagging_mode=RAPID_TAG_REVIEW_MODE_BEST,
            tag_definitions=self.tags,
        )
        resolved = resolve_final_rapid_tag_row(updated.iloc[0], tag_definitions=self.tags)
        self.assertEqual(resolved["Final Rapid Best Tag"], "AP")
        self.assertEqual(resolved["Final Rapid Tags"], "AP; Career")
        self.assertEqual(
            resolved["Final Rapid Tag Source"],
            f"Best fit: {SOURCE_HUMAN_ASSIGNED}; Applicable tags: {SOURCE_HUMAN_ASSIGNED}",
        )

    def test_applicable_human_review_does_not_override_best_fit(self) -> None:
        df = pd.DataFrame([self.row(**{"Jev Best Tag": "SAT", "Jev Tags": "SAT"})])
        updated = apply_rapid_tag_human_selection(
            df,
            1,
            assignment="AP, Career",
            tagging_mode=RAPID_TAG_REVIEW_MODE_APPLICABLE,
            tag_definitions=self.tags,
        )
        resolved = resolve_final_rapid_tag_row(updated.iloc[0], tag_definitions=self.tags)
        self.assertEqual(resolved["Final Rapid Best Tag"], "SAT")
        self.assertEqual(resolved["Final Rapid Tags"], "AP; Career")
        self.assertEqual(resolved["Final Rapid Tag Source"], f"Applicable tags: {SOURCE_HUMAN_ASSIGNED}")

    def test_final_frame_helpers_preserve_row_indexes(self) -> None:
        df = pd.DataFrame([self.row(**{"Group ID": 10}), self.row(**{"Group ID": 11, "Jev Best Tag": "AP", "Jev Tags": "AP"})])
        sentiment = build_final_rapid_sentiment_frame(df)
        tags = build_final_rapid_tag_frame(df, tag_definitions=self.tags)
        self.assertEqual(list(sentiment.index), [0, 1])
        self.assertEqual(list(tags.index), [0, 1])

    def test_highlight_keywords_use_analysis_context_categories(self) -> None:
        keywords = build_rapid_highlight_keywords(
            {
                "primary_name": "College Board",
                "alternate_names": ["CB"],
                "spokespeople": ["Jane Doe"],
                "products": ["SAT", "AP"],
                "highlight_keywords": ["National Recognition Program", "SAT"],
            }
        )
        self.assertEqual(
            keywords,
            ["College Board", "CB", "Jane Doe", "SAT", "AP", "National Recognition Program"],
        )
        self.assertEqual(build_rapid_highlight_keywords({}), [])

    def test_format_rapid_numeric_rounds_and_omits_missing_values(self) -> None:
        self.assertEqual(format_rapid_numeric(1.23456), "1.23")
        self.assertEqual(format_rapid_numeric("0.9876"), "0.99")
        self.assertEqual(format_rapid_numeric(pd.NA), "")
        self.assertEqual(format_rapid_numeric(None), "")


if __name__ == "__main__":
    unittest.main()
