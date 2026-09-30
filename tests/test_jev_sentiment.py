from __future__ import annotations

import json
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from processing.jev_sentiment import (
    DEFAULT_JEV_MODEL,
    DEFAULT_JEV_TAGGING_MODE,
    JEV_3WAY_SENTIMENT_ORDER,
    JEV_5WAY_SENTIMENT_ORDER,
    JEV_OTHER_TAG_COLOR,
    JEV_SENTIMENT_5WAY_SCHEME,
    JEV_SENTIMENT_SCORE_SCHEME,
    apply_jev_sentiment_result_to_unique_df,
    apply_jev_combined_result_to_unique_df,
    build_jev_all_applicable_tag_distribution,
    build_jev_combined_sentiment_payload,
    build_jev_best_tag_distribution,
    build_jev_results_display,
    build_jev_sentiment_distribution,
    build_jev_sentiment_payload,
    build_jev_tag_color_scale,
    build_jev_tag_config_fingerprint,
    build_jev_tag_question_map,
    call_jev_sentiment,
    ensure_jev_sentiment_columns,
    filter_jev_all_applicable_tag_distribution,
    filter_jev_best_tag_distribution,
    filter_jev_sentiment_distribution,
    get_remaining_jev_sentiment_rows,
    parse_jev_tag_definitions,
    parse_jev_combined_sentiment_response,
    parse_jev_sentiment_response,
    recompute_jev_official_tag_assignments,
    run_jev_sentiment_batch,
)
from processing.download_exports import (
    build_export_metadata_sheet,
    build_jev_sentiment_sample_export,
    build_rapid_clean_export_map,
    build_sentiment_sample_export,
    build_tagging_sample_export,
    labeling_audit_available,
    merge_full_scope_ai_columns_into_clean_trad,
    remove_inactive_workflow_columns,
)


class JevSentimentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.analysis_payload = {
            "primary_name": "College Board",
            "alternate_names": ["The College Board"],
            "spokespeople": ["Example Spokesperson"],
            "products": ["AP", "SAT", "PSAT", "BigFuture"],
            "general_guidance": "Treat test-prep ecosystem criticism separately from entity criticism.",
            "sentiment_guidance": "Positive student outcomes are positive only when credited to College Board.",
        }
        self.row = pd.Series(
            {
                "Group ID": 10,
                "Headline": "Students recognized by College Board",
                "Snippet": "The College Board program recognized high-achieving students.",
                "Outlet": "Example Daily",
                "Type": "ONLINE",
            }
        )

    def combined_response(self, *, three_way: str = "POSITIVE", five_way: str = "SOMEWHAT POSITIVE") -> dict:
        return {
            "model": "typesafe/jev-1.13-20260917",
            "answers": {
                "sentiment": {
                    "type": "choice",
                    "choice": three_way,
                    "confidence": 0.84,
                    "probabilities": {
                        "POSITIVE": 0.88 if three_way == "POSITIVE" else 0.05,
                        "NEUTRAL": 0.12 if three_way == "POSITIVE" else 0.9,
                        "NEGATIVE": 0.0,
                        "NOT RELEVANT": 0.0,
                    },
                },
                "sentiment_5_way": {
                    "type": "choice",
                    "choice": five_way,
                    "confidence": 0.72,
                    "probabilities": {
                        "VERY POSITIVE": 0.1,
                        "SOMEWHAT POSITIVE": 0.7,
                        "NEUTRAL": 0.2,
                        "SOMEWHAT NEGATIVE": 0.0,
                        "VERY NEGATIVE": 0.0,
                        "NOT RELEVANT": 0.0,
                    },
                },
                "sentiment_score": {
                    "type": "score",
                    "score": 5,
                    "legend": {str(i): f"level {i}" for i in range(9)},
                    "probabilities": {
                        "0": 0.0,
                        "1": 0.0,
                        "2": 0.0,
                        "3": 0.0,
                        "4": 0.1,
                        "5": 0.2,
                        "6": 0.7,
                        "7": 0.0,
                        "8": 0.0,
                    },
                    "confidence": 0.72,
                },
                "sentiment_relevant": {"type": "noul", "noul": 0.98},
                "sentiment_mixture": {
                    "type": "score",
                    "score": 1.2,
                    "legend": {
                        "0": "VERY CONSISTENT: uniform",
                        "1": "MOSTLY CONSISTENT: dominant direction",
                        "2": "SOMEWHAT MIXED: meaningful different material",
                        "3": "SUBSTANTIALLY MIXED: multiple prominent directions",
                        "4": "HIGHLY MIXED: strongly contrasting treatment",
                    },
                    "probabilities": {
                        "0": 0.1,
                        "1": 0.7,
                        "2": 0.2,
                        "3": 0.0,
                        "4": 0.0,
                    },
                    "confidence": 0.76,
                },
            },
            "usage": {"input_tokens": 1200, "output_tokens": 20, "cost": 0.0000504},
        }

    def tag_response(self) -> dict:
        response = self.combined_response()
        response["answers"].update(
            {
                "tag_best_fit": {
                    "type": "choice",
                    "choice": "Access",
                    "confidence": 0.81,
                    "probabilities": {
                        "Access": 0.72,
                        "Affordability": 0.18,
                        "Other": 0.10,
                    },
                },
                "tag_001": {
                    "type": "choice",
                    "choice": "YES",
                    "confidence": 0.77,
                    "probabilities": {"YES": 0.74, "NO": 0.26},
                },
                "tag_002": {
                    "type": "choice",
                    "choice": "NO",
                    "confidence": 0.66,
                    "probabilities": {"YES": 0.41, "NO": 0.59},
                },
            }
        )
        return response

    def test_request_construction_uses_fixed_v1_choice_scheme_and_context(self) -> None:
        payload = build_jev_sentiment_payload(self.row, self.analysis_payload)

        self.assertEqual(payload["model"], DEFAULT_JEV_MODEL)
        self.assertEqual(payload["state"]["entity_context"]["primary_entity"], "College Board")
        self.assertIn("AP", payload["state"]["entity_context"]["products_subbrands_programs"])
        self.assertEqual(payload["state"]["story"]["headline"], "Students recognized by College Board")
        self.assertIn("directly mentioned", payload["state"]["sentiment_rules"]["direct_mention_scope"])
        self.assertIn(
            "high-priority",
            payload["state"]["analysis_guidance"]["sentiment_specific_guidance_priority"].lower(),
        )

        sentiment_question = payload["questions"]["sentiment"]
        self.assertEqual(sentiment_question["type"], "choice")
        self.assertEqual(
            list(sentiment_question["criteria"].keys()),
            ["POSITIVE", "NEUTRAL", "NEGATIVE", "NOT RELEVANT"],
        )
        self.assertIn("broader topic", sentiment_question["criteria"]["NEGATIVE"])

    def test_response_parsing_extracts_choice_probabilities_confidence_and_cost(self) -> None:
        response = {
            "model": "typesafe/jev-1.13-20260917",
            "answers": {
                "sentiment": {
                    "type": "choice",
                    "choice": "POSITIVE",
                    "confidence": 0.84,
                    "probabilities": {
                        "POSITIVE": 0.88,
                        "NEUTRAL": 0.12,
                        "NEGATIVE": 0.0,
                        "NOT RELEVANT": 0.0,
                    },
                }
            },
            "usage": {"input_tokens": 1000, "output_tokens": 10, "cost": 0.000014994},
            "id": "gen-dec-test",
            "provider": "TypeSafe",
        }

        parsed = parse_jev_sentiment_response(response)

        self.assertEqual(parsed["sentiment"], "POSITIVE")
        self.assertEqual(parsed["selected_probability"], 0.88)
        self.assertEqual(parsed["confidence"], 0.84)
        self.assertEqual(parsed["probabilities"]["NEUTRAL"], 0.12)
        self.assertEqual(parsed["input_tokens"], 1000)
        self.assertAlmostEqual(parsed["cost_usd"], 0.000014994)
        self.assertEqual(json.loads(parsed["raw_response"])["model"], "typesafe/jev-1.13-20260917")

    def test_five_way_request_construction_uses_intensity_labels(self) -> None:
        payload = build_jev_sentiment_payload(
            self.row,
            self.analysis_payload,
            scheme=JEV_SENTIMENT_5WAY_SCHEME,
        )

        sentiment_question = payload["questions"]["sentiment_5_way"]
        self.assertEqual(sentiment_question["type"], "choice")
        self.assertEqual(
            list(sentiment_question["criteria"].keys()),
            [
                "VERY POSITIVE",
                "SOMEWHAT POSITIVE",
                "NEUTRAL",
                "SOMEWHAT NEGATIVE",
                "VERY NEGATIVE",
                "NOT RELEVANT",
            ],
        )
        self.assertIn("Strong praise", sentiment_question["criteria"]["VERY POSITIVE"])
        self.assertIn("broader topic is negative", sentiment_question["criteria"]["SOMEWHAT NEGATIVE"])
        self.assertIn("directly mentioned", sentiment_question["criteria"]["NOT RELEVANT"])

    def test_jev_state_clarifies_configured_products_are_relevant_without_parent_name(self) -> None:
        payload = build_jev_combined_sentiment_payload(self.row, self.analysis_payload)

        rules = payload["state"]["sentiment_rules"]

        self.assertIn("parent organization does not need to be explicitly named", rules["configured_member_scope"])
        self.assertIn("configured alias, spokesperson", rules["configured_member_scope"])
        self.assertIn("product, sub-brand, or program is in scope", rules["configured_member_scope"])

    def test_five_way_response_parsing_extracts_all_probabilities(self) -> None:
        response = {
            "model": "typesafe/jev-1.13-20260917",
            "answers": {
                "sentiment_5_way": {
                    "type": "choice",
                    "choice": "SOMEWHAT NEGATIVE",
                    "confidence": 0.72,
                    "probabilities": {
                        "VERY POSITIVE": 0.0,
                        "SOMEWHAT POSITIVE": 0.03,
                        "NEUTRAL": 0.25,
                        "SOMEWHAT NEGATIVE": 0.70,
                        "VERY NEGATIVE": 0.02,
                        "NOT RELEVANT": 0.0,
                    },
                }
            },
            "usage": {"input_tokens": 1100, "output_tokens": 12, "cost": 0.0000462},
        }

        parsed = parse_jev_sentiment_response(response, scheme=JEV_SENTIMENT_5WAY_SCHEME)

        self.assertEqual(parsed["sentiment"], "SOMEWHAT NEGATIVE")
        self.assertEqual(parsed["selected_probability"], 0.70)
        self.assertEqual(parsed["confidence"], 0.72)
        self.assertEqual(parsed["probabilities"]["VERY NEGATIVE"], 0.02)
        self.assertEqual(parsed["cost_usd"], 0.0000462)

    def test_score_request_construction_uses_score_and_relevance_questions(self) -> None:
        payload = build_jev_sentiment_payload(
            self.row,
            self.analysis_payload,
            scheme=JEV_SENTIMENT_SCORE_SCHEME,
        )

        self.assertEqual(payload["model"], DEFAULT_JEV_MODEL)
        self.assertEqual(set(payload["questions"].keys()), {"sentiment_score", "sentiment_relevant"})
        score_question = payload["questions"]["sentiment_score"]
        relevance_question = payload["questions"]["sentiment_relevant"]
        self.assertEqual(score_question["type"], "score")
        self.assertEqual(relevance_question["type"], "noul")
        self.assertEqual(len(score_question["criteria"]), 9)
        self.assertIn("Extremely negative", score_question["criteria"][0])
        self.assertIn("Neutral", score_question["criteria"][4])
        self.assertIn("Extremely positive", score_question["criteria"][-1])
        self.assertIn("collective entity", relevance_question["instructions"])

    def test_combined_request_contains_all_expected_jev_questions(self) -> None:
        payload = build_jev_combined_sentiment_payload(self.row, self.analysis_payload)

        self.assertEqual(payload["model"], DEFAULT_JEV_MODEL)
        self.assertEqual(
            set(payload["questions"].keys()),
            {"sentiment", "sentiment_5_way", "sentiment_score", "sentiment_relevant", "sentiment_mixture"},
        )
        self.assertEqual(payload["questions"]["sentiment"]["type"], "choice")
        self.assertEqual(payload["questions"]["sentiment_5_way"]["type"], "choice")
        self.assertEqual(payload["questions"]["sentiment_score"]["type"], "score")
        self.assertEqual(payload["questions"]["sentiment_relevant"]["type"], "noul")
        self.assertEqual(payload["questions"]["sentiment_mixture"]["type"], "score")
        self.assertEqual(payload["state"]["story"]["headline"], "Students recognized by College Board")
        mixture_criteria = payload["questions"]["sentiment_mixture"]["criteria"]
        self.assertEqual(len(mixture_criteria), 5)
        self.assertIn("VERY CONSISTENT", mixture_criteria[0])
        self.assertIn("Ordinary factual/background material does not make a story mixed", mixture_criteria[0])
        self.assertIn("HIGHLY MIXED", mixture_criteria[-1])

    def test_jev_tag_parser_preserves_protected_other_and_rejects_bad_lines(self) -> None:
        parsed = parse_jev_tag_definitions("Access: Availability of programs\nAffordability: Cost barriers")

        self.assertEqual(parsed["Access"], "Availability of programs")
        self.assertEqual(parsed["Other"], "None of the other tags apply.")

        with self.assertRaisesRegex(ValueError, "Line 1"):
            parse_jev_tag_definitions("Broken line")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            parse_jev_tag_definitions("Access: One\naccess: Two")
        with self.assertRaisesRegex(ValueError, "protected"):
            parse_jev_tag_definitions("Other: fallback")
        self.assertEqual(parse_jev_tag_definitions("   \n"), {})

    def test_jev_tag_fingerprint_is_order_independent(self) -> None:
        first = parse_jev_tag_definitions("Access: Availability\nAffordability: Costs")
        second = parse_jev_tag_definitions("Affordability: Costs\nAccess: Availability")

        self.assertEqual(build_jev_tag_config_fingerprint(first), build_jev_tag_config_fingerprint(second))
        self.assertEqual([item["question_id"] for item in build_jev_tag_question_map(first)], ["tag_001", "tag_002"])

    def test_jev_3way_distribution_uses_group_count_and_not_relevant_toggle(self) -> None:
        df = pd.DataFrame(
            {
                "Jev Sentiment": ["POSITIVE", "POSITIVE", "NEGATIVE", "NOT RELEVANT", ""],
                "Group Count": [2, 3, 1, 4, 99],
            }
        )

        dist = build_jev_sentiment_distribution(df, column="Jev Sentiment", order=JEV_3WAY_SENTIMENT_ORDER)
        filtered = filter_jev_sentiment_distribution(
            dist,
            order=JEV_3WAY_SENTIMENT_ORDER,
            include_not_relevant=False,
        )
        by_sentiment = filtered.set_index("Sentiment")

        self.assertEqual(int(by_sentiment.loc["POSITIVE", "Count"]), 5)
        self.assertEqual(int(by_sentiment.loc["POSITIVE", "Grouped Stories"]), 2)
        self.assertNotIn("NOT RELEVANT", filtered["Sentiment"].astype(str).tolist())
        self.assertAlmostEqual(float(by_sentiment.loc["POSITIVE", "Share"]), 5 / 6)

        included = filter_jev_sentiment_distribution(
            dist,
            order=JEV_3WAY_SENTIMENT_ORDER,
            include_not_relevant=True,
        )
        self.assertAlmostEqual(float(included.set_index("Sentiment").loc["POSITIVE", "Share"]), 5 / 10)

    def test_jev_5way_distribution_preserves_order_for_partial_batches(self) -> None:
        df = pd.DataFrame(
            {
                "Jev 5-Way Sentiment": ["SOMEWHAT NEGATIVE", "VERY POSITIVE", ""],
                "Group Count": [4, 2, 7],
            }
        )

        dist = filter_jev_sentiment_distribution(
            build_jev_sentiment_distribution(
                df,
                column="Jev 5-Way Sentiment",
                order=JEV_5WAY_SENTIMENT_ORDER,
            ),
            order=JEV_5WAY_SENTIMENT_ORDER,
            include_not_relevant=False,
        )

        self.assertEqual(dist["Sentiment"].astype(str).tolist(), JEV_5WAY_SENTIMENT_ORDER[:-1])
        self.assertEqual(int(dist.set_index("Sentiment").loc["VERY POSITIVE", "Count"]), 2)
        self.assertEqual(int(dist.set_index("Sentiment").loc["SOMEWHAT NEGATIVE", "Grouped Stories"]), 1)

    def test_jev_best_fit_tag_distribution_is_mutually_exclusive_with_other_toggle(self) -> None:
        df = pd.DataFrame(
            {
                "Jev Best Tag": ["Access", "Affordability", "Other", "Careers", ""],
                "Group Count": [2, 3, 50, 1, 11],
            }
        )

        dist = build_jev_best_tag_distribution(df)
        excluded = filter_jev_best_tag_distribution(dist, include_other=False)
        included = filter_jev_best_tag_distribution(dist, include_other=True)

        self.assertEqual(excluded["Tag"].tolist(), ["Affordability", "Access", "Careers"])
        self.assertNotIn("Other", excluded["Tag"].tolist())
        self.assertAlmostEqual(float(excluded["Share"].sum()), 1.0)
        self.assertAlmostEqual(float(excluded.set_index("Tag").loc["Affordability", "Share"]), 3 / 6)
        self.assertEqual(included["Tag"].tolist(), ["Affordability", "Access", "Careers", "Other"])
        self.assertAlmostEqual(float(included.set_index("Tag").loc["Other", "Share"]), 50 / 56)

    def test_jev_tag_color_scale_is_stable_when_other_is_toggled(self) -> None:
        tags = ["Access", "Affordability", "Careers", "Other"]

        domain_without_other, colors_without_other = build_jev_tag_color_scale(tags, include_other=False)
        domain_with_other, colors_with_other = build_jev_tag_color_scale(tags, include_other=True)

        self.assertEqual(domain_without_other, ["Access", "Affordability", "Careers"])
        self.assertEqual(domain_with_other, ["Access", "Affordability", "Careers", "Other"])
        self.assertEqual(colors_without_other, colors_with_other[:3])
        self.assertEqual(colors_with_other[-1], JEV_OTHER_TAG_COLOR)

    def test_best_fit_donut_is_configured_with_visible_legend(self) -> None:
        page_source = Path(__file__).resolve().parents[1].joinpath("pages/Jev_Sentiment_Experimental.py").read_text()

        self.assertIn("alt.Legend(", page_source)
        self.assertIn("orient=\"right\"", page_source)
        self.assertIn("show_legend=True", page_source)

    def test_jev_all_applicable_tag_distribution_counts_overlap_and_other_fallback(self) -> None:
        tag_definitions = parse_jev_tag_definitions("Access: Availability\nAffordability: Costs")
        df = ensure_jev_sentiment_columns(
            pd.DataFrame(
                {
                    "Group Count": [2, 3, 5, 7],
                    "Jev Tag [Access]": ["YES", "YES", "NO", ""],
                    "Jev Tag [Affordability]": ["YES", "NO", "NO", "YES"],
                }
            ),
            tag_definitions=tag_definitions,
        )

        dist = build_jev_all_applicable_tag_distribution(df, tag_definitions=tag_definitions)
        by_tag = dist.set_index("Tag")

        self.assertEqual(int(by_tag.loc["Access", "Count"]), 5)
        self.assertEqual(int(by_tag.loc["Affordability", "Count"]), 2)
        self.assertEqual(int(by_tag.loc["Other", "Count"]), 5)
        self.assertEqual(int(by_tag.loc["Access", "Processed Grouped Stories"]), 3)
        self.assertAlmostEqual(float(by_tag.loc["Access", "Grouped Story Share"]), 2 / 3)
        self.assertAlmostEqual(float(by_tag.loc["Other", "Grouped Story Share"]), 1 / 3)

        no_other = filter_jev_all_applicable_tag_distribution(dist, include_other=False)
        self.assertNotIn("Other", no_other["Tag"].tolist())

    def test_jev_distribution_helpers_handle_no_tag_configuration(self) -> None:
        dist = build_jev_all_applicable_tag_distribution(
            pd.DataFrame({"Jev Best Tag": ["Access"], "Group Count": [1]}),
            tag_definitions={},
        )

        self.assertTrue(dist.empty)

    def test_jev_tag_distributions_do_not_depend_on_assignment_mode(self) -> None:
        tag_definitions = parse_jev_tag_definitions("Access: Availability\nAffordability: Costs")
        df = ensure_jev_sentiment_columns(
            pd.DataFrame(
                {
                    "Jev Tags": ["Affordability"],
                    "Jev Best Tag": ["Access"],
                    "Group Count": [2],
                    "Jev Tag [Access]": ["YES"],
                    "Jev Tag [Affordability]": ["YES"],
                }
            ),
            tag_definitions=tag_definitions,
        )

        best_dist = build_jev_best_tag_distribution(df)
        applicable_dist = build_jev_all_applicable_tag_distribution(df, tag_definitions=tag_definitions)

        self.assertEqual(best_dist.loc[0, "Tag"], "Access")
        self.assertEqual(set(applicable_dist["Tag"].tolist()), {"Access", "Affordability"})

    def test_combined_request_includes_best_fit_and_binary_tag_questions(self) -> None:
        tag_definitions = parse_jev_tag_definitions("Access: Availability\nAffordability: Costs")
        payload = build_jev_combined_sentiment_payload(
            self.row,
            self.analysis_payload,
            tag_definitions=tag_definitions,
        )

        self.assertEqual(len(payload["questions"]), 8)
        self.assertEqual(payload["questions"]["tag_best_fit"]["type"], "choice")
        self.assertEqual(
            list(payload["questions"]["tag_best_fit"]["criteria"].keys()),
            ["Access", "Affordability", "Other"],
        )
        self.assertEqual(payload["questions"]["tag_001"]["type"], "choice")
        self.assertEqual(list(payload["questions"]["tag_001"]["criteria"].keys()), ["YES", "NO"])
        self.assertEqual(payload["questions"]["tag_002"]["type"], "choice")
        self.assertNotIn("tag_003", payload["questions"])

    def test_score_response_parsing_extracts_score_relevance_and_distribution(self) -> None:
        response = {
            "model": "typesafe/jev-1.13-20260917",
            "answers": {
                "sentiment_score": {
                    "type": "score",
                    "score": 5.25,
                    "legend": {
                        "0": "Extremely negative",
                        "1": "Strongly negative",
                        "2": "Moderately negative",
                        "3": "Slightly negative",
                        "4": "Neutral",
                        "5": "Slightly positive",
                        "6": "Moderately positive",
                        "7": "Strongly positive",
                        "8": "Extremely positive",
                    },
                    "probabilities": {
                        "0": 0.0,
                        "1": 0.0,
                        "2": 0.0,
                        "3": 0.0,
                        "4": 0.1,
                        "5": 0.2,
                        "6": 0.7,
                        "7": 0.0,
                        "8": 0.0,
                    },
                    "confidence": 0.72,
                },
                "sentiment_relevant": {"type": "noul", "noul": 0.98},
            },
            "usage": {"input_tokens": 1200, "output_tokens": 20, "cost": 0.0000504},
        }

        parsed = parse_jev_sentiment_response(response, scheme=JEV_SENTIMENT_SCORE_SCHEME)

        self.assertAlmostEqual(parsed["score"], 2.3)
        self.assertEqual(parsed["native_score"], 5.25)
        self.assertEqual(parsed["confidence"], 0.72)
        self.assertEqual(parsed["relevant_probability"], 0.98)
        self.assertEqual(json.loads(parsed["score_probabilities"])["6"], 0.7)
        self.assertEqual(parsed["cost_usd"], 0.0000504)

    def test_combined_response_populates_all_three_sentiment_approaches(self) -> None:
        parsed = parse_jev_combined_sentiment_response(self.combined_response())

        self.assertEqual(parsed["model"], "typesafe/jev-1.13-20260917")
        self.assertEqual(parsed["input_tokens"], 1200)
        self.assertEqual(parsed["cost_usd"], 0.0000504)
        self.assertEqual(parsed["methods"]["3-way"]["sentiment"], "POSITIVE")
        self.assertEqual(parsed["methods"]["5-way"]["sentiment"], "SOMEWHAT POSITIVE")
        self.assertAlmostEqual(parsed["methods"]["score-7"]["score"], 2.3)
        self.assertEqual(parsed["methods"]["score-7"]["relevant_probability"], 0.98)
        self.assertEqual(parsed["methods"]["mixture"]["label"], "MOSTLY CONSISTENT")
        self.assertEqual(parsed["methods"]["mixture"]["score"], 1.2)
        self.assertEqual(parsed["methods"]["mixture"]["confidence"], 0.76)

    def test_tag_response_parsing_and_assignment_modes(self) -> None:
        tag_definitions = parse_jev_tag_definitions("Access: Availability\nAffordability: Costs")

        parsed_single = parse_jev_combined_sentiment_response(
            self.tag_response(),
            tag_definitions=tag_definitions,
            tagging_mode="Single best tag",
        )
        self.assertEqual(parsed_single["tagging"]["best_tag"], "Access")
        self.assertEqual(parsed_single["tagging"]["tags"], ["Access"])
        self.assertAlmostEqual(parsed_single["tagging"]["independent"]["Access"]["yes_probability"], 0.74)
        self.assertAlmostEqual(parsed_single["tagging"]["independent"]["Affordability"]["yes_probability"], 0.41)

        response = self.tag_response()
        response["answers"]["tag_best_fit"]["choice"] = "Other"
        parsed_multi = parse_jev_combined_sentiment_response(
            response,
            tag_definitions=tag_definitions,
            tagging_mode="Multiple applicable tags",
        )
        self.assertEqual(parsed_multi["tagging"]["best_tag"], "Other")
        self.assertEqual(parsed_multi["tagging"]["tags"], ["Access"])

    def test_zero_independent_yes_assigns_other_in_multi_mode(self) -> None:
        tag_definitions = parse_jev_tag_definitions("Access: Availability\nAffordability: Costs")
        response = self.tag_response()
        response["answers"]["tag_001"]["choice"] = "NO"
        response["answers"]["tag_001"]["probabilities"] = {"YES": 0.12, "NO": 0.88}
        response["answers"]["tag_002"]["choice"] = "NO"
        response["answers"]["tag_002"]["probabilities"] = {"YES": 0.41, "NO": 0.59}

        parsed = parse_jev_combined_sentiment_response(
            response,
            tag_definitions=tag_definitions,
            tagging_mode="Multiple applicable tags",
        )

        self.assertEqual(parsed["tagging"]["tags"], ["Other"])
        self.assertEqual(parsed["tagging"]["tag_count"], 1)

    def test_mixture_label_uses_highest_probability_level_not_rounded_score(self) -> None:
        response = self.combined_response()
        response["answers"]["sentiment_mixture"]["score"] = 1.8
        response["answers"]["sentiment_mixture"]["probabilities"] = {
            "0": 0.05,
            "1": 0.60,
            "2": 0.35,
            "3": 0.0,
            "4": 0.0,
        }

        parsed = parse_jev_combined_sentiment_response(response)

        self.assertEqual(parsed["methods"]["mixture"]["score"], 1.8)
        self.assertEqual(parsed["methods"]["mixture"]["label"], "MOSTLY CONSISTENT")

    def test_jev_result_columns_do_not_overwrite_existing_sentiment_fields(self) -> None:
        df = pd.DataFrame(
            {
                "Group ID": [1],
                "AI Sentiment": ["NEGATIVE"],
                "Assigned Sentiment": ["POSITIVE"],
                "Final Sentiment": ["POSITIVE"],
            }
        )
        result = {
            "sentiment": "NEUTRAL",
            "selected_probability": 0.9,
            "confidence": 0.8,
            "probabilities": {
                "POSITIVE": 0.05,
                "NEUTRAL": 0.9,
                "NEGATIVE": 0.05,
                "NOT RELEVANT": 0.0,
            },
            "model": "typesafe/jev-1.13-20260917",
            "input_tokens": 123,
            "cost_usd": 0.000005,
            "raw_response": "{}",
        }

        updated = apply_jev_sentiment_result_to_unique_df(df, 0, result)

        self.assertEqual(updated.loc[0, "Jev Sentiment"], "NEUTRAL")
        self.assertEqual(updated.loc[0, "AI Sentiment"], "NEGATIVE")
        self.assertEqual(updated.loc[0, "Assigned Sentiment"], "POSITIVE")
        self.assertEqual(updated.loc[0, "Final Sentiment"], "POSITIVE")

    def test_method_specific_results_do_not_overwrite_each_other(self) -> None:
        df = ensure_jev_sentiment_columns(pd.DataFrame({"Group ID": [1]}))
        combined = parse_jev_combined_sentiment_response(self.combined_response())
        updated = apply_jev_combined_result_to_unique_df(df, 0, combined)

        self.assertEqual(updated.loc[0, "Jev Sentiment"], "POSITIVE")
        self.assertEqual(updated.loc[0, "Jev 5-Way Sentiment"], "SOMEWHAT POSITIVE")
        self.assertEqual(updated.loc[0, "Jev Probability POSITIVE"], 0.88)
        self.assertEqual(updated.loc[0, "Jev 5-Way Probability SOMEWHAT POSITIVE"], 0.7)
        self.assertAlmostEqual(updated.loc[0, "Jev Sentiment Score"], 2.3)
        self.assertEqual(updated.loc[0, "Jev Mixture Label"], "MOSTLY CONSISTENT")
        self.assertEqual(updated.loc[0, "Jev Mixture Score"], 1.2)
        self.assertEqual(updated.loc[0, "Jev Mixture Confidence"], 0.76)
        self.assertEqual(updated.loc[0, "Jev Model"], "typesafe/jev-1.13-20260917")
        self.assertEqual(updated.loc[0, "Jev Input Tokens"], 1200)

    def test_tag_results_write_explicit_columns_and_recompute_without_api(self) -> None:
        tag_definitions = parse_jev_tag_definitions("Access: Availability\nAffordability: Costs")
        df = ensure_jev_sentiment_columns(pd.DataFrame({"Group ID": [1]}), tag_definitions=tag_definitions)
        combined = parse_jev_combined_sentiment_response(
            self.tag_response(),
            tag_definitions=tag_definitions,
            tagging_mode="Single best tag",
        )

        updated = apply_jev_combined_result_to_unique_df(
            df,
            0,
            combined,
            tag_definitions=tag_definitions,
            tagging_mode="Single best tag",
        )

        self.assertEqual(updated.loc[0, "Jev Tags"], "Access")
        self.assertEqual(updated.loc[0, "Jev Tag Count"], 1)
        self.assertEqual(updated.loc[0, "Jev Best Tag"], "Access")
        self.assertEqual(updated.loc[0, "Jev Tag [Access]"], "YES")
        self.assertEqual(updated.loc[0, "Jev Tag [Affordability]"], "NO")
        self.assertAlmostEqual(updated.loc[0, "Jev Tag Probability [Affordability]"], 0.41)
        self.assertAlmostEqual(updated.loc[0, "Jev Best Tag Probability [Other]"], 0.10)

        recomputed = recompute_jev_official_tag_assignments(
            updated,
            tag_definitions=tag_definitions,
            tagging_mode="Multiple applicable tags",
        )
        self.assertEqual(recomputed.loc[0, "Jev Tagging Mode"], "Multiple applicable tags")
        self.assertEqual(recomputed.loc[0, "Jev Tags"], "Access")

    def test_success_and_error_batch_progression_skips_failed_rows(self) -> None:
        df = ensure_jev_sentiment_columns(
            pd.DataFrame(
                {
                    "Group ID": [1, 2, 3],
                    "Headline": ["Good story", "Broken story", "Neutral story"],
                    "Snippet": ["College Board praised.", "College Board mentioned.", "College Board announced dates."],
                }
            )
        )
        batch = get_remaining_jev_sentiment_rows(df).iloc[:2].copy()

        def fake_call(payload, api_key):
            self.assertEqual(
                set(payload["questions"].keys()),
                {"sentiment", "sentiment_5_way", "sentiment_score", "sentiment_relevant", "sentiment_mixture"},
            )
            headline = payload["state"]["story"]["headline"]
            if headline == "Broken story":
                raise RuntimeError("temporary outage")
            return self.combined_response()

        updated, summary = run_jev_sentiment_batch(
            df,
            batch,
            self.analysis_payload,
            "test-key",
            call_fn=fake_call,
        )

        self.assertEqual(summary["done"], 2)
        self.assertEqual(summary["successful"], 1)
        self.assertEqual(len(summary["errors"]), 1)
        self.assertEqual(updated.loc[0, "Jev Sentiment"], "POSITIVE")
        self.assertEqual(updated.loc[0, "Jev 5-Way Sentiment"], "SOMEWHAT POSITIVE")
        self.assertAlmostEqual(updated.loc[0, "Jev Sentiment Score"], 2.3)
        self.assertEqual(updated.loc[0, "Jev Mixture Label"], "MOSTLY CONSISTENT")
        self.assertIn("temporary outage", updated.loc[1, "Jev Error"])

        remaining = get_remaining_jev_sentiment_rows(updated)
        self.assertEqual(remaining["Group ID"].tolist(), [3])

    def test_batch_uses_one_combined_api_call_per_story(self) -> None:
        df = ensure_jev_sentiment_columns(
            pd.DataFrame(
                {
                    "Group ID": [1, 2, 3],
                    "Headline": ["Mild praise", "Neutral story", "Negative story"],
                    "Snippet": [
                        "College Board program helped students.",
                        "College Board announced dates.",
                        "College Board faced criticism.",
                    ],
                }
            )
        )
        batch = get_remaining_jev_sentiment_rows(df).copy()
        calls = []

        def fake_call(payload, api_key):
            calls.append(payload)
            return self.combined_response()

        updated, summary = run_jev_sentiment_batch(
            df,
            batch,
            self.analysis_payload,
            "test-key",
            call_fn=fake_call,
        )

        self.assertEqual(summary["done"], 3)
        self.assertEqual(summary["successful"], 3)
        self.assertEqual(len(calls), 3)
        self.assertTrue(all(len(call["questions"]) == 5 for call in calls))
        self.assertEqual(len(get_remaining_jev_sentiment_rows(updated)), 0)

    def test_batch_uses_one_combined_api_call_per_story_with_tags(self) -> None:
        tag_definitions = parse_jev_tag_definitions("Access: Availability\nAffordability: Costs")
        df = ensure_jev_sentiment_columns(
            pd.DataFrame(
                {
                    "Group ID": [1, 2],
                    "Headline": ["Access story", "Other story"],
                    "Snippet": ["College Board access.", "College Board dates."],
                }
            ),
            tag_definitions=tag_definitions,
        )
        batch = get_remaining_jev_sentiment_rows(df, tag_definitions=tag_definitions).copy()
        calls = []

        def fake_call(payload, api_key):
            calls.append(payload)
            return self.tag_response()

        updated, summary = run_jev_sentiment_batch(
            df,
            batch,
            self.analysis_payload,
            "test-key",
            call_fn=fake_call,
            max_workers=2,
            tag_definitions=tag_definitions,
            tagging_mode="Multiple applicable tags",
        )

        self.assertEqual(summary["done"], 2)
        self.assertEqual(summary["successful"], 2)
        self.assertEqual(len(calls), 2)
        self.assertTrue(all(len(call["questions"]) == 8 for call in calls))
        self.assertTrue(all("tag_best_fit" in call["questions"] for call in calls))
        self.assertEqual(updated["Jev Tags"].tolist(), ["Access", "Access"])
        self.assertEqual(len(get_remaining_jev_sentiment_rows(updated, tag_definitions=tag_definitions)), 0)

    def test_threaded_batch_maps_out_of_order_results_to_original_group_ids(self) -> None:
        df = ensure_jev_sentiment_columns(
            pd.DataFrame(
                {
                    "Group ID": [101, 102, 103],
                    "Headline": ["Slow praise", "Fast neutral", "Middle negative"],
                    "Snippet": [
                        "College Board program helped students.",
                        "College Board announced dates.",
                        "College Board faced criticism.",
                    ],
                }
            )
        )
        batch = get_remaining_jev_sentiment_rows(df).copy()
        calls = []

        def fake_call(payload, api_key):
            calls.append(payload)
            headline = payload["state"]["story"]["headline"]
            if headline == "Slow praise":
                time.sleep(0.04)
                return self.combined_response(three_way="POSITIVE")
            if headline == "Middle negative":
                time.sleep(0.02)
                return self.combined_response(three_way="NEGATIVE")
            return self.combined_response(three_way="NEUTRAL")

        updated, summary = run_jev_sentiment_batch(
            df,
            batch,
            self.analysis_payload,
            "test-key",
            call_fn=fake_call,
            max_workers=3,
        )

        self.assertEqual(summary["done"], 3)
        self.assertEqual(summary["successful"], 3)
        self.assertEqual(len(calls), 3)
        self.assertTrue(all(len(call["questions"]) == 5 for call in calls))
        self.assertEqual(updated.loc[0, "Group ID"], 101)
        self.assertEqual(updated.loc[0, "Jev Sentiment"], "POSITIVE")
        self.assertEqual(updated.loc[1, "Group ID"], 102)
        self.assertEqual(updated.loc[1, "Jev Sentiment"], "NEUTRAL")
        self.assertEqual(updated.loc[2, "Group ID"], 103)
        self.assertEqual(updated.loc[2, "Jev Sentiment"], "NEGATIVE")

    def test_threaded_batch_progress_advances_for_success_and_error(self) -> None:
        df = ensure_jev_sentiment_columns(
            pd.DataFrame(
                {
                    "Group ID": [1, 2, 3],
                    "Headline": ["Good story", "Broken story", "Neutral story"],
                    "Snippet": ["College Board praised.", "College Board mentioned.", "College Board announced dates."],
                }
            )
        )
        batch = get_remaining_jev_sentiment_rows(df).copy()
        progress_events = []

        def fake_call(payload, api_key):
            if payload["state"]["story"]["headline"] == "Broken story":
                raise RuntimeError("temporary outage")
            return self.combined_response()

        updated, summary = run_jev_sentiment_batch(
            df,
            batch,
            self.analysis_payload,
            "test-key",
            call_fn=fake_call,
            max_workers=3,
            progress_callback=lambda done, total: progress_events.append((done, total)),
        )

        self.assertEqual(summary["done"], 3)
        self.assertEqual(summary["successful"], 2)
        self.assertEqual(len(summary["errors"]), 1)
        self.assertEqual(progress_events[-1], (3, 3))
        self.assertEqual(sorted(done for done, _ in progress_events), [1, 2, 3])
        self.assertEqual(updated.loc[0, "Jev Sentiment"], "POSITIVE")
        self.assertIn("temporary outage", updated.loc[1, "Jev Error"])
        self.assertEqual(updated.loc[2, "Jev Sentiment"], "POSITIVE")

    def test_openrouter_retry_behavior_is_preserved_for_rate_limit_responses(self) -> None:
        class FakeResponse:
            def __init__(self, status_code: int, body: dict | None = None) -> None:
                self.status_code = status_code
                self._body = body or {}
                self.headers = {}

            def raise_for_status(self) -> None:
                if self.status_code >= 400:
                    raise RuntimeError(f"HTTP {self.status_code}")

            def json(self) -> dict:
                return self._body

        responses = [
            FakeResponse(429),
            FakeResponse(200, self.combined_response()),
        ]

        with patch("processing.jev_sentiment.requests.post", side_effect=responses) as post:
            with patch("processing.jev_sentiment.time.sleep") as sleep:
                parsed = call_jev_sentiment(
                    {"model": DEFAULT_JEV_MODEL, "state": {}, "questions": {}},
                    "test-key",
                    max_retries=1,
                )

        self.assertEqual(parsed["model"], "typesafe/jev-1.13-20260917")
        self.assertEqual(post.call_count, 2)
        sleep.assert_called_once()

    def test_display_includes_existing_sentiment_columns_for_comparison(self) -> None:
        df = ensure_jev_sentiment_columns(
            pd.DataFrame(
                {
                    "Group ID": [1],
                    "Headline": ["Comparison story"],
                    "AI Sentiment": ["NEUTRAL"],
                    "AI Sentiment Confidence": [0.91],
                    "Assigned Sentiment": ["POSITIVE"],
                    "Jev Sentiment": ["POSITIVE"],
                    "Jev Selected Probability": [0.88],
                }
            )
        )

        display = build_jev_results_display(df)

        self.assertIn("AI Sentiment", display.columns)
        self.assertIn("Assigned Sentiment", display.columns)
        self.assertEqual(display.loc[0, "AI Sentiment"], "NEUTRAL")
        self.assertEqual(display.loc[0, "Jev Sentiment"], "POSITIVE")

    def test_download_exports_merge_and_preserve_rapid_labeling_columns(self) -> None:
        sentiment_rows = pd.DataFrame(
            {
                "Group ID": [1, 2],
                "Headline": ["Shared story", "Sentiment only story"],
                "AI Sentiment": ["NEUTRAL", "NEGATIVE"],
                "AI Sentiment Confidence": [0.8, 0.7],
            }
        )
        sentiment_unique = sentiment_rows.copy()
        jev_unique = ensure_jev_sentiment_columns(
            pd.DataFrame(
                {
                    "Group ID": [1],
                    "Headline": ["Shared story"],
                    "AI Sentiment": ["NEUTRAL"],
                    "Jev Sentiment": ["POSITIVE"],
                    "Jev Selected Probability": [0.88],
                    "Jev Model": ["typesafe/jev-1.13-20260917"],
                }
            )
        )
        session_state = {
            "df_sentiment_rows": sentiment_rows,
            "df_sentiment_unique": sentiment_unique,
            "df_jev_sentiment_unique": jev_unique,
            "sentiment_config_step": True,
            "sentiment_sample_mode": "representative",
        }

        sentiment_export = build_sentiment_sample_export(session_state, include_audit=True)
        jev_export = build_jev_sentiment_sample_export(session_state, include_audit=True)

        self.assertIn("Rapid Sentiment", sentiment_export.columns)
        self.assertNotIn("Jev Sentiment", sentiment_export.columns)
        self.assertEqual(sentiment_export.loc[sentiment_export["Group ID"] == 1, "Rapid Sentiment"].iloc[0], "POSITIVE")
        self.assertEqual(sentiment_export.loc[sentiment_export["Group ID"] == 2, "AI Sentiment"].iloc[0], "NEGATIVE")
        self.assertIn("AI Sentiment", jev_export.columns)
        self.assertIn("Rapid Labeling Raw Response", jev_export.columns)
        self.assertNotIn("Jev Raw Response", jev_export.columns)
        self.assertEqual(jev_export.loc[0, "Rapid Sentiment"], "POSITIVE")

    def test_sentiment_sample_omits_jev_request_metadata_on_repeated_mentions(self) -> None:
        sentiment_rows = pd.DataFrame(
            {
                "Group ID": [1, 1],
                "Headline": ["Shared story mention 1", "Shared story mention 2"],
                "AI Sentiment": ["NEUTRAL", "NEUTRAL"],
                "AI Sentiment Confidence": [80, 80],
            }
        )
        sentiment_unique = pd.DataFrame(
            {
                "Group ID": [1],
                "Headline": ["Shared story"],
                "AI Sentiment": ["NEUTRAL"],
                "AI Sentiment Confidence": [80],
            }
        )
        jev_unique = ensure_jev_sentiment_columns(
            pd.DataFrame(
                {
                    "Group ID": [1],
                    "Headline": ["Shared story"],
                    "Jev Sentiment": ["POSITIVE"],
                    "Jev Selected Probability": [0.88],
                    "Jev Confidence": [0.84],
                    "Jev 5-Way Sentiment": ["SOMEWHAT POSITIVE"],
                    "Jev Sentiment Score": [2.3],
                    "Jev Score Relevant Probability": [0.99],
                    "Jev Mixture Label": ["VERY CONSISTENT"],
                    "Jev Tags": ["Access"],
                    "Jev Tag Count": [1],
                    "Jev Best Tag": ["Access"],
                    "Jev Tag [Access]": ["YES"],
                    "Jev Tag Probability [Access]": [0.74],
                    "Jev Model": ["typesafe/jev-1.13-20260917"],
                    "Jev Input Tokens": [1200],
                    "Jev Cost USD": [0.00009],
                    "Jev Error": [pd.NA],
                    "Jev Raw Response": ['{"answers": {}}'],
                }
            )
        )
        session_state = {
            "df_sentiment_rows": sentiment_rows,
            "df_sentiment_unique": sentiment_unique,
            "df_jev_sentiment_unique": jev_unique,
            "sentiment_config_step": True,
            "sentiment_sample_mode": "representative",
        }

        sentiment_export = build_sentiment_sample_export(session_state, include_audit=True)
        jev_export = build_jev_sentiment_sample_export(session_state, include_audit=True)

        self.assertEqual(len(sentiment_export), 2)
        self.assertEqual(sentiment_export["Rapid Sentiment"].tolist(), ["POSITIVE", "POSITIVE"])
        self.assertEqual(sentiment_export["Rapid Sentiment Score"].tolist(), [2.3, 2.3])
        self.assertEqual(sentiment_export["Rapid Sentiment Mixture Label"].tolist(), ["VERY CONSISTENT", "VERY CONSISTENT"])
        self.assertEqual(sentiment_export["Rapid Tags"].tolist(), ["Access", "Access"])
        self.assertEqual(sentiment_export["Rapid Tag [Access]"].tolist(), ["YES", "YES"])
        self.assertEqual(sentiment_export["Rapid Tag Probability [Access]"].tolist(), [0.74, 0.74])
        for column in ["Rapid Labeling Model", "Rapid Labeling Input Tokens", "Rapid Labeling Cost USD", "Rapid Labeling Error", "Rapid Labeling Raw Response"]:
            self.assertNotIn(column, sentiment_export.columns)

        self.assertEqual(len(jev_export), 1)
        self.assertIn("Rapid Labeling Model", jev_export.columns)
        self.assertIn("Rapid Labeling Input Tokens", jev_export.columns)
        self.assertIn("Rapid Labeling Cost USD", jev_export.columns)
        self.assertIn("Rapid Labeling Raw Response", jev_export.columns)
        self.assertIn("Rapid Tags", jev_export.columns)
        self.assertIn("Rapid Tag Probability [Access]", jev_export.columns)
        self.assertEqual(int(jev_export["Rapid Labeling Input Tokens"].sum()), 1200)
        self.assertAlmostEqual(float(jev_export["Rapid Labeling Cost USD"].sum()), 0.00009)

    def _rapid_export_state(
        self,
        rows: pd.DataFrame,
        *,
        tag_definitions: dict[str, str] | None = None,
    ) -> dict:
        return {
            "df_jev_sentiment_unique": ensure_jev_sentiment_columns(rows),
            "jev_tag_definitions": tag_definitions or {"Access": "Access to education."},
        }

    def test_clean_trad_cascades_final_rapid_values_by_canonical_group_id(self) -> None:
        rapid_rows = pd.DataFrame(
            {
                "Group ID": [10, 20],
                "Jev Sentiment": ["POSITIVE", pd.NA],
                "Jev 5-Way Sentiment": ["SOMEWHAT POSITIVE", pd.NA],
                "Jev Sentiment Score": [2.5, pd.NA],
                "Jev Best Tag": ["Access", pd.NA],
                "Jev Tags": ["Access", pd.NA],
                "Jev Error": [pd.NA, "request failed"],
            }
        )
        traditional = pd.DataFrame(
            {
                "Group ID": [10, 10, 20, 30],
                "Headline": ["First mention", "Second mention", "Failed story", "Unlabeled story"],
                "Final Sentiment": ["NEUTRAL", "NEUTRAL", "NEGATIVE", "POSITIVE"],
                "Final Tag": ["Legacy", "Legacy", "Legacy", "Legacy"],
            }
        )

        exported = merge_full_scope_ai_columns_into_clean_trad(
            self._rapid_export_state(rapid_rows),
            traditional,
        )

        expected_core = {
            "Final Rapid Relevance",
            "Final Rapid Sentiment 3-Way",
            "Final Rapid Sentiment 5-Way",
            "Final Rapid Sentiment Score",
            "Final Rapid Best Tag",
            "Final Rapid Tags",
        }
        self.assertTrue(expected_core.issubset(exported.columns))
        group_ten = exported[exported["Group ID"] == 10]
        self.assertEqual(group_ten["Final Rapid Sentiment 3-Way"].tolist(), ["POSITIVE", "POSITIVE"])
        self.assertEqual(group_ten["Final Rapid Best Tag"].tolist(), ["Access", "Access"])
        self.assertEqual(group_ten["Final Rapid Relevance"].tolist(), ["RELEVANT", "RELEVANT"])
        self.assertEqual(exported.loc[exported["Group ID"] == 20, "Final Rapid Sentiment 3-Way"].iloc[0], "")
        self.assertTrue(pd.isna(exported.loc[exported["Group ID"] == 30, "Final Rapid Sentiment 3-Way"].iloc[0]))
        self.assertEqual(exported.loc[0, "Final Sentiment"], "NEUTRAL")
        self.assertEqual(exported.loc[0, "Final Tag"], "Legacy")

    def test_clean_trad_rapid_final_respects_human_scale_authority_and_conflict(self) -> None:
        rapid_rows = pd.DataFrame(
            {
                "Group ID": [1, 2, 3, 4],
                "Jev Sentiment": ["POSITIVE", "POSITIVE", "POSITIVE", "POSITIVE"],
                "Jev 5-Way Sentiment": [
                    "SOMEWHAT POSITIVE",
                    "SOMEWHAT POSITIVE",
                    "SOMEWHAT POSITIVE",
                    "NOT RELEVANT",
                ],
                "Jev Sentiment Score": [2.0, 2.0, 2.0, 2.0],
                "Rapid Human Sentiment Review State": ["Assigned", "Assigned", "Assigned", pd.NA],
                "Rapid Human Sentiment Scale": ["3-way", "5-way", "3-way", pd.NA],
                "Rapid Human Sentiment": ["NEGATIVE", "VERY NEGATIVE", "NOT RELEVANT", pd.NA],
            }
        )
        traditional = pd.DataFrame({"Group ID": [1, 2, 3, 4]})

        exported = merge_full_scope_ai_columns_into_clean_trad(
            self._rapid_export_state(rapid_rows),
            traditional,
        ).set_index("Group ID")

        self.assertEqual(exported.loc[1, "Final Rapid Sentiment 3-Way"], "NEGATIVE")
        self.assertEqual(exported.loc[1, "Final Rapid Sentiment 5-Way"], "SOMEWHAT POSITIVE")
        self.assertEqual(exported.loc[2, "Final Rapid Sentiment 3-Way"], "POSITIVE")
        self.assertEqual(exported.loc[2, "Final Rapid Sentiment 5-Way"], "VERY NEGATIVE")
        self.assertEqual(exported.loc[3, "Final Rapid Relevance"], "NOT_RELEVANT")
        self.assertEqual(exported.loc[3, "Final Rapid Sentiment 3-Way"], "NOT RELEVANT")
        self.assertEqual(exported.loc[3, "Final Rapid Sentiment 5-Way"], "NOT RELEVANT")
        self.assertTrue(pd.isna(exported.loc[3, "Final Rapid Sentiment Score"]))
        self.assertEqual(exported.loc[4, "Final Rapid Sentiment 3-Way"], "")
        self.assertEqual(exported.loc[4, "Final Rapid Sentiment 5-Way"], "")

    def test_clean_trad_rapid_tag_formulations_remain_independent(self) -> None:
        rapid_rows = pd.DataFrame(
            {
                "Group ID": [1],
                "Jev Sentiment": ["NEUTRAL"],
                "Jev 5-Way Sentiment": ["NEUTRAL"],
                "Jev Sentiment Score": [0.0],
                "Jev Best Tag": ["Access"],
                "Jev Tags": ["Access"],
                "Rapid Human Best Tag Review State": ["Assigned"],
                "Rapid Human Best Tag Assignment": ["Programs"],
                "Rapid Human Applicable Tags Review State": ["Assigned"],
                "Rapid Human Applicable Tags Assignment": ["Access; Programs"],
            }
        )
        state = self._rapid_export_state(
            rapid_rows,
            tag_definitions={"Access": "Access", "Programs": "Programs"},
        )
        exported = merge_full_scope_ai_columns_into_clean_trad(state, pd.DataFrame({"Group ID": [1]}))

        self.assertEqual(exported.loc[0, "Final Rapid Best Tag"], "Programs")
        self.assertEqual(exported.loc[0, "Final Rapid Tags"], "Access; Programs")

    def test_clean_trad_rapid_audit_columns_exclude_request_metadata(self) -> None:
        rapid_rows = pd.DataFrame(
            {
                "Group ID": [1],
                "Jev Sentiment": ["POSITIVE"],
                "Jev 5-Way Sentiment": ["SOMEWHAT POSITIVE"],
                "Jev Sentiment Score": [2.5],
                "Jev Best Tag": ["Access"],
                "Jev Tags": ["Access"],
                "Jev Confidence": [0.9],
                "Jev Model": ["typesafe/jev-1.13"],
                "Jev Input Tokens": [123],
                "Jev Cost USD": [0.0001],
                "Jev Raw Response": ["{raw}"],
                "Jev Error": [pd.NA],
                "Rapid Review Status": ["COMPLETED"],
                "Rapid Review 3-Way Sentiment": ["POSITIVE"],
                "Rapid Review 5-Way Sentiment": ["SOMEWHAT POSITIVE"],
                "Rapid Review Sentiment Outcome": ["POSITIVE"],
                "Rapid Review Sentiment Score": [2.0],
                "Rapid Review Sentiment Rationale": ["Review rationale"],
                "Rapid Review Model": ["gpt-5.6-luna"],
                "Rapid Review Input Tokens": [45],
                "Rapid Review Cost USD": [0.0002],
                "Rapid Review Raw Response": ["{review raw}"],
                "Rapid Review Error": [pd.NA],
                "Rapid Human Sentiment Review State": ["Assigned"],
                "Rapid Human Sentiment Scale": ["3-way"],
                "Rapid Human Sentiment": ["NEGATIVE"],
            }
        )
        state = self._rapid_export_state(rapid_rows)
        traditional = pd.DataFrame({"Group ID": [1]})

        default_export = merge_full_scope_ai_columns_into_clean_trad(state, traditional)
        audit_export = merge_full_scope_ai_columns_into_clean_trad(
            state,
            traditional,
            include_labeling_audit_columns=True,
        )

        self.assertEqual(
            {column for column in default_export.columns if column.startswith("Rapid ") or column.startswith("Final Rapid")},
            {
                "Final Rapid Relevance",
                "Final Rapid Sentiment 3-Way",
                "Final Rapid Sentiment 5-Way",
                "Final Rapid Sentiment Score",
                "Final Rapid Best Tag",
                "Final Rapid Tags",
            },
        )
        self.assertIn("Effective Rapid Sentiment 3-Way", audit_export.columns)
        self.assertIn("Rapid Sentiment", audit_export.columns)
        self.assertIn("Rapid Review 3-Way Sentiment", audit_export.columns)
        self.assertIn("Rapid Review Sentiment Rationale", audit_export.columns)
        self.assertIn("Rapid Human Sentiment", audit_export.columns)
        for column in [
            "Rapid Labeling Model",
            "Rapid Labeling Input Tokens",
            "Rapid Labeling Cost USD",
            "Rapid Labeling Raw Response",
            "Rapid Labeling Error",
            "Rapid Review Model",
            "Rapid Review Input Tokens",
            "Rapid Review Cost USD",
            "Rapid Review Raw Response",
            "Rapid Review Error",
        ]:
            self.assertNotIn(column, audit_export.columns)

        grouped_export = build_jev_sentiment_sample_export(state, include_audit=True)
        for column in [
            "Rapid Labeling Model",
            "Rapid Labeling Input Tokens",
            "Rapid Labeling Cost USD",
            "Rapid Labeling Raw Response",
        ]:
            self.assertIn(column, grouped_export.columns)

    def test_inactive_rapid_state_removes_clean_trad_columns_and_saved_state_exports(self) -> None:
        rapid_rows = pd.DataFrame(
            {
                "Group ID": [7],
                "Jev Sentiment": ["NEUTRAL"],
                "Jev 5-Way Sentiment": ["NEUTRAL"],
                "Jev Sentiment Score": [0.0],
            }
        )
        saved_state = self._rapid_export_state(rapid_rows.copy(deep=True))
        restored_export = merge_full_scope_ai_columns_into_clean_trad(
            saved_state,
            pd.DataFrame({"Group ID": [7]}),
        )
        self.assertEqual(restored_export.loc[0, "Final Rapid Sentiment 3-Way"], "NEUTRAL")

        reset_state = {"df_jev_sentiment_unique": ensure_jev_sentiment_columns(pd.DataFrame({"Group ID": [7]}))}
        stale = restored_export.copy()
        cleaned = remove_inactive_workflow_columns(stale, reset_state)
        self.assertNotIn("Final Rapid Sentiment 3-Way", cleaned.columns)
        self.assertNotIn("Final Rapid Best Tag", cleaned.columns)

    def test_export_metadata_uses_rapid_labeling_sheet_name(self) -> None:
        jev_unique = ensure_jev_sentiment_columns(
            pd.DataFrame(
                {
                    "Group ID": [1],
                    "Jev Sentiment": ["POSITIVE"],
                    "Jev Raw Response": ['{"answers": {}}'],
                }
            )
        )
        metadata = build_export_metadata_sheet({"df_jev_sentiment_unique": jev_unique})
        value_by_field = dict(zip(metadata["Field"], metadata["Value"]))

        self.assertEqual(value_by_field["Rapid Labeling Sample Sheet"], "RAPID LABELING SAMPLE")
        self.assertNotIn("Jev Experimental Sample Sheet", value_by_field)

    def test_sentiment_export_preserves_raw_first_pass_and_review_fields(self) -> None:
        sentiment_rows = pd.DataFrame(
            {
                "Group ID": [1],
                "Headline": ["Disagreement story"],
                "AI Sentiment": ["NEGATIVE"],
                "AI Sentiment Confidence": [78],
                "AI Sentiment Rationale": ["First pass saw criticism."],
                "Review AI Sentiment": ["NEUTRAL"],
                "Review AI Confidence": [96],
                "Review AI Rationale": ["Review saw factual coverage."],
                "AI Agreement": ["Disagree"],
                "Needs Human Review": ["Yes"],
            }
        )
        session_state = {
            "df_sentiment_rows": sentiment_rows,
            "df_sentiment_unique": sentiment_rows.copy(),
            "sentiment_config_step": True,
            "sentiment_sample_mode": "representative",
        }

        sentiment_export = build_sentiment_sample_export(session_state, include_audit=True)

        self.assertEqual(sentiment_export.loc[0, "AI Sentiment"], "NEGATIVE")
        self.assertEqual(sentiment_export.loc[0, "Review AI Sentiment"], "NEUTRAL")
        self.assertEqual(sentiment_export.loc[0, "AI Agreement"], "Disagree")
        self.assertEqual(sentiment_export.loc[0, "Needs Human Review"], "Yes")
        self.assertEqual(sentiment_export.loc[0, "Effective AI Sentiment"], "NEUTRAL")
        self.assertEqual(sentiment_export.loc[0, "Final Sentiment"], "NEUTRAL")
        self.assertEqual(sentiment_export.loc[0, "Final Sentiment Source"], "Review AI")

    def test_sentiment_export_assigned_override_sets_final_source(self) -> None:
        sentiment_rows = pd.DataFrame(
            {
                "Group ID": [1],
                "Headline": ["Assigned story"],
                "AI Sentiment": ["NEGATIVE"],
                "AI Sentiment Confidence": [78],
                "Review AI Sentiment": ["NEUTRAL"],
                "Review AI Confidence": [96],
                "AI Agreement": ["Disagree"],
                "Needs Human Review": ["Yes"],
                "Assigned Sentiment": ["POSITIVE"],
            }
        )
        session_state = {
            "df_sentiment_rows": sentiment_rows,
            "df_sentiment_unique": sentiment_rows.copy(),
            "sentiment_config_step": True,
            "sentiment_sample_mode": "representative",
        }

        sentiment_export = build_sentiment_sample_export(session_state, include_audit=True)

        self.assertEqual(sentiment_export.loc[0, "AI Sentiment"], "NEGATIVE")
        self.assertEqual(sentiment_export.loc[0, "Review AI Sentiment"], "NEUTRAL")
        self.assertEqual(sentiment_export.loc[0, "Assigned Sentiment"], "POSITIVE")
        self.assertEqual(sentiment_export.loc[0, "Final Sentiment"], "POSITIVE")
        self.assertEqual(sentiment_export.loc[0, "Final Sentiment Source"], "Assigned")

    def test_jev_export_merges_current_production_sentiment_comparison_fields(self) -> None:
        sentiment_unique = pd.DataFrame(
            {
                "Group ID": [1],
                "AI Sentiment": ["NEGATIVE"],
                "AI Sentiment Confidence": [78],
                "Review AI Sentiment": ["NEUTRAL"],
                "Review AI Confidence": [96],
                "AI Agreement": ["Disagree"],
                "Needs Human Review": ["Yes"],
            }
        )
        jev_unique = ensure_jev_sentiment_columns(
            pd.DataFrame(
                {
                    "Group ID": [1],
                    "Headline": ["Shared story"],
                    "AI Sentiment": ["STALE"],
                    "Jev Sentiment": ["POSITIVE"],
                    "Jev Model": ["typesafe/jev-1.13-20260917"],
                }
            )
        )
        session_state = {
            "df_sentiment_unique": sentiment_unique,
            "df_jev_sentiment_unique": jev_unique,
        }

        jev_export = build_jev_sentiment_sample_export(session_state, include_audit=True)

        self.assertEqual(jev_export.loc[0, "AI Sentiment"], "NEGATIVE")
        self.assertEqual(jev_export.loc[0, "Review AI Sentiment"], "NEUTRAL")
        self.assertEqual(jev_export.loc[0, "Effective AI Sentiment"], "NEUTRAL")
        self.assertEqual(jev_export.loc[0, "Final Sentiment Source"], "Review AI")
        self.assertEqual(jev_export.loc[0, "Rapid Sentiment"], "POSITIVE")

    def _labeling_path_session(self, *, sentiment: bool = False, tagging: bool = False, rapid: bool = False) -> dict:
        session_state: dict = {}
        if sentiment:
            sent = pd.DataFrame(
                {
                    "Group ID": [1],
                    "Headline": ["Sentiment story"],
                    "AI Sentiment": ["NEGATIVE"],
                    "AI Sentiment Confidence": [78],
                    "AI Sentiment Rationale": ["First pass rationale"],
                    "Review AI Sentiment": ["NEUTRAL"],
                    "Review AI Confidence": [96],
                    "Review AI Rationale": ["Review rationale"],
                    "AI Agreement": ["Disagree"],
                    "Needs Human Review": ["Yes"],
                }
            )
            session_state.update(
                {
                    "df_sentiment_rows": sent.copy(),
                    "df_sentiment_unique": sent.copy(),
                    "sentiment_config_step": True,
                    "sentiment_sample_mode": "representative",
                }
            )
        if tagging:
            tag = pd.DataFrame(
                {
                    "Group ID": [1],
                    "Headline": ["Tagging story"],
                    "AI Tag": ["Access"],
                    "AI Tag Confidence": [88],
                    "AI Tag Rationale": ["Tag rationale"],
                    "Review AI Tag": ["Access"],
                    "Review AI Confidence": [91],
                    "Review AI Rationale": ["Review tag rationale"],
                    "Tag_Processed": [True],
                }
            )
            session_state.update(
                {
                    "df_tagging_rows": tag.copy(),
                    "df_tagging_unique": tag.copy(),
                    "tagging_config_step": True,
                    "tagging_sample_mode": "representative",
                }
            )
        if rapid:
            session_state["df_jev_sentiment_unique"] = ensure_jev_sentiment_columns(
                pd.DataFrame(
                    {
                        "Group ID": [1],
                        "Headline": ["Rapid story"],
                        "Jev Sentiment": ["POSITIVE"],
                        "Jev Selected Probability": [0.88],
                        "Jev Confidence": [0.84],
                        "Jev 5-Way Sentiment": ["SOMEWHAT POSITIVE"],
                        "Jev 5-Way Selected Probability": [0.7],
                        "Jev Sentiment Score": [2.3],
                        "Jev Score Relevant Probability": [0.99],
                        "Jev Mixture Label": ["VERY CONSISTENT"],
                        "Jev Mixture Score": [0.0],
                        "Jev Tags": ["Access"],
                        "Jev Tag Count": [1],
                        "Jev Best Tag": ["Access"],
                        "Jev Model": ["typesafe/jev-1.13-20260917"],
                        "Jev Input Tokens": [1200],
                        "Jev Cost USD": [0.00009],
                        "Jev Raw Response": ['{"answers": {}}'],
                    }
                )
            )
        return session_state

    def test_labeling_workflow_exports_are_conditional_by_completed_path_and_audit_flag(self) -> None:
        combinations = [
            (False, False, False),
            (True, False, False),
            (False, True, False),
            (False, False, True),
            (True, True, False),
            (True, False, True),
            (False, True, True),
            (True, True, True),
        ]

        for sentiment, tagging, rapid in combinations:
            with self.subTest(sentiment=sentiment, tagging=tagging, rapid=rapid, audit=False):
                session_state = self._labeling_path_session(sentiment=sentiment, tagging=tagging, rapid=rapid)
                self.assertEqual(labeling_audit_available(session_state), sentiment or tagging or rapid)

                sent_export = build_sentiment_sample_export(session_state, include_audit=False)
                tag_export = build_tagging_sample_export(session_state, include_audit=False)
                rapid_export = build_jev_sentiment_sample_export(session_state, include_audit=False)

                self.assertEqual(not sent_export.empty, sentiment)
                self.assertEqual(not tag_export.empty, tagging)
                self.assertEqual(not rapid_export.empty, rapid)

                if sentiment:
                    self.assertIn("Final Sentiment", sent_export.columns)
                    self.assertNotIn("AI Sentiment Confidence", sent_export.columns)
                    self.assertNotIn("Final Tag", sent_export.columns)
                    if rapid:
                        self.assertIn("Rapid Sentiment", sent_export.columns)
                        self.assertNotIn("Rapid Sentiment Confidence", sent_export.columns)
                if tagging:
                    self.assertIn("Final Tag", tag_export.columns)
                    self.assertNotIn("AI Tag Confidence", tag_export.columns)
                    self.assertNotIn("Final Sentiment", tag_export.columns)
                if rapid:
                    self.assertIn("Rapid Sentiment", rapid_export.columns)
                    self.assertIn("Rapid Tags", rapid_export.columns)
                    self.assertNotIn("Rapid Sentiment Confidence", rapid_export.columns)
                    self.assertNotIn("Rapid Labeling Model", rapid_export.columns)
                    if not sentiment:
                        self.assertNotIn("AI Sentiment", rapid_export.columns)

            with self.subTest(sentiment=sentiment, tagging=tagging, rapid=rapid, audit=True):
                session_state = self._labeling_path_session(sentiment=sentiment, tagging=tagging, rapid=rapid)
                sent_export = build_sentiment_sample_export(session_state, include_audit=True)
                tag_export = build_tagging_sample_export(session_state, include_audit=True)
                rapid_export = build_jev_sentiment_sample_export(session_state, include_audit=True)

                if sentiment:
                    self.assertIn("AI Sentiment Confidence", sent_export.columns)
                    self.assertIn("Review AI Sentiment", sent_export.columns)
                if tagging:
                    self.assertIn("AI Tag Confidence", tag_export.columns)
                    self.assertIn("Review AI Tag", tag_export.columns)
                if rapid:
                    self.assertIn("Rapid Sentiment Confidence", rapid_export.columns)
                    self.assertIn("Rapid Labeling Model", rapid_export.columns)
                if rapid and not sentiment:
                    self.assertNotIn("AI Sentiment", rapid_export.columns)

    def test_initialized_empty_labeling_columns_do_not_count_as_completed_output(self) -> None:
        session_state = {
            "df_sentiment_unique": pd.DataFrame({"Group ID": [1], "AI Sentiment": [pd.NA], "Assigned Sentiment": [pd.NA]}),
            "df_sentiment_rows": pd.DataFrame({"Group ID": [1], "AI Sentiment": [pd.NA]}),
            "df_tagging_unique": pd.DataFrame({"Group ID": [1], "Tag_Processed": [False], "AI Tag": [pd.NA]}),
            "df_tagging_rows": pd.DataFrame({"Group ID": [1], "AI Tag": [pd.NA]}),
            "df_jev_sentiment_unique": ensure_jev_sentiment_columns(pd.DataFrame({"Group ID": [1]})),
        }

        self.assertFalse(labeling_audit_available(session_state))
        self.assertTrue(build_sentiment_sample_export(session_state).empty)
        self.assertTrue(build_tagging_sample_export(session_state).empty)
        self.assertTrue(build_jev_sentiment_sample_export(session_state).empty)

        cleaned = remove_inactive_workflow_columns(
            pd.DataFrame(
                {
                    "Headline": ["Empty"],
                    "Final Sentiment": [pd.NA],
                    "Final Tag": [pd.NA],
                }
            ),
            session_state,
        )
        self.assertNotIn("Final Sentiment", cleaned.columns)
        self.assertNotIn("Final Tag", cleaned.columns)


if __name__ == "__main__":
    unittest.main()
