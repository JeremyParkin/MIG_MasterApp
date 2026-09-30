from __future__ import annotations

import unittest

import pandas as pd

from processing.rapid_insights import (
    RAPID_SENTIMENT_SCALE_3_WAY,
    RAPID_SENTIMENT_SCALE_5_WAY,
    build_rapid_sentiment_distribution,
    build_rapid_sentiment_observation_payload,
    build_rapid_tag_distribution,
    build_rapid_tag_observation_payload,
    rapid_observation_fingerprint,
)
from processing.rapid_resolution import (
    HUMAN_SENTIMENT_SCALE_3_WAY,
    HUMAN_STATE_ASSIGNED,
    RAPID_TAG_REVIEW_MODE_APPLICABLE,
    RAPID_TAG_REVIEW_MODE_BEST,
)


class RapidInsightsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tags = {
            "AP": "Advanced Placement coverage.",
            "SAT": "SAT or PSAT coverage.",
            "Career": "Career readiness coverage.",
        }

    def row(self, **overrides) -> dict:
        base = {
            "Group ID": 1,
            "Headline": "College Board story",
            "Snippet": "Story text.",
            "Outlet": "Example News",
            "Type": "Online",
            "URL": "https://example.com/story",
            "Date": "2026-09-01",
            "Group Count": 2,
            "Mentions": 3,
            "Impressions": 1000,
            "Effective Reach": 500,
            "Jev Sentiment": "NEGATIVE",
            "Jev 5-Way Sentiment": "SOMEWHAT NEGATIVE",
            "Jev Sentiment Score": -2,
            "Jev Best Tag": "SAT",
            "Jev Tags": "SAT",
            "Rapid Review Status": "",
            "Rapid Review Error": "",
        }
        base.update(overrides)
        return base

    def test_sentiment_distribution_uses_selected_scale_final_labels(self) -> None:
        df = pd.DataFrame(
            [
                self.row(
                    **{
                        "Rapid Human Sentiment Review State": HUMAN_STATE_ASSIGNED,
                        "Rapid Human Sentiment Scale": HUMAN_SENTIMENT_SCALE_3_WAY,
                        "Rapid Human Sentiment": "NEUTRAL",
                    }
                )
            ]
        )
        three = build_rapid_sentiment_distribution(df, scale=RAPID_SENTIMENT_SCALE_3_WAY)
        five = build_rapid_sentiment_distribution(df, scale=RAPID_SENTIMENT_SCALE_5_WAY)
        self.assertEqual(int(three.loc[three["Sentiment"] == "NEUTRAL", "Grouped Stories"].iloc[0]), 1)
        self.assertEqual(int(five.loc[five["Sentiment"] == "SOMEWHAT NEGATIVE", "Grouped Stories"].iloc[0]), 1)
        self.assertEqual(int(five.loc[five["Sentiment"] == "NEUTRAL", "Grouped Stories"].iloc[0]), 0)

    def test_not_relevant_exclusion_is_payload_specific(self) -> None:
        df = pd.DataFrame(
            [
                self.row(**{"Group ID": 1, "Jev Sentiment": "NOT RELEVANT", "Jev 5-Way Sentiment": "NOT RELEVANT"}),
                self.row(**{"Group ID": 2, "Jev Sentiment": "POSITIVE", "Jev 5-Way Sentiment": "SOMEWHAT POSITIVE"}),
            ]
        )
        payload = build_rapid_sentiment_observation_payload(
            df,
            scale=RAPID_SENTIMENT_SCALE_3_WAY,
            include_not_relevant=False,
        )
        labels = {record["Sentiment"] for record in payload["distribution"]}
        self.assertEqual(labels, {"POSITIVE"})

    def test_best_fit_tag_distribution_uses_final_best_tag_only(self) -> None:
        df = pd.DataFrame(
            [
                self.row(
                    **{
                        "Rapid Human Best Tag Review State": "Assigned",
                        "Rapid Human Best Tag Assignment": "AP",
                        "Rapid Human Applicable Tags Review State": "Assigned",
                        "Rapid Human Applicable Tags Assignment": "AP; SAT",
                    }
                )
            ]
        )
        dist = build_rapid_tag_distribution(
            df,
            mode=RAPID_TAG_REVIEW_MODE_BEST,
            tag_definitions=self.tags,
        )
        self.assertEqual(dist["Tag"].tolist(), ["AP"])
        self.assertEqual(int(dist.loc[0, "Grouped Stories"]), 1)

    def test_all_applicable_tag_distribution_expands_multiple_tags(self) -> None:
        df = pd.DataFrame(
            [
                self.row(
                    **{
                        "Rapid Human Applicable Tags Review State": "Assigned",
                        "Rapid Human Applicable Tags Assignment": "AP; SAT",
                    }
                )
            ]
        )
        dist = build_rapid_tag_distribution(
            df,
            mode=RAPID_TAG_REVIEW_MODE_APPLICABLE,
            tag_definitions=self.tags,
        )
        self.assertEqual(set(dist["Tag"]), {"AP", "SAT"})
        self.assertEqual(int(dist["Grouped Stories"].sum()), 2)

    def test_tag_payload_can_exclude_other(self) -> None:
        df = pd.DataFrame([self.row(**{"Jev Best Tag": "Other", "Jev Tags": "Other"})])
        payload = build_rapid_tag_observation_payload(
            df,
            mode=RAPID_TAG_REVIEW_MODE_APPLICABLE,
            include_other=False,
            tag_definitions=self.tags,
        )
        self.assertEqual(payload["distribution"], [])

    def test_fingerprint_does_not_change_for_second_opinion_rationale_only(self) -> None:
        df = pd.DataFrame([self.row(**{"Rapid Review Status": "Completed", "Rapid Review Sentiment Rationale": "old"})])
        changed = df.copy()
        changed.loc[0, "Rapid Review Sentiment Rationale"] = "new"
        payload = build_rapid_sentiment_observation_payload(
            df,
            scale=RAPID_SENTIMENT_SCALE_3_WAY,
            include_not_relevant=True,
        )
        changed_payload = build_rapid_sentiment_observation_payload(
            changed,
            scale=RAPID_SENTIMENT_SCALE_3_WAY,
            include_not_relevant=True,
        )
        first = rapid_observation_fingerprint(payload, settings={"family": "sentiment"}, analysis_context="context")
        second = rapid_observation_fingerprint(changed_payload, settings={"family": "sentiment"}, analysis_context="context")
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
