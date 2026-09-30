from __future__ import annotations

import io
import unittest
import zipfile

import pandas as pd

from processing.download_exports import build_report_copy_docx_bytes


def _docx_xml(docx_bytes: bytes) -> str:
    with zipfile.ZipFile(io.BytesIO(docx_bytes)) as zf:
        return zf.read("word/document.xml").decode("utf-8")


def _sentiment_output(prefix: str = "") -> dict:
    return {
        "overall_observation": f"{prefix}sentiment overall",
        "sentiment_sections": [
            {"sentiment": "POSITIVE", "observation": f"{prefix}positive observation"},
        ],
        "_examples_by_sentiment": {
            "POSITIVE": [
                {
                    "headline": f"{prefix}positive example",
                    "url": "https://example.com/positive",
                    "outlet": "Example Outlet",
                    "example_type": "Online",
                    "mentions": 2,
                    "impressions": 100,
                    "effective_reach": 50,
                }
            ]
        },
    }


def _tag_output(prefix: str = "") -> dict:
    return {
        "overall_observation": f"{prefix}tag overall",
        "tag_sections": [
            {"tag": "AP", "observation": f"{prefix}AP observation"},
        ],
        "_examples_by_tag": {
            "AP": [
                {
                    "headline": f"{prefix}AP example",
                    "url": "https://example.com/ap",
                    "outlet": "Example Outlet",
                    "example_type": "Online",
                    "mentions": 3,
                    "impressions": 200,
                    "effective_reach": 75,
                }
            ]
        },
    }


def _session(**overrides) -> dict:
    base = {
        "client_name": "Client",
        "df_traditional": pd.DataFrame(
            columns=[
                "Group ID",
                "Headline",
                "Author",
                "Outlet",
                "Type",
                "Coverage Flags",
                "Mentions",
                "Impressions",
                "Effective Reach",
                "Prime Example",
            ]
        ),
    }
    base.update(overrides)
    return base


class RapidReportCopyObservationTests(unittest.TestCase):
    def test_rapid_sentiment_observation_only_is_included(self) -> None:
        xml = _docx_xml(
            build_report_copy_docx_bytes(
                _session(rapid_sentiment_observation_output=_sentiment_output("rapid "))
            )
        )
        self.assertIn("Rapid Sentiment Insights", xml)
        self.assertIn("rapid sentiment overall", xml)
        self.assertIn("rapid positive observation", xml)
        self.assertIn("rapid positive example", xml)
        self.assertNotIn("<w:t>Sentiment Insights</w:t>", xml)

    def test_rapid_tag_observation_only_is_included(self) -> None:
        xml = _docx_xml(
            build_report_copy_docx_bytes(
                _session(rapid_tag_observation_output=_tag_output("rapid "))
            )
        )
        self.assertIn("Rapid Tag Insights", xml)
        self.assertIn("rapid tag overall", xml)
        self.assertIn("rapid AP observation", xml)
        self.assertIn("rapid AP example", xml)

    def test_both_rapid_observation_families_are_included(self) -> None:
        xml = _docx_xml(
            build_report_copy_docx_bytes(
                _session(
                    rapid_sentiment_observation_output=_sentiment_output("rapid "),
                    rapid_tag_observation_output=_tag_output("rapid "),
                )
            )
        )
        self.assertIn("Rapid Sentiment Insights", xml)
        self.assertIn("Rapid Tag Insights", xml)

    def test_established_observations_are_still_included(self) -> None:
        xml = _docx_xml(
            build_report_copy_docx_bytes(
                _session(
                    sentiment_observation_output=_sentiment_output("established "),
                    tagging_observation_output=_tag_output("established "),
                )
            )
        )
        self.assertIn("Sentiment Insights", xml)
        self.assertIn("Tag Insights", xml)
        self.assertIn("established sentiment overall", xml)
        self.assertIn("established tag overall", xml)
        self.assertNotIn("Rapid Sentiment Insights", xml)
        self.assertNotIn("Rapid Tag Insights", xml)

    def test_established_and_rapid_outputs_coexist_with_distinct_headings(self) -> None:
        xml = _docx_xml(
            build_report_copy_docx_bytes(
                _session(
                    sentiment_observation_output=_sentiment_output("established "),
                    rapid_sentiment_observation_output=_sentiment_output("rapid "),
                    tagging_observation_output=_tag_output("established "),
                    rapid_tag_observation_output=_tag_output("rapid "),
                )
            )
        )
        self.assertIn("Sentiment Insights", xml)
        self.assertIn("Rapid Sentiment Insights", xml)
        self.assertIn("Tag Insights", xml)
        self.assertIn("Rapid Tag Insights", xml)
        self.assertIn("established positive example", xml)
        self.assertIn("rapid positive example", xml)
        self.assertIn("established AP example", xml)
        self.assertIn("rapid AP example", xml)

    def test_empty_rapid_outputs_do_not_create_sections(self) -> None:
        with self.assertRaises(ValueError):
            build_report_copy_docx_bytes(
                _session(
                    rapid_sentiment_observation_output={},
                    rapid_tag_observation_output={"_error": "boom"},
                )
            )


if __name__ == "__main__":
    unittest.main()
