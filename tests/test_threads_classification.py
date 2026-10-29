from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd

from processing.sentiment_config import get_sentiment_source_rows
from processing.standard_cleaning import (
    CANONICAL_MEDIA_TYPES,
    SOCIAL_TYPES,
    is_threads_url,
    normalize_media_type_value,
    run_standard_cleaning,
)
from processing.tagging_config import get_tagging_source_rows
from utils.io import build_upload_quality_report, normalize_uploaded_dataframe


THREADS_CSV = Path(__file__).parent / "fixtures" / "agility" / "agility_threads_cases.csv"

# Expected (Type, Original Type) after Basic Cleaning, keyed by the resolved URL.
# Original Type keeps the source label (normalized at upload) for provenance.
EXPECTED_SOCIAL = {
    "https://www.threads.com/@northfieldco/post/DQa1Threads01": ("THREADS", "THREADS"),
    "https://www.threads.net/@harbourfin/post/DQa2Threads02": ("THREADS", "THREADS"),
    "https://www.threads.com/share/DQa3ShareLink03": ("THREADS", "THREADS"),
    "threads.net/@lakeshoreventures/post/DQa4Threads04": ("THREADS", "THREADS"),
    "https://www.threads.com/@prairienews/post/DQa5Threads05": ("THREADS", "ONLINE NEWS"),
    "https://www.threads.net/@coastalfounders/post/DQa6Threads06": ("THREADS", "INSTAGRAM"),
    "https://threads.com/@northernbuild/post/DQa7Threads07": ("THREADS", ""),
    "": ("THREADS", "THREADS"),
    "https://bsky.app/profile/example.bsky.social/post/3mthreads09": ("BLUESKY", "THREADS"),
    "https://www.threads.com/@northfieldco/post/DQa13Dupe": ("THREADS", "THREADS"),
    "https://www.threads.com/@sparsefields/post/DQa15Sparse": ("THREADS", "THREADS"),
    "https://www.threads.com/@redirected/post/DQa16Original": ("THREADS", "ONLINE NEWS"),
    "https://x.com/controlaccount/status/1717000000000000017": ("X", "X"),
}
EXPECTED_TRADITIONAL_URLS = {
    "https://forums.example-tech.com/threads/threads-usage-overtakes-x.2476275/",
    "https://l.threads.com/?u=https%3A%2F%2Fwww.example-news.com%2Fstory-eleven",
    "https://www.example-news.com/story-twelve?utm_source=threads.net",
    "https://www.example-news.com/story-eighteen",
}
EXPECTED_DUPE_URLS = ["https://www.threads.com/@northfieldco/post/DQa13Dupe"]


def _load_threads_raw() -> pd.DataFrame:
    return pd.read_csv(THREADS_CSV)


def _clean(normalized: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return run_standard_cleaning(df=normalized, merge_online=True, drop_dupes=True, add_flags=False)


def _urls(df: pd.DataFrame) -> list[str]:
    return df["URL"].fillna("").astype(str).tolist()


class ThreadsUrlDetectionTests(unittest.TestCase):
    def test_threads_hostnames_and_paths_are_detected(self) -> None:
        for url in [
            "https://www.threads.com/@brand/post/DQabc123",
            "https://www.threads.net/@brand/post/DQabc123",
            "https://threads.com/@brand/post/DQabc123?xmt=AQG",
            "http://threads.net/@brand/post/DQabc123",
            "https://www.threads.com/share/DQabc123",
            "https://www.threads.net/t/CuW6-7KyXme",
            "https://WWW.THREADS.COM/@Brand/post/DQabc123",
            "threads.net/@brand/post/DQabc123",
            "www.threads.com/@brand",
        ]:
            with self.subTest(url=url):
                self.assertTrue(is_threads_url(url))

    def test_non_threads_urls_are_not_detected(self) -> None:
        for url in [
            "",
            None,
            float("nan"),
            "https://forums.macrumors.com/threads/threads-usage-overtakes-x-on-mobile.2476275/",
            "https://www.example-news.com/story?utm_source=threads.net",
            "https://l.threads.com/?u=https%3A%2F%2Fexample.com",
            "https://threads.com.example.org/@brand/post/1",
            "https://notthreads.net/@brand/post/1",
            "https://www.instagram.com/p/abc123/",
            "not a url",
        ]:
            with self.subTest(url=url):
                self.assertFalse(is_threads_url(url))


class ThreadsMediaTypeTests(unittest.TestCase):
    def test_threads_is_a_canonical_social_type(self) -> None:
        self.assertIn("THREADS", SOCIAL_TYPES)
        self.assertIn("THREADS", CANONICAL_MEDIA_TYPES)
        for value in ["THREADS", "Threads", " threads "]:
            with self.subTest(value=value):
                self.assertEqual(normalize_media_type_value(value), "THREADS")
                self.assertEqual(normalize_media_type_value(value, merge_online=False), "THREADS")

    def test_threads_upload_is_not_flagged_as_unrecognized(self) -> None:
        raw = _load_threads_raw()
        report = build_upload_quality_report(raw, normalize_uploaded_dataframe(raw))

        self.assertEqual(report["unrecognized_media_type_values"], [])
        self.assertNotIn("Some media types are unrecognized", [w["title"] for w in report["warnings"]])
        # TH-007 has a blank source type; upload still flags it even though cleaning recovers it from the URL.
        self.assertEqual(report["media_type_issue_row_numbers"], [8])

    def test_other_platform_normalization_is_unchanged(self) -> None:
        for value, expected in [
            ("TWITTER", "X"),
            ("X", "X"),
            ("BLUESKY", "BLUESKY"),
            ("INSTAGRAM", "INSTAGRAM"),
            ("ONLINE_NEWS", "ONLINE"),
            ("PRESS_RELEASE", "ONLINE"),
        ]:
            with self.subTest(value=value):
                self.assertEqual(normalize_media_type_value(value), expected)


class ThreadsSocialSplitTests(unittest.TestCase):
    def _assert_threads_fixture_routing(self, raw: pd.DataFrame) -> None:
        normalized = normalize_uploaded_dataframe(raw)
        results = _clean(normalized)
        df_traditional = results["df_traditional"]
        df_social = results["df_social"]
        df_dupes = results["df_dupes"]

        self.assertEqual(len(normalized), len(raw))
        self.assertEqual(len(df_traditional) + len(df_social) + len(df_dupes), len(raw))
        self.assertEqual((len(df_traditional), len(df_social), len(df_dupes)), (4, 13, 1))

        self.assertNotIn("THREADS", set(df_traditional["Type"]))
        self.assertEqual(set(_urls(df_traditional)), EXPECTED_TRADITIONAL_URLS)
        self.assertEqual(_urls(df_dupes), EXPECTED_DUPE_URLS)
        self.assertEqual(df_dupes["Type"].tolist(), ["THREADS"])

        social_urls = _urls(df_social)
        self.assertEqual(len(social_urls), len(set(social_urls)))
        self.assertEqual(set(social_urls), set(EXPECTED_SOCIAL))
        for _, row in df_social.iterrows():
            url = "" if pd.isna(row["URL"]) else str(row["URL"])
            with self.subTest(url=url):
                original_type = "" if pd.isna(row["Original Type"]) else str(row["Original Type"])
                self.assertEqual((row["Type"], original_type), EXPECTED_SOCIAL[url])

    def test_social_listening_export_with_media_type_column(self) -> None:
        raw = _load_threads_raw()
        self.assertIn("Media Type", raw.columns)
        self._assert_threads_fixture_routing(raw)

    def test_traditional_monitoring_export_with_type_column(self) -> None:
        raw = _load_threads_raw().rename(columns={"Media Type": "Type"})
        self._assert_threads_fixture_routing(raw)

    def test_export_with_both_type_columns_prefers_media_type(self) -> None:
        raw = _load_threads_raw()
        raw.insert(raw.columns.get_loc("Media Type"), "Type", "POST")

        normalized = normalize_uploaded_dataframe(raw)

        self.assertEqual(list(normalized.columns).count("Type"), 1)
        self.assertEqual(set(normalized["Source Type"]), {"POST"})
        self._assert_threads_fixture_routing(raw)

    def test_threads_type_without_url_column_routes_to_social(self) -> None:
        raw = _load_threads_raw().drop(columns=["URL", "Original URL"])
        raw = raw.loc[raw["Media Type"].fillna("").str.upper().eq("THREADS")].reset_index(drop=True)

        results = _clean(normalize_uploaded_dataframe(raw))

        self.assertTrue(results["df_traditional"].empty)
        self.assertEqual(set(results["df_social"]["Type"]), {"THREADS"})
        self.assertEqual(len(results["df_social"]) + len(results["df_dupes"]), len(raw))

    def test_threads_rows_are_excluded_from_sentiment_and_tagging_sources(self) -> None:
        df = pd.DataFrame({"Type": ["ONLINE", "THREADS", "X"], "Headline": ["a", "b", "c"]})

        self.assertEqual(get_sentiment_source_rows(df)["Type"].tolist(), ["ONLINE"])
        self.assertEqual(get_tagging_source_rows(df)["Type"].tolist(), ["ONLINE"])


if __name__ == "__main__":
    unittest.main()
