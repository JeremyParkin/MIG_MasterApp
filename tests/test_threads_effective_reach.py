from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from processing.effective_reach import (
    DEFAULT_PLATFORM_PARAMS,
    apply_effective_reach_social,
    compute_social_single,
)
from processing.standard_cleaning import run_standard_cleaning
from utils.io import normalize_uploaded_dataframe


THREADS_CSV = Path(__file__).parent / "fixtures" / "agility" / "agility_threads_cases.csv"


class ThreadsEffectiveReachTests(unittest.TestCase):
    def test_threads_reach_from_followers_matches_documented_curve(self) -> None:
        for followers, expected in [(936, 25), (10_000, 270), (100_000, 2_338), (1_000_000, 14_274)]:
            with self.subTest(followers=followers):
                self.assertEqual(compute_social_single("THREADS", followers, 0, DEFAULT_PLATFORM_PARAMS), expected)

    def test_threads_engagement_does_not_move_reach(self) -> None:
        df = pd.DataFrame(
            {
                "Type": ["THREADS"] * 4,
                "Impressions": [10_000] * 4,
                "Engagements": [np.nan, 0, 5, 10_000],
            }
        )

        result = apply_effective_reach_social(df)

        self.assertEqual(result["Effective Reach"].tolist(), [270, 270, 270, 270])

    def test_threads_sits_between_x_and_bluesky_at_expected_engagement(self) -> None:
        for followers in [936, 10_000, 100_000, 1_000_000]:
            with self.subTest(followers=followers):
                threads = compute_social_single("THREADS", followers, 0, DEFAULT_PLATFORM_PARAMS)
                x = compute_social_single(
                    "X", followers, followers * DEFAULT_PLATFORM_PARAMS["x"]["EXPECTED_ENG_RATE"], DEFAULT_PLATFORM_PARAMS
                )
                bluesky = compute_social_single(
                    "BLUESKY",
                    followers,
                    followers * DEFAULT_PLATFORM_PARAMS["bluesky"]["EXPECTED_ENG_RATE"],
                    DEFAULT_PLATFORM_PARAMS,
                )
                self.assertGreater(threads, x)
                self.assertLess(threads, bluesky)

    def test_other_platforms_keep_existing_blank_engagement_behavior(self) -> None:
        # Blank engagement still scores at PERF_FLOOR outside Threads; changing that is a separate decision.
        df = pd.DataFrame(
            {
                "Type": ["X", "INSTAGRAM"],
                "Impressions": [10_000, 133_000],
                "Engagements": [np.nan, np.nan],
            }
        )

        self.assertEqual(apply_effective_reach_social(df)["Effective Reach"].tolist(), [122, 8_187])

    def test_threads_fixture_rows_receive_effective_reach(self) -> None:
        raw = pd.read_csv(THREADS_CSV)
        cleaned = run_standard_cleaning(
            df=normalize_uploaded_dataframe(raw), merge_online=True, drop_dupes=True, add_flags=False
        )

        df_social = apply_effective_reach_social(cleaned["df_social"])
        threads = df_social.loc[df_social["Type"].eq("THREADS")]

        self.assertEqual(len(threads), 11)
        self.assertFalse(threads["Effective Reach"].isna().any())
        # TH-015 has blank followers, which upload normalization stores as 0.
        sparse = threads.loc[threads["URL"].eq("https://www.threads.com/@sparsefields/post/DQa15Sparse")]
        self.assertEqual(sparse["Effective Reach"].tolist(), [0])
        self.assertTrue((threads.loc[threads["Impressions"].gt(0), "Effective Reach"] > 0).all())


if __name__ == "__main__":
    unittest.main()
