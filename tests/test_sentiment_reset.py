from __future__ import annotations

from pathlib import Path
import unittest

import pandas as pd

from processing.ai_sentiment import (
    reset_ai_sentiment_results,
    reset_sentiment_processing_state,
)
from processing.sentiment_config import reset_sentiment_config_state


class State(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


class SentimentProcessingResetTests(unittest.TestCase):
    def _processed_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "Group ID": [1],
                "Headline": ["A retained story identifier"],
                "AI Sentiment": ["NEGATIVE"],
                "AI Sentiment Confidence": [88],
                "Review AI Sentiment": ["NEUTRAL"],
                "Review AI Confidence": [81],
                "AI Agreement": ["Disagree"],
                "Needs Human Review": ["Yes"],
                "Assigned Sentiment": ["POSITIVE"],
                "Assigned Sentiment Source": ["HUMAN"],
                "Hybrid Sentiment": ["NEUTRAL"],
            }
        )

    def test_result_reset_preserves_story_rows_but_clears_workflow_results(self) -> None:
        unique, grouped = reset_ai_sentiment_results(self._processed_frame(), self._processed_frame())

        self.assertEqual(unique.loc[0, "Group ID"], 1)
        self.assertEqual(unique.loc[0, "Headline"], "A retained story identifier")
        for frame in (unique, grouped):
            for column in [
                "AI Sentiment",
                "AI Sentiment Confidence",
                "Review AI Sentiment",
                "Review AI Confidence",
                "AI Agreement",
                "Needs Human Review",
                "Assigned Sentiment",
                "Assigned Sentiment Source",
                "Hybrid Sentiment",
            ]:
                self.assertTrue(pd.isna(frame.loc[0, column]), column)

    def test_processing_reset_preserves_configuration_and_prepared_data(self) -> None:
        prepared_rows = self._processed_frame()
        prepared_unique = self._processed_frame()
        state = State(
            sentiment_config_step=True,
            sentiment_type="4-way",
            model_choice="gpt-5.6-luna",
            pre_prompt="configured prompt",
            sentiment_instruction="configured instructions",
            post_prompt="configured clarifications",
            functions=[{"name": "sentiment"}],
            df_sentiment_rows=prepared_rows,
            df_sentiment_unique=prepared_unique,
            sentiment_observation_output={"overall": "stale"},
            initial_ai_label={1: "NEGATIVE"},
            spot_checked_groups={1},
            accepted_initial={1},
            spot_idx=4,
            spot_lock_gid=1,
            spot_ai_loading=True,
            spot_ai_refresh_requested=True,
            spot_ai_model_override="old-model",
            sentiment_second_opinion_target_batch=19,
            sentiment_second_opinion_target_source_count=42,
            spotcheck_auto_review_n=19,
            spotcheck_pre_review_message="stale",
            spotcheck_auto_resolve_message="stale",
            spotcheck_review_mode="Needs review",
            spotcheck_selected_bucket="NEGATIVE",
            __last_sentiment_batch_summary__={"done": 42},
        )

        reset_sentiment_processing_state(state)

        self.assertEqual(state["sentiment_type"], "4-way")
        self.assertEqual(state["model_choice"], "gpt-5.6-luna")
        self.assertEqual(state["pre_prompt"], "configured prompt")
        self.assertEqual(state["sentiment_instruction"], "configured instructions")
        self.assertEqual(state["post_prompt"], "configured clarifications")
        self.assertTrue(state["sentiment_config_step"])
        self.assertIs(state["df_sentiment_rows"], prepared_rows)
        self.assertIs(state["df_sentiment_unique"], prepared_unique)
        self.assertEqual(state["sentiment_observation_output"], {})
        self.assertEqual(state["initial_ai_label"], {})
        self.assertEqual(state["spot_checked_groups"], set())
        self.assertEqual(state["accepted_initial"], set())
        self.assertEqual(state["spot_idx"], 0)
        self.assertIsNone(state["spot_lock_gid"])
        self.assertFalse(state["spot_ai_loading"])
        self.assertFalse(state["spot_ai_refresh_requested"])
        for key in [
            "sentiment_second_opinion_target_batch",
            "sentiment_second_opinion_target_source_count",
            "spotcheck_auto_review_n",
            "spotcheck_pre_review_message",
            "spotcheck_auto_resolve_message",
            "spotcheck_review_mode",
            "spotcheck_selected_bucket",
            "__last_sentiment_batch_summary__",
        ]:
            self.assertNotIn(key, state)

    def test_full_configuration_reset_remains_available_to_setup(self) -> None:
        state = State(
            sentiment_config_step=True,
            sentiment_sample_mode="custom",
            sentiment_sample_size=99,
            sentiment_full_override=True,
            df_sentiment_rows=self._processed_frame(),
            df_sentiment_grouped_rows=self._processed_frame(),
            df_sentiment_unique=self._processed_frame(),
            sentiment_elapsed_time=7.0,
            client_name="Client",
            sentiment_type="4-way",
            model_choice="gpt-5.6-luna",
            pre_prompt="configured prompt",
            sentiment_second_opinion_target_batch=19,
            spotcheck_auto_review_n=19,
        )

        reset_sentiment_config_state(state)

        self.assertFalse(state.sentiment_config_step)
        self.assertEqual(state.sentiment_sample_mode, "representative")
        self.assertTrue(state.df_sentiment_rows.empty)
        self.assertTrue(state.df_sentiment_grouped_rows.empty)
        self.assertTrue(state.df_sentiment_unique.empty)
        self.assertEqual(state.ui_sentiment_type, "3-way")
        for key in ["sentiment_type", "model_choice", "pre_prompt", "spotcheck_auto_review_n"]:
            self.assertNotIn(key, state)

    def test_run_step_only_exposes_processing_reset(self) -> None:
        page_source = (
            Path(__file__).resolve().parents[1] / "pages" / "Sentiment.py"
        ).read_text(encoding="utf-8")

        self.assertIn('st.button("Reset Processed Rows")', page_source)
        self.assertNotIn("Reset Sentiment Dataset", page_source)
        self.assertIn('st.button("Prepare Sentiment Dataset", type="primary")', page_source)


if __name__ == "__main__":
    unittest.main()
