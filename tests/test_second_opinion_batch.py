from __future__ import annotations

import unittest

from utils.second_opinion_batch import prepare_second_opinion_batch_size


class State(dict):
    pass


class SecondOpinionBatchInputTests(unittest.TestCase):
    def test_sentiment_initial_recommendation_seeds_the_input(self) -> None:
        state = State()

        selected = prepare_second_opinion_batch_size(
            state,
            input_key="spotcheck_auto_review_n",
            recommended_batch=19,
            available_count=60,
            refresh_recommendation=True,
        )

        self.assertEqual(selected, 19)
        self.assertEqual(state["spotcheck_auto_review_n"], 19)

    def test_tagging_initial_recommendation_seeds_the_input(self) -> None:
        state = State()

        selected = prepare_second_opinion_batch_size(
            state,
            input_key="tagging_pre_review_n",
            recommended_batch=32,
            available_count=80,
            refresh_recommendation=True,
        )

        self.assertEqual(selected, 32)
        self.assertEqual(state["tagging_pre_review_n"], 32)

    def test_manual_override_survives_the_same_recommendation_cycle(self) -> None:
        state = State(spotcheck_auto_review_n=7)

        selected = prepare_second_opinion_batch_size(
            state,
            input_key="spotcheck_auto_review_n",
            recommended_batch=19,
            available_count=60,
            refresh_recommendation=False,
        )

        self.assertEqual(selected, 7)

    def test_new_first_pass_population_reseeds_the_input(self) -> None:
        state = State(tagging_pre_review_n=7)

        selected = prepare_second_opinion_batch_size(
            state,
            input_key="tagging_pre_review_n",
            recommended_batch=32,
            available_count=80,
            refresh_recommendation=True,
        )

        self.assertEqual(selected, 32)
        self.assertEqual(state["tagging_pre_review_n"], 32)

    def test_selected_batch_is_bounded_by_remaining_available_stories(self) -> None:
        state = State(tagging_pre_review_n=32)

        selected = prepare_second_opinion_batch_size(
            state,
            input_key="tagging_pre_review_n",
            recommended_batch=19,
            available_count=12,
            refresh_recommendation=False,
        )

        self.assertEqual(selected, 12)


if __name__ == "__main__":
    unittest.main()
