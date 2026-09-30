from __future__ import annotations

import unittest

from processing.analysis_context import (
    get_analysis_context_payload,
    normalize_analysis_context_terms,
    save_analysis_context,
)


class SessionState(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value


class AnalysisContextNormalizationTests(unittest.TestCase):
    def test_one_value_entry_is_preserved(self) -> None:
        self.assertEqual(normalize_analysis_context_terms(["College Board"]), ["College Board"])

    def test_comma_separated_paste_splits_values(self) -> None:
        self.assertEqual(
            normalize_analysis_context_terms(["Jane Smith, John Doe, Priya Patel"]),
            ["Jane Smith", "John Doe", "Priya Patel"],
        )

    def test_newline_separated_paste_splits_values(self) -> None:
        self.assertEqual(
            normalize_analysis_context_terms(["Jane Smith\nJohn Doe\nPriya Patel"]),
            ["Jane Smith", "John Doe", "Priya Patel"],
        )

    def test_mixed_comma_newline_input_splits_values(self) -> None:
        self.assertEqual(
            normalize_analysis_context_terms(["Jane Smith, John Doe\nPriya Patel"]),
            ["Jane Smith", "John Doe", "Priya Patel"],
        )

    def test_trims_blanks_and_dedupes_case_insensitively(self) -> None:
        self.assertEqual(
            normalize_analysis_context_terms([" Jane Smith ,, jane smith\nJOHN DOE ", "", "John Doe"]),
            ["Jane Smith", "JOHN DOE"],
        )

    def test_existing_list_values_are_preserved_and_expanded_when_needed(self) -> None:
        self.assertEqual(
            normalize_analysis_context_terms(["SAT", "AP, PSAT", "BigFuture"]),
            ["SAT", "AP", "PSAT", "BigFuture"],
        )

    def test_save_context_stores_split_multivalue_fields(self) -> None:
        session = SessionState(client_name="College Board")
        save_analysis_context(
            session,
            client_name="College Board",
            primary_name="College Board",
            alternate_names=["The College Board, CB"],
            spokespeople=["Jane Smith\nJohn Doe"],
            products=["SAT, AP", "BigFuture"],
            highlight_keywords=["National Recognition Program, Scholarships"],
            general_guidance="",
            sentiment_guidance="",
            qualitative_excluded_flags=[],
            dataset_excluded_flags=[],
            exclude_aggregators_from_outlet_insights=True,
        )

        self.assertEqual(session.analysis_alternate_names, ["The College Board", "CB"])
        self.assertEqual(session.analysis_spokespeople, ["Jane Smith", "John Doe"])
        self.assertEqual(session.analysis_products, ["SAT", "AP", "BigFuture"])
        self.assertEqual(session.analysis_highlight_keywords, ["National Recognition Program", "Scholarships"])

    def test_payload_returns_individual_normalized_values_for_downstream_consumers(self) -> None:
        session = SessionState(
            client_name="College Board",
            analysis_primary_names=["College Board"],
            analysis_alternate_names=["The College Board, CB"],
            analysis_spokespeople=["Jane Smith\nJohn Doe"],
            analysis_products=["SAT, AP"],
            analysis_highlight_keywords=["National Recognition Program, SAT"],
        )
        payload = get_analysis_context_payload(session)

        self.assertEqual(payload["alternate_names"], ["The College Board", "CB"])
        self.assertEqual(payload["spokespeople"], ["Jane Smith", "John Doe"])
        self.assertEqual(payload["products"], ["SAT", "AP"])
        self.assertEqual(payload["highlight_keywords"], ["National Recognition Program", "SAT"])


if __name__ == "__main__":
    unittest.main()
