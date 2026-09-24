from __future__ import annotations

import io
import unittest

import dill
import pandas as pd

from processing.ai_tagging import (
    RESERVED_OTHER_DEFINITION,
    apply_tagging_result_to_unique_df,
    apply_tag_review_flags_to_group,
    auto_assign_resolved_tag_matches,
    ensure_canonical_tag_definitions,
    normalize_tag_assignment,
    parse_tag_definitions,
    set_assigned_tag,
)
from utils.ai_checkpoints import acknowledge_checkpoint, record_checkpoint_progress, reset_workflow_checkpoints
from utils.session_snapshot import build_serializable_session_payload, load_session_state_from_file


class State(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


class ProtectedOtherTests(unittest.TestCase):
    def setUp(self) -> None:
        self.definitions = {"Innovation": "New technology", "Other": RESERVED_OTHER_DEFINITION}

    def test_other_is_added_and_cannot_be_user_defined(self) -> None:
        parsed = parse_tag_definitions("Innovation: New technology")
        self.assertEqual(parsed["Other"], RESERVED_OTHER_DEFINITION)
        with self.assertRaises(ValueError):
            parse_tag_definitions("other: Custom fallback")

    def test_assignment_normalization_uses_other_as_exclusive_fallback(self) -> None:
        self.assertEqual(normalize_tag_assignment([], self.definitions), ["Other"])
        self.assertEqual(normalize_tag_assignment(["Unknown"], self.definitions), ["Other"])
        self.assertEqual(
            normalize_tag_assignment(["Other", "Innovation"], self.definitions),
            ["Innovation"],
        )

    def test_ai_and_human_empty_assignments_become_other(self) -> None:
        unique = pd.DataFrame({"Group ID": [1]})
        tagged = apply_tagging_result_to_unique_df(
            unique,
            original_index=0,
            result={"tags": [], "confidence": 80, "explanations": []},
            tagging_mode="Multiple applicable tags",
            tag_definitions=self.definitions,
        )
        self.assertEqual(tagged.loc[0, "AI Tag"], "Other")

        assigned, _ = set_assigned_tag(
            tagged,
            tagged.copy(),
            1,
            "",
            tag_definitions=self.definitions,
        )
        self.assertEqual(assigned.loc[0, "Assigned Tag"], "Other")

    def test_matching_other_opinions_can_auto_resolve(self) -> None:
        unique = pd.DataFrame(
            {
                "Group ID": [1],
                "AI Tag": ["Other"],
                "Review AI Tag": ["Other"],
                "Review AI Confidence": [95],
                "Assigned Tag": [pd.NA],
            }
        )
        flagged, grouped = apply_tag_review_flags_to_group(
            unique,
            unique.copy(),
            1,
            ai_label="Other",
            review_label="Other",
            review_confidence=95,
            tagging_mode="Multiple applicable tags",
        )
        resolved, _, accepted, _ = auto_assign_resolved_tag_matches(
            flagged,
            grouped,
            tagging_mode="Multiple applicable tags",
        )
        self.assertEqual(accepted, 1)
        self.assertEqual(resolved.loc[0, "Assigned Tag"], "Other")

    def test_legacy_custom_other_definition_is_replaced(self) -> None:
        definitions = ensure_canonical_tag_definitions({"Innovation": "New", "OTHER": "Custom"})
        self.assertEqual(definitions, {"Innovation": "New", "Other": RESERVED_OTHER_DEFINITION})


class CheckpointAndSnapshotTests(unittest.TestCase):
    def test_checkpoint_progress_acknowledgement_and_reset(self) -> None:
        state = State()
        record_checkpoint_progress(state, workflow="tagging", stage="first", successful_results=1200, interval=1000)
        self.assertEqual(state["ai_checkpoint_tagging_first_completed"], 1200)
        self.assertEqual(state["ai_checkpoint_tagging_first_next"], 1000)
        acknowledge_checkpoint(state, workflow="tagging", stage="first", interval=1000)
        self.assertEqual(state["ai_checkpoint_tagging_first_next"], 2000)
        reset_workflow_checkpoints(state, "tagging")
        self.assertFalse(any(key.startswith("ai_checkpoint_tagging_") for key in state))

    def test_snapshot_round_trip_repairs_legacy_other_definition(self) -> None:
        state = State(
            client_name="Example",
            tag_definitions={"Innovation": "New", "Other": "Legacy custom text"},
            tags_text="Innovation: New\nOther: Legacy custom text",
            df_tagging_unique=pd.DataFrame({"Group ID": [1], "AI Tag": ["Other"]}),
        )
        payload, skipped = build_serializable_session_payload(state)
        self.assertEqual(skipped, [])

        restored = State()
        load_session_state_from_file(restored, io.BytesIO(dill.dumps(payload)))
        self.assertEqual(restored.tag_definitions["Other"], RESERVED_OTHER_DEFINITION)
        self.assertEqual(restored.tags_text, "Innovation: New")
        self.assertEqual(restored.df_tagging_unique.loc[0, "AI Tag"], "Other")


if __name__ == "__main__":
    unittest.main()
