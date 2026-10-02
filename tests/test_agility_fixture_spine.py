from __future__ import annotations

import io
import re
import unittest
from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal

from processing.coverage_flags import (
    add_coverage_flags,
    apply_story_family_press_release_flags,
)
from processing.download_exports import build_clean_workbook_bytes
from processing.effective_reach import (
    apply_effective_reach_social,
    apply_effective_reach_traditional,
)
from processing.standard_cleaning import normalize_media_type_value, run_standard_cleaning
from processing.story_grouping import (
    NORMAL_TEXT_SIMILARITY_THRESHOLD,
    build_unique_story_table,
    cluster_by_media_type,
    mark_prime_examples,
)
from utils.io import (
    build_upload_quality_report,
    detect_original_ave_col,
    normalize_uploaded_dataframe,
)


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "agility"
GOLDEN_CSV = FIXTURE_DIR / "agility_golden_corpus.csv"
MALFORMED_CSV = FIXTURE_DIR / "agility_malformed_inputs.csv"
CLEANED_WORKBOOK = FIXTURE_DIR / "agility_golden_cleaned_workbook.xlsx"
MULTISHEET_WORKBOOK = FIXTURE_DIR / "agility_golden_multisheet_upload.xlsx"
MANIFEST = FIXTURE_DIR / "MANIFEST.md"


def _load_manifest() -> str:
    return MANIFEST.read_text(encoding="utf-8")


def _documented_raw_media_types() -> set[str]:
    manifest = _load_manifest()
    match = re.search(
        r"Includes all source media channels present in the supplied export: (.+?)\.",
        manifest,
    )
    if not match:
        raise AssertionError("Manifest does not document source media channels.")
    return {
        re.sub(r"^and\s+", "", value.strip())
        for value in match.group(1).split(",")
    }


def _documented_unique_story_count() -> int:
    manifest = _load_manifest()
    match = re.search(r"unique story rows: (\d+)", manifest)
    if not match:
        raise AssertionError("Manifest does not document unique story row count.")
    return int(match.group(1))


def _load_golden_raw() -> pd.DataFrame:
    return pd.read_csv(GOLDEN_CSV)


def _load_malformed_raw() -> pd.DataFrame:
    return pd.read_csv(MALFORMED_CSV)


def _run_basic_cleaning_like_page(raw_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    normalized = normalize_uploaded_dataframe(raw_df)
    cleaning_results = run_standard_cleaning(
        df=normalized,
        merge_online=True,
        drop_dupes=True,
        add_flags=False,
    )

    df_traditional = apply_effective_reach_traditional(cleaning_results["df_traditional"])
    df_social = apply_effective_reach_social(cleaning_results["df_social"])
    df_traditional = add_coverage_flags(df_traditional)

    df_grouped = cluster_by_media_type(
        df=df_traditional,
        similarity_threshold=NORMAL_TEXT_SIMILARITY_THRESHOLD,
        max_batch_size=1800,
    )
    df_grouped = apply_story_family_press_release_flags(df_grouped)
    df_grouped = mark_prime_examples(df_grouped)
    df_unique = build_unique_story_table(df_grouped)

    return {
        "normalized": normalized,
        "df_traditional": df_grouped,
        "df_social": df_social,
        "df_dupes": cleaning_results["df_dupes"],
        "df_ai_unique": df_unique,
    }


class _SessionState(dict):
    def __getattr__(self, name: str):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value):
        self[name] = value


def _export_state(raw_df: pd.DataFrame, cleaning_state: dict[str, pd.DataFrame]) -> _SessionState:
    return _SessionState(
        {
            "df_untouched": raw_df,
            "df_traditional": cleaning_state["df_traditional"],
            "df_social": cleaning_state["df_social"],
            "df_dupes": cleaning_state["df_dupes"],
            "df_ai_grouped": cleaning_state["df_traditional"],
            "df_ai_unique": cleaning_state["df_ai_unique"],
            "standard_step": True,
            "original_ave_col": detect_original_ave_col(raw_df),
            "include_labeling_audit_columns": False,
            "outlet_rollup_map": {},
        }
    )


def _nonblank_duplicate_urls(raw_df: pd.DataFrame) -> list[str]:
    urls = raw_df["URL"].fillna("").astype(str).str.strip()
    counts = urls[urls.ne("")].value_counts()
    return sorted(counts[counts > 1].index.tolist())


def _row_group_ids(df: pd.DataFrame, mask: pd.Series) -> set:
    return set(df.loc[mask, "Group ID"].tolist())


class AgilityFixtureSpineTests(unittest.TestCase):
    def test_golden_csv_upload_normalization_contract(self) -> None:
        raw = _load_golden_raw()
        normalized = normalize_uploaded_dataframe(raw)
        report = build_upload_quality_report(raw, normalized)

        documented_raw_types = _documented_raw_media_types()
        documented_normalized_types = {
            normalize_media_type_value(media_type, merge_online=False)
            for media_type in documented_raw_types
        }

        self.assertEqual(raw.shape, (100, 48))
        self.assertEqual(set(raw["Media Type"].dropna().astype(str)), documented_raw_types)
        self.assertEqual(detect_original_ave_col(raw), "AVE(USD)")
        self.assertEqual([warning["title"] for warning in report["warnings"]], [])
        self.assertEqual(len(normalized), len(raw))
        self.assertEqual(set(normalized["Type"].dropna().astype(str)), documented_normalized_types)

        for column in ["Date", "Type", "Snippet", "Prov/State", "AVE", "Mentions", "SyndicationId"]:
            self.assertIn(column, normalized.columns)
        for source_column in ["Published Date", "Published Time", "Media Type", "Coverage Snippet", "AVE(USD)"]:
            self.assertNotIn(source_column, normalized.columns)

    def test_golden_basic_cleaning_reconciliation(self) -> None:
        raw = _load_golden_raw()
        state = _run_basic_cleaning_like_page(raw)

        self.assertEqual(len(state["df_traditional"]), 49)
        self.assertEqual(len(state["df_social"]), 47)
        self.assertEqual(len(state["df_dupes"]), 4)
        self.assertEqual(len(state["df_ai_unique"]), _documented_unique_story_count())
        self.assertEqual(
            len(state["df_traditional"]) + len(state["df_social"]) + len(state["df_dupes"]),
            len(raw),
        )
        self.assertEqual(set(state["df_traditional"]["Type"]), {"ONLINE", "PRINT", "RADIO", "TV"})
        self.assertEqual(
            set(state["df_social"]["Type"]),
            {"BLUESKY", "FACEBOOK", "INSTAGRAM", "LINKEDIN", "REDDIT", "TIKTOK", "X", "YOUTUBE"},
        )

        for column in [
            "Group ID",
            "Prime Example",
            "Original Type",
            "Coverage Flags",
            "Effective Reach",
            "Grouping Source",
            "Grouping Warning",
        ]:
            self.assertIn(column, state["df_traditional"].columns)

    def test_exact_duplicate_cases_route_to_dupes(self) -> None:
        manifest = _load_manifest()
        for case_id in ["GC-020", "GC-021", "GC-039", "GC-040", "GC-041", "GC-042", "GC-084", "GC-085"]:
            self.assertIn(case_id, manifest)

        raw = _load_golden_raw()
        state = _run_basic_cleaning_like_page(raw)
        duplicate_urls = _nonblank_duplicate_urls(raw)

        self.assertEqual(len(duplicate_urls), 4)
        active_urls = pd.concat(
            [
                state["df_traditional"]["URL"],
                state["df_social"]["URL"],
            ],
            ignore_index=True,
        ).fillna("").astype(str).str.strip()
        dupe_urls = state["df_dupes"]["URL"].fillna("").astype(str).str.strip()

        for url in duplicate_urls:
            self.assertEqual(int(active_urls.eq(url).sum()), 1, url)
            self.assertEqual(int(dupe_urls.eq(url).sum()), 1, url)

    def test_grouping_semantic_membership_and_prime_invariants(self) -> None:
        manifest = _load_manifest()
        for case_id in ["GC-008", "GC-009", "GC-012", "GC-013", "GC-015", "GC-016", "GC-017", "GC-018"]:
            self.assertIn(case_id, manifest)

        grouped = _run_basic_cleaning_like_page(_load_golden_raw())["df_traditional"]

        defence_mask = grouped["SyndicationId"].eq("7PHLCnVZFrGsHh69bLwdHg==")
        self.assertEqual(int(defence_mask.sum()), 4)
        self.assertEqual(len(_row_group_ids(grouped, defence_mask)), 1)
        self.assertEqual(set(grouped.loc[defence_mask, "Grouping Source"]), {"SyndicationId"})

        business_for_sale_mask = grouped["SyndicationId"].eq("iu84YaqMh30WzUH40minOQ==")
        self.assertEqual(int(business_for_sale_mask.sum()), 2)
        self.assertEqual(len(_row_group_ids(grouped, business_for_sale_mask)), 1)

        black_entrepreneur_mask = grouped["Headline"].str.contains("Black Entrepreneurs", case=False, na=False)
        black_owned_mask = grouped["Headline"].str.contains("Black-Owned Businesses", case=False, na=False)
        self.assertEqual(int(black_entrepreneur_mask.sum()), 1)
        self.assertEqual(int(black_owned_mask.sum()), 1)
        self.assertNotEqual(
            next(iter(_row_group_ids(grouped, black_entrepreneur_mask))),
            next(iter(_row_group_ids(grouped, black_owned_mask))),
        )

        prime_counts = grouped.groupby("Group ID", dropna=False)["Prime Example"].sum()
        self.assertTrue((prime_counts == 1).all())
        self.assertEqual(len(prime_counts), _documented_unique_story_count())

    def test_clean_workbook_export_structure_and_semantic_content(self) -> None:
        raw = _load_golden_raw()
        state = _run_basic_cleaning_like_page(raw)
        workbook_bytes = build_clean_workbook_bytes(_export_state(raw, state))

        generated_workbook = pd.ExcelFile(io.BytesIO(workbook_bytes))
        committed_workbook = pd.ExcelFile(CLEANED_WORKBOOK)
        expected_sheets = [
            "CLEAN TRAD",
            "CLEAN SOCIAL",
            "Authors",
            "Outlets",
            "DLTD DUPES",
            "RAW",
            "EXPORT METADATA",
        ]

        self.assertEqual(generated_workbook.sheet_names, expected_sheets)
        self.assertEqual(committed_workbook.sheet_names, expected_sheets)

        generated_trad = pd.read_excel(generated_workbook, sheet_name="CLEAN TRAD")
        generated_social = pd.read_excel(generated_workbook, sheet_name="CLEAN SOCIAL")
        generated_dupes = pd.read_excel(generated_workbook, sheet_name="DLTD DUPES")
        generated_raw = pd.read_excel(generated_workbook, sheet_name="RAW")

        self.assertEqual(len(generated_trad), 49)
        self.assertEqual(len(generated_social), 47)
        self.assertEqual(len(generated_dupes), 4)
        self.assertEqual(len(generated_raw), 100)

        for column in ["Group ID", "Prime Example", "Original Type", "Coverage Flags", "Effective Reach"]:
            self.assertIn(column, generated_trad.columns)
        self.assertEqual(set(generated_social["Type"]), {"BLUESKY", "FACEBOOK", "INSTAGRAM", "LINKEDIN", "REDDIT", "TIKTOK", "X", "YOUTUBE"})
        self.assertEqual(set(generated_dupes["URL"].fillna("").astype(str).str.strip()), set(_nonblank_duplicate_urls(raw)))

    def test_cleaned_workbook_sheets_are_valid_headless_reentry_inputs(self) -> None:
        workbook = pd.ExcelFile(CLEANED_WORKBOOK)
        clean_trad = pd.read_excel(workbook, sheet_name="CLEAN TRAD")
        clean_social = pd.read_excel(workbook, sheet_name="CLEAN SOCIAL")

        normalized_trad = normalize_uploaded_dataframe(clean_trad)
        normalized_social = normalize_uploaded_dataframe(clean_social)

        self.assertEqual(len(normalized_trad), 49)
        self.assertEqual(len(normalized_social), 47)
        for column in ["Date", "Type", "Group ID", "Prime Example", "Original Type"]:
            self.assertIn(column, normalized_trad.columns)
        for column in ["Date", "Type", "Original Type"]:
            self.assertIn(column, normalized_social.columns)
        self.assertEqual(set(normalized_trad["Type"]), {"ONLINE", "PRINT", "RADIO", "TV"})
        self.assertEqual(set(normalized_social["Type"]), {"BLUESKY", "FACEBOOK", "INSTAGRAM", "LINKEDIN", "REDDIT", "TIKTOK", "X", "YOUTUBE"})

    def test_multisheet_workbook_discovery_and_intended_sheet(self) -> None:
        workbook = pd.ExcelFile(MULTISHEET_WORKBOOK)
        self.assertEqual(workbook.sheet_names, ["README", "Decoy Summary", "Agility Export"])

        golden = _load_golden_raw()
        intended = pd.read_excel(workbook, sheet_name="Agility Export")
        readme = pd.read_excel(workbook, sheet_name="README")
        decoy = pd.read_excel(workbook, sheet_name="Decoy Summary")

        assert_frame_equal(intended, golden, check_dtype=False)
        self.assertEqual(intended.shape, (100, 48))
        self.assertLess(len(readme), len(intended))
        self.assertLess(len(decoy), len(intended))
        self.assertTrue(
            decoy["Headline"]
            .fillna("")
            .astype(str)
            .str.contains("DECOY SHEET - do not select", regex=False)
            .all()
        )

    def test_malformed_fixture_upload_quality_and_cleaning_termination(self) -> None:
        raw = _load_malformed_raw()
        normalized = normalize_uploaded_dataframe(raw)
        report = build_upload_quality_report(raw, normalized)

        self.assertEqual(raw.shape, (14, 48))
        self.assertEqual(len(normalized), len(raw))
        self.assertEqual(
            [warning["title"] for warning in report["warnings"]],
            [
                "Some date values could not be parsed",
                "Some rows are missing media type",
                "Some media types are unrecognized",
            ],
        )
        self.assertEqual(report["date_issue_row_numbers"], [2])
        self.assertEqual(report["media_type_issue_row_numbers"], [3])
        self.assertEqual(report["unrecognized_media_type_values"], ["THREADS"])

        state = _run_basic_cleaning_like_page(raw)
        self.assertEqual(
            len(state["df_traditional"]) + len(state["df_social"]) + len(state["df_dupes"]),
            len(normalized),
        )


if __name__ == "__main__":
    unittest.main()
