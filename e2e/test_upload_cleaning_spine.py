from __future__ import annotations

import re
from pathlib import Path

from playwright.sync_api import Page, expect


FIXTURE_DIR = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "agility"
GOLDEN_CSV = FIXTURE_DIR / "agility_golden_corpus.csv"
MALFORMED_CSV = FIXTURE_DIR / "agility_malformed_inputs.csv"
MULTISHEET_WORKBOOK = FIXTURE_DIR / "agility_golden_multisheet_upload.xlsx"
CLEANED_WORKBOOK = FIXTURE_DIR / "agility_golden_cleaned_workbook.xlsx"


def _upload_file(page: Page, path: Path) -> None:
    page.locator("input[type='file']").set_input_files(str(path))


def _submit_upload(page: Page, path: Path, *, client: str = "BDC", period: str = "Fixture E2E") -> None:
    _upload_file(page, path)
    page.get_by_role("textbox", name="Client organization name*").fill(client)
    page.get_by_role("textbox", name="Reporting period or focus*").fill(period)
    page.get_by_role("button", name="Submit").click()
    expect(page.get_by_text(re.compile(r"File uploaded:"))).to_be_visible(timeout=15_000)


def _select_worksheet(page: Page, sheet_name: str) -> None:
    page.get_by_label("Select a sheet:").click()
    page.locator("[data-testid='stSelectboxVirtualDropdown']").get_by_text(sheet_name, exact=True).click()


def _go_to_basic_cleaning(page: Page) -> None:
    page.get_by_role("link", name=re.compile("Basic Cleaning")).click()
    expect(page.get_by_role("heading", name="Basic Cleaning")).to_be_visible()


def _run_basic_cleaning(page: Page) -> None:
    page.get_by_role("button", name="Run Basic Cleaning").click()
    expect(page.get_by_text("Basic cleaning completed.")).to_be_visible(timeout=30_000)


def _upload_warning(page: Page, text: str | re.Pattern[str]):
    return page.get_by_role("alert").filter(has_text=text)


def test_golden_csv_upload_and_basic_cleaning(e2e_page: Page) -> None:
    page = e2e_page

    _submit_upload(page, GOLDEN_CSV)

    expect(page.get_by_role("heading", name="Initial Stats")).to_be_visible()
    expect(page.get_by_text("Mentions", exact=True).first).to_be_visible()
    expect(page.get_by_text("100", exact=True).first).to_be_visible()

    _go_to_basic_cleaning(page)
    _run_basic_cleaning(page)

    expect(page.get_by_role("heading", name="Row Reconciliation")).to_be_visible()
    expect(page.get_by_text("Row reconciliation passed")).to_be_visible()
    for label in ["Original Rows", "Traditional", "Social", "Deleted Duplicates", "Reconciled Total"]:
        expect(page.get_by_text(label, exact=True).first).to_be_visible()
    for value in ["100", "49", "47", "4"]:
        expect(page.get_by_text(value, exact=True).first).to_be_visible()


def test_malformed_csv_upload_warnings_and_drop_invalid_date(e2e_page: Page) -> None:
    page = e2e_page

    _submit_upload(page, MALFORMED_CSV)

    date_warning = _upload_warning(page, re.compile(r"could not be converted into\s*Date"))
    missing_type_warning = _upload_warning(page, "have no media type value")
    unrecognized_warning = _upload_warning(page, "does not currently normalize these media type value(s): THREADS")

    expect(date_warning).to_be_visible()
    expect(missing_type_warning).to_be_visible()
    expect(unrecognized_warning).to_be_visible()
    expect(page.get_by_role("button", name="Drop 1 row").first).to_be_enabled()

    page.get_by_role("button", name="Drop 1 row").first.click()

    expect(date_warning).to_have_count(0, timeout=10_000)
    expect(missing_type_warning).to_be_visible()
    expect(unrecognized_warning).to_be_visible()
    expect(page.get_by_role("heading", name="Initial Stats")).to_be_visible()


def test_multisheet_xlsx_selects_agility_export(e2e_page: Page) -> None:
    page = e2e_page

    _upload_file(page, MULTISHEET_WORKBOOK)
    expect(page.get_by_label("Select a sheet:")).to_be_visible(timeout=15_000)
    _select_worksheet(page, "Agility Export")

    page.get_by_role("textbox", name="Client organization name*").fill("BDC")
    page.get_by_role("textbox", name="Reporting period or focus*").fill("Fixture E2E")
    page.get_by_role("button", name="Submit").click()

    expect(page.get_by_text(re.compile(r"File uploaded:"))).to_be_visible(timeout=15_000)
    expect(page.get_by_role("heading", name="Initial Stats")).to_be_visible()
    expect(page.get_by_text("Mentions", exact=True).first).to_be_visible()
    expect(page.get_by_text("100", exact=True).first).to_be_visible()


def test_cleaned_workbook_clean_trad_reentry_boundary(e2e_page: Page) -> None:
    page = e2e_page

    _upload_file(page, CLEANED_WORKBOOK)
    expect(page.get_by_label("Select a sheet:")).to_be_visible(timeout=15_000)
    _select_worksheet(page, "CLEAN TRAD")

    page.get_by_role("textbox", name="Client organization name*").fill("BDC")
    page.get_by_role("textbox", name="Reporting period or focus*").fill("Clean Workbook Re-entry")
    page.get_by_role("button", name="Submit").click()

    expect(page.get_by_text(re.compile(r"File uploaded:"))).to_be_visible(timeout=15_000)
    expect(page.get_by_role("heading", name="Initial Stats")).to_be_visible()
    expect(page.get_by_text("Mentions", exact=True).first).to_be_visible()
    expect(page.get_by_text("49", exact=True).first).to_be_visible()

    _go_to_basic_cleaning(page)
    expect(page.get_by_role("heading", name="Cleaning Options")).to_be_visible()
    expect(page.get_by_role("button", name="Run Basic Cleaning")).to_be_enabled()
