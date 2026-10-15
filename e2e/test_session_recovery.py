from __future__ import annotations

import base64
import random
import re
from pathlib import Path

import dill
import pandas as pd
from playwright.sync_api import Page, Route, expect

from test_upload_cleaning_spine import (
    GOLDEN_CSV,
    _go_to_basic_cleaning,
    _run_basic_cleaning,
    _submit_upload,
)


def _block_optional_components(page: Page, *, include_button: bool = False) -> list[str]:
    blocked: list[str] = []

    def block(route: Route) -> None:
        blocked.append(route.request.url)
        route.abort()

    names = "DownloadButton|LinkButton|ButtonGroup"
    if include_button:
        names += "|Button"
    page.route(re.compile(rf"/static/js/(?:{names})\.[^/]+\.js$"), block)
    return blocked


def _open_save_load(page: Page) -> None:
    expect(page.get_by_role("link", name=re.compile("Getting Started"))).to_be_visible()
    more_pages = page.get_by_text(re.compile(r"^View \d+ more$"))
    if more_pages.is_visible():
        more_pages.click()
    page.get_by_role("link", name=re.compile("Save & Load")).click()
    expect(page.get_by_role("heading", name="Save & Load", exact=True)).to_be_visible()


def _download_session(page: Page, tmp_path: Path) -> tuple[Path, dict]:
    link = page.get_by_role("link", name="Download session file", exact=True)
    expect(link).to_be_visible(timeout=30_000)
    url = link.get_attribute("href")
    assert "/media/" in url
    assert len(url) < 200

    with page.expect_download() as download_info:
        link.click()
    download = download_info.value
    assert download.failure() is None
    assert download.suggested_filename.endswith(".pkl")
    saved_path = tmp_path / "recovery.pkl"
    download.save_as(saved_path)
    return saved_path, dill.loads(saved_path.read_bytes())


def _build_control(page: Page, *, rebuild: bool = False):
    frame = page.frame_locator('iframe[title="ui.session_file_controls.session_file_build_control"]')
    return frame.get_by_role("link", name="Rebuild session file" if rebuild else "Build session file", exact=True)


def _build_session_file(page: Page) -> None:
    expect(page.get_by_role("link", name="Download session file", exact=True)).to_have_count(0)
    expect(page.get_by_text(re.compile("^Current session file built:"))).to_have_count(0)
    expect(page.get_by_role("button", name="Build session file", exact=True)).to_have_count(0)
    _build_control(page).click()
    expect(page.get_by_role("link", name="Download session file", exact=True)).to_be_visible(timeout=30_000)
    expect(_build_control(page, rebuild=True)).to_be_visible()


def test_session_recovery_with_download_component_blocked(e2e_page: Page, tmp_path: Path) -> None:
    page = e2e_page
    blocked = _block_optional_components(page)
    client = 'Recovery & "quoted" café <client>'
    _submit_upload(page, GOLDEN_CSV, client=client)
    _go_to_basic_cleaning(page)
    _run_basic_cleaning(page)
    _open_save_load(page)
    _build_session_file(page)

    link = page.get_by_role("link", name="Download session file", exact=True)
    expect(link).to_have_attribute("download", re.compile(re.escape(client) + r" - .*\.pkl$"))
    saved_path, payload = _download_session(page, tmp_path)
    assert not any("/DownloadButton." in url for url in blocked)
    assert payload["client_name"] == client
    assert payload["standard_step"] is True
    assert len(payload["df_traditional"]) == 49
    assert len(payload["df_social"]) == 47

    built_at = page.get_by_text(re.compile("^Current session file built:")).inner_text()
    prepared_href = link.get_attribute("href")
    _go_to_basic_cleaning(page)
    _open_save_load(page)
    expect(page.get_by_text(built_at, exact=True)).to_be_visible()
    expect(link).to_have_attribute("href", prepared_href)
    expect(_build_control(page, rebuild=True)).to_be_visible()
    # Returning must renew the server-side file reference, not just retain a dead URL.
    _download_session(page, tmp_path)

    page.reload()
    expect(page.get_by_text("Please upload a CSV before saving.")).to_be_visible()
    expect(link).to_have_count(0)
    page.locator("input[type='file']").set_input_files(str(saved_path))
    expect(page.get_by_text(re.compile("Session state loaded successfully"))).to_be_visible(timeout=15_000)
    _go_to_basic_cleaning(page)
    expect(page.get_by_text("Row reconciliation passed")).to_be_visible()
    _open_save_load(page)
    _build_session_file(page)
    _, restored = _download_session(page, tmp_path)
    assert restored["client_name"] == client
    pd.testing.assert_frame_equal(restored["df_traditional"], payload["df_traditional"])
    pd.testing.assert_frame_equal(restored["df_social"], payload["df_social"])
    assert "_prepared_session_file" not in restored
    assert "_restore_session_file" not in restored


def test_large_session_download_does_not_embed_or_prefetch_the_file(e2e_page: Page, tmp_path: Path) -> None:
    page = e2e_page
    _block_optional_components(page)
    _open_save_load(page)
    rng = random.Random(17)
    rows = 32 * 1024
    large_df = pd.DataFrame({
        "Group ID": range(rows),
        "Snippet": [base64.b64encode(rng.randbytes(768)).decode("ascii") for _ in range(rows)],
    })
    seed = dill.dumps(
        {
            "_snapshot_version": 2,
            "df_names": ["df_traditional"],
            "df_traditional": large_df,
            "client_name": "Large recovery session",
            "upload_step": True,
            "standard_step": True,
        }
    )
    assert len(seed) > 32 * 1024**2
    page.locator("input[type='file']").set_input_files(
        {"name": "large-session.pkl", "mimeType": "application/octet-stream", "buffer": seed}
    )
    expect(page.get_by_text(re.compile("Session state loaded successfully"))).to_be_visible(timeout=30_000)
    downloads = []
    page.on("request", lambda request: downloads.append(request.url) if re.search(r"/media/[^/]+\.bin$", request.url) else None)
    _build_session_file(page)
    assert downloads == [], "Building must not transfer the file before the download click."
    assert len(page.content()) < 150_000, "File bytes must never be embedded in the page."
    saved_path, payload = _download_session(page, tmp_path)
    pd.testing.assert_frame_equal(payload["df_traditional"], large_df)

    page.reload()
    expect(page.get_by_text("Please upload a CSV before saving.")).to_be_visible()
    page.locator("input[type='file']").set_input_files(str(saved_path))
    expect(page.get_by_text(re.compile("Session state loaded successfully"))).to_be_visible(timeout=30_000)
    _build_session_file(page)
    _, restored = _download_session(page, tmp_path)
    pd.testing.assert_frame_equal(restored["df_traditional"], large_df)
    assert "_prepared_session_file" not in restored


def test_session_controls_work_without_native_button_scripts(
    page: Page, streamlit_server: str, tmp_path: Path
) -> None:
    # Block Button before the app starts; otherwise its already-imported module is cached.
    blocked = _block_optional_components(page, include_button=True)
    page.goto(f"{streamlit_server}/Save_Load", wait_until="domcontentloaded")
    expect(page.get_by_role("heading", name="Save & Load", exact=True)).to_be_visible()
    expected_df = pd.DataFrame({"Group ID": [1], "Headline": ["Analyst's reviewed story"]})
    seed = dill.dumps(
        {
            "_snapshot_version": 2,
            "df_names": ["df_traditional"],
            "df_traditional": expected_df,
            "client_name": "Text-only recovery",
            "analyst_note": "Keep this live session intact",
            "upload_step": True,
            "standard_step": True,
        }
    )
    page.locator("input[type='file']").set_input_files(
        {"name": "session.pkl", "mimeType": "application/octet-stream", "buffer": seed}
    )
    expect(page.get_by_text(re.compile("Session state loaded successfully"))).to_be_visible(timeout=15_000)
    build_link = _build_control(page)
    expect(build_link).to_be_visible()
    expect(page.get_by_role("link", name="Download session file", exact=True)).to_have_count(0)
    session_url = page.url
    expect(build_link).to_be_enabled()
    busy_state = build_link.evaluate("""link => {
        link.click();
        const state = {disabled: link.getAttribute('aria-disabled'), label: link.textContent};
        link.click();
        return state;
    }""")
    assert busy_state == {"disabled": "true", "label": "Building session file…"}
    saved_path, payload = _download_session(page, tmp_path)
    expect(page.get_by_text(re.compile("^Session file built at"))).to_be_visible()
    expect(page.get_by_text(re.compile("^Session file rebuilt at"))).to_have_count(0)
    assert page.url == session_url, "The build link must preserve the live session."
    expect(page.get_by_role("button", name="Build session file", exact=True)).to_have_count(0)
    expect(page.get_by_role("button", name="Download session file", exact=True)).to_have_count(0)
    assert not any(re.search(r"/DownloadButton\.", url) for url in blocked)
    assert payload["analyst_note"] == "Keep this live session intact"
    pd.testing.assert_frame_equal(payload["df_traditional"], expected_df)
    assert "_session_file_text_build" not in payload

    built_at = page.get_by_text(re.compile("^Current session file built:")).inner_text()
    expect(page.get_by_text(built_at, exact=True)).to_be_visible()
    _build_control(page, rebuild=True).click()
    expect(page.get_by_text(re.compile("^Session file rebuilt at"))).to_be_visible(timeout=15_000)
    _, rebuilt = _download_session(page, tmp_path)
    assert rebuilt["analyst_note"] == payload["analyst_note"]
    assert "_prepared_session_file" not in rebuilt

    page.reload()
    expect(page.get_by_text("Please upload a CSV before saving.")).to_be_visible()
    page.locator("input[type='file']").set_input_files(str(saved_path))
    expect(page.get_by_text(re.compile("Session state loaded successfully"))).to_be_visible(timeout=15_000)
    expect(_build_control(page)).to_be_visible()
