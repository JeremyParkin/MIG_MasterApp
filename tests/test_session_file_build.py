from pathlib import Path
from unittest.mock import patch

import dill
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from utils.session_snapshot import build_session_snapshot_bytes


SAVE_LOAD_PAGE = Path(__file__).resolve().parents[1] / "pages" / "12-Save_Load.py"


@pytest.fixture(autouse=True)
def media_registration():
    with patch("utils.session_download.register_session_download", return_value="/media/session.bin") as register:
        yield register


def _request_build(app: AppTest) -> AppTest:
    app.session_state["_session_file_build_requested"] = True
    return app.run()


def _save_page() -> AppTest:
    app = AppTest.from_file(str(SAVE_LOAD_PAGE), default_timeout=10)
    for key, value in {
        "upload_step": True,
        "standard_step": True,
        "client_name": "Original client",
        "df_traditional": pd.DataFrame({"Group ID": [1], "Headline": ["Saved story"]}),
    }.items():
        app.session_state[key] = value
    return app


def test_session_file_is_built_only_on_request_and_reused_until_rebuilt() -> None:
    with (
        patch("utils.session_snapshot.build_session_snapshot_bytes", wraps=build_session_snapshot_bytes) as build,
        patch("utils.time_display.format_local_timestamp", side_effect=["First build time", "Second build time"]),
    ):
        app = _save_page().run()
        assert not app.exception
        build.assert_not_called()
        assert "_prepared_session_file" not in app.session_state

        _request_build(app)
        assert not app.exception
        assert build.call_count == 1
        prepared = app.session_state["_prepared_session_file"]
        assert prepared["built_at"] == "First build time"
        assert dill.loads(prepared["data"])["client_name"] == "Original client"
        assert "encoded_data" not in prepared
        assert not app.button
        assert not app.get("download_button")

        app.session_state["client_name"] = "Updated client"
        app.run()
        assert not app.exception
        assert build.call_count == 1
        assert app.session_state["_prepared_session_file"] == prepared
        assert any(c.value == "Current session file built: First build time" for c in app.caption)

        _request_build(app)
        assert not app.exception
        assert build.call_count == 2
        rebuilt = app.session_state["_prepared_session_file"]
        payload = dill.loads(rebuilt["data"])
        assert payload["client_name"] == "Updated client"
        assert "_prepared_session_file" not in payload
        assert rebuilt["built_at"] == "Second build time"
        assert any(c.value == "Current session file built: Second build time" for c in app.caption)


def test_failed_rebuild_keeps_previous_download_and_timestamp() -> None:
    with patch("utils.session_snapshot.build_session_snapshot_bytes", wraps=build_session_snapshot_bytes) as build:
        app = _save_page().run()
        _request_build(app)
        assert not app.exception
        prepared = app.session_state["_prepared_session_file"]

        build.side_effect = RuntimeError("Build failed")
        _request_build(app)
        assert not app.exception
        assert app.session_state["_prepared_session_file"] == prepared
        assert any(e.value == "Error building session file: Build failed" for e in app.error)
        assert any(c.value == f"Current session file built: {prepared['built_at']}" for c in app.caption)


def test_persistent_component_value_does_not_request_another_build() -> None:
    with (
        patch("ui.session_file_controls._text_build_link", return_value=1) as text_link,
        patch("utils.session_snapshot.build_session_snapshot_bytes", wraps=build_session_snapshot_bytes) as build,
    ):
        app = _save_page().run()
        assert not app.exception
        build.assert_not_called()
        assert callable(text_link.call_args.kwargs["on_change"])

        _request_build(app)
        assert build.call_count == 1
        prepared = app.session_state["_prepared_session_file"]

        app.run()
        assert not app.exception
        assert build.call_count == 1
        assert app.session_state["_prepared_session_file"] == prepared

        _request_build(app)
        assert not app.exception
        assert build.call_count == 2
        payload = dill.loads(app.session_state["_prepared_session_file"]["data"])
        assert "_session_file_build_requested" not in payload


def test_registration_failure_preserves_previous_file_and_timestamp(media_registration) -> None:
    app = _save_page().run()
    _request_build(app)
    prepared = app.session_state["_prepared_session_file"]
    media_registration.side_effect = RuntimeError("File endpoint unavailable")
    _request_build(app)
    assert not app.exception
    assert app.session_state["_prepared_session_file"] == prepared
    assert any(e.value == "Error building session file: File endpoint unavailable" for e in app.error)


def test_existing_prepared_file_releases_legacy_base64_without_rebuilding() -> None:
    with patch("utils.session_snapshot.build_session_snapshot_bytes", wraps=build_session_snapshot_bytes) as build:
        app = _save_page()
        app.session_state["_prepared_session_file"] = {
            "data": b"previously prepared bytes",
            "encoded_data": "obsolete base64 copy",
            "file_name": "Previous.pkl",
            "built_at": "Original build time",
            "skipped_keys": [],
        }
        app.run()
        assert not app.exception
        build.assert_not_called()
        assert "encoded_data" not in app.session_state["_prepared_session_file"]
        assert app.session_state["_prepared_session_file"]["data"] == b"previously prepared bytes"
        assert any(c.value == "Current session file built: Original build time" for c in app.caption)
