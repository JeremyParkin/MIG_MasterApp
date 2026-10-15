from types import SimpleNamespace
from unittest.mock import patch

import dill
import pandas as pd
import pytest
from streamlit.runtime.media_file_manager import MediaFileManager
from streamlit.runtime.media_file_storage import MediaFileKind, MediaFileStorageError
from streamlit.runtime.memory_media_file_storage import MemoryMediaFileStorage

from utils.session_download import register_session_download
from utils.session_snapshot import build_session_snapshot_bytes


class State(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


def test_snapshot_success_serializes_the_payload_only_once() -> None:
    state = State(df_traditional=pd.DataFrame({"Group ID": [1]}), client_name="Example")
    with patch("utils.session_snapshot.dill.dumps", wraps=dill.dumps) as dumps:
        data, skipped = build_session_snapshot_bytes(state, payload_overrides={"checkpoint": 100})
        assert dumps.call_count == 1
    payload = dill.loads(data)
    assert skipped == []
    assert payload["checkpoint"] == 100
    pd.testing.assert_frame_equal(payload["df_traditional"], state.df_traditional)


def test_snapshot_diagnoses_bad_values_only_if_whole_serialization_fails() -> None:
    bad_value = (i for i in range(2))
    state = State(
        client_name="Example",
        unsupported=bad_value,
        df_bad=pd.DataFrame({"Unsupported": [bad_value]}),
        df_traditional=pd.DataFrame({"Group ID": [1]}),
    )
    data, skipped = build_session_snapshot_bytes(state)
    payload = dill.loads(data)
    assert set(skipped) == {"unsupported", "df_bad"}
    assert payload["df_names"] == ["df_traditional"]
    assert payload["client_name"] == "Example"


@pytest.mark.parametrize("base_path", ["", "/mig/app/"])
def test_download_adapter_serves_raw_bytes_with_a_short_base_path_aware_url(base_path: str) -> None:
    storage = MemoryMediaFileStorage("/media")
    manager = MediaFileManager(storage)
    data = b"prepared session bytes"
    with (
        patch("utils.session_download.runtime.get_instance", return_value=SimpleNamespace(media_file_mgr=manager)),
        patch("utils.session_download.config.get_option", return_value=base_path),
        patch("streamlit.runtime.media_file_manager._get_session_id", return_value="analyst-one"),
    ):
        url = register_session_download(data, "Example.pkl")
        assert url.startswith(f"{base_path.rstrip('/')}/media/")
        assert len(url) < 200
        stored = storage.get_file(url.rsplit("/", 1)[1])
        assert stored.content == data
        assert stored.filename == "Example.pkl"
        assert stored.kind == MediaFileKind.DOWNLOADABLE

        # Renew the reference on a rerun, and survive the manager's orphan sweep.
        manager.clear_session_refs("analyst-one")
        assert register_session_download(data, "Example.pkl") == url
        manager.remove_orphaned_files()
        manager.remove_orphaned_files()
        assert storage.get_file(url.rsplit("/", 1)[1]).content == data


def test_download_adapter_keeps_other_sessions_files_alive_during_cleanup() -> None:
    storage = MemoryMediaFileStorage("/media")
    manager = MediaFileManager(storage)
    with (
        patch("utils.session_download.runtime.get_instance", return_value=SimpleNamespace(media_file_mgr=manager)),
        patch("utils.session_download.config.get_option", return_value=""),
        patch("streamlit.runtime.media_file_manager._get_session_id") as session_id,
    ):
        session_id.return_value = "analyst-one"
        first = register_session_download(b"first session", "First.pkl")
        session_id.return_value = "analyst-two"
        second = register_session_download(b"second session", "Second.pkl")
        manager.clear_session_refs("analyst-one")
        manager.remove_orphaned_files()
        manager.remove_orphaned_files()
        with pytest.raises(MediaFileStorageError):
            storage.get_file(first.rsplit("/", 1)[1])
        assert storage.get_file(second.rsplit("/", 1)[1]).content == b"second session"
