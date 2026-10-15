from __future__ import annotations

from streamlit import config, runtime
from streamlit.url_util import make_url_path


def register_session_download(data: bytes, file_name: str) -> str:
    """Serve prepared bytes through Streamlit without mounting a download widget.

    This isolates the internal media-manager API used by Streamlit's own download
    button. Compatibility is tested against the runtime in Pipfile.lock (1.56.0).
    Register on every page run: Streamlit clears each session's media references
    between runs and garbage-collects files that are no longer referenced.
    """
    file_url = runtime.get_instance().media_file_mgr.add(
        data,
        "application/octet-stream",
        coordinates="session-file-download",
        file_name=file_name,
        is_for_static_download=True,
    )
    return make_url_path(config.get_option("server.baseUrlPath"), file_url)
