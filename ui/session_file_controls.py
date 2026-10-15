from __future__ import annotations

from html import escape
from pathlib import Path

import streamlit as st
from streamlit.components.v1 import declare_component


_text_build_link = declare_component(
    "session_file_build_control",
    path=str(Path(__file__).resolve().parent / "components" / "session_build_link"),
)


def _request_session_build() -> None:
    st.session_state._session_file_build_requested = True


def render_session_build_link(*, rebuild: bool) -> None:
    """Request a build in the live session without the native Button module."""
    action = "Rebuild" if rebuild else "Build"
    _text_build_link(
        label=f"{action} session file",
        key="_session_file_text_build",
        default=None,
        on_change=_request_session_build,
    )


def render_session_download_link(url: str, file_name: str) -> None:
    st.markdown(
        """
        <style>
        a.session-file-download {
            display: inline-block;
            padding: 0.5rem 0.75rem;
            border: 1px solid rgba(128, 128, 128, 0.5);
            border-radius: 0.5rem;
            color: inherit;
            font-weight: 400;
            line-height: 1.6;
            text-decoration: none;
        }
        a.session-file-download:hover {
            color: var(--primary-color, #AB26EF);
            border-color: currentColor;
        }
        a.session-file-download:focus-visible {
            outline: 2px solid currentColor;
            outline-offset: 2px;
        }
        </style>
        """
        f'<a class="session-file-download" href="{escape(url, quote=True)}" '
        f'download="{escape(file_name, quote=True)}" target="_self">Download session file</a>',
        unsafe_allow_html=True,
    )
