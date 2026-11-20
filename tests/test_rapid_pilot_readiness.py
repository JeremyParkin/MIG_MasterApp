import io
import zipfile
from unittest.mock import patch

import pandas as pd

from processing.download_exports import build_clean_workbook_bytes, build_report_copy_docx_bytes
from ui.app_shell import build_pages
from utils.session_snapshot import build_session_snapshot_bytes, load_session_state_from_file


class SessionState(dict):
    __setattr__ = dict.__setitem__


def test_deployment_switch_blocks_rapid_navigation_and_exports_after_restore(monkeypatch):
    source = pd.DataFrame({"Group ID": [1], "Headline": ["Story"]})
    state = {
        "df_traditional": source,
        "df_jev_sentiment_grouped_rows": source,
        "df_jev_sentiment_unique": source.assign(
            **{"Jev Sentiment": "POSITIVE", "Jev 5-Way Sentiment": "SOMEWHAT POSITIVE"}
        ),
        "rapid_sentiment_observation_output": {"overall_observation": "Rapid-only narrative"},
        "sentiment_observation_output": {"overall_observation": "Established narrative"},
    }
    snapshot, _ = build_session_snapshot_bytes(state)
    restored = SessionState()
    load_session_state_from_file(restored, io.BytesIO(snapshot))
    monkeypatch.setenv("MIG_ENABLE_RAPID_LABELING", "0")

    with patch("ui.app_shell.st.Page", side_effect=lambda path, **kwargs: kwargs["title"]):
        pages = build_pages()
        assert "Sentiment" in pages and "Tagging" in pages
        assert "Rapid Labeling" not in pages
        monkeypatch.delenv("MIG_ENABLE_RAPID_LABELING")
        assert "Rapid Labeling" in build_pages()

    workbook = pd.ExcelFile(io.BytesIO(build_clean_workbook_bytes(restored)))
    assert "RAPID LABELING SAMPLE" in workbook.sheet_names
    monkeypatch.setenv("MIG_ENABLE_RAPID_LABELING", "0")
    workbook = pd.ExcelFile(io.BytesIO(build_clean_workbook_bytes(restored)))
    assert "RAPID LABELING SAMPLE" not in workbook.sheet_names
    assert "Final Rapid Sentiment 3-Way" not in workbook.parse("CLEAN TRAD").columns
    assert not workbook.parse("EXPORT METADATA")["Field"].dropna().str.startswith("Rapid Labeling").any()
    assert "Rapid-only narrative" not in _report_xml(restored)
    assert "df_jev_sentiment_unique" in restored


def _report_xml(state):
    with zipfile.ZipFile(io.BytesIO(build_report_copy_docx_bytes(state))) as document:
        return document.read("word/document.xml").decode()
