from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from pandas.testing import assert_frame_equal

from utils.io import build_upload_quality_report, normalize_uploaded_dataframe, read_uploaded_file


FIXTURES = Path(__file__).parent / "fixtures" / "agility"


class Upload(BytesIO):
    def __init__(self, data: bytes, name: str, file_id: str):
        super().__init__(data)
        self.name = name
        self.file_id = file_id


def test_csv_reruns_reuse_raw_data_and_same_named_replacement_reloads():
    upload = Upload(b"Value,Other\n1,2\n3,4\n", "data.csv", "first")
    state = {}
    with patch("utils.io.pd.read_csv", wraps=pd.read_csv) as read:
        state["df_untouched"] = read_uploaded_file(upload, session_state=state)
        first = state["df_untouched"]
        assert read_uploaded_file(upload, session_state=state) is first
        assert read.call_count == 1

        replacement = Upload(b"Value,Other\n5,6\n", "data.csv", "second")
        state["df_untouched"] = read_uploaded_file(replacement, session_state=state)
        assert read.call_count == 2
        assert state["df_untouched"]["Value"].tolist() == [5]


def test_csv_preserves_legacy_mixed_type_inference_across_chunk_boundaries():
    data = ("Value,Other\n" + "123,1\n" * 5000 + "text,2\n").encode()
    with pd.read_csv(BytesIO(data), chunksize=5000) as reader:
        expected = pd.concat(list(reader), ignore_index=True)
    actual = read_uploaded_file(Upload(data, "mixed.csv", "mixed"))
    assert_frame_equal(actual, expected)
    assert actual.iloc[0]["Value"] == 123
    assert actual.iloc[-1]["Value"] == "text"


def test_xlsx_reruns_keep_sheet_selector_and_reload_when_sheet_changes():
    upload = Upload((FIXTURES / "agility_golden_multisheet_upload.xlsx").read_bytes(), "data.xlsx", "book")
    state = {}
    with (
        patch("utils.io.st.selectbox", return_value="Agility Export") as select,
        patch("utils.io.pd.ExcelFile", wraps=pd.ExcelFile) as open_book,
        patch("utils.io.pd.read_excel", wraps=pd.read_excel) as read,
    ):
        state["df_untouched"] = read_uploaded_file(upload, session_state=state)
        first = state["df_untouched"]
        assert len(first) == 100
        assert read_uploaded_file(upload, session_state=state) is first
        assert open_book.call_count == 1
        assert read.call_count == 1
        assert select.call_count == 2

        other_sheet = next(name for name in select.call_args.args[1] if name != "Agility Export")
        select.return_value = other_sheet
        state["df_untouched"] = read_uploaded_file(upload, session_state=state)
        assert read.call_count == 2
        assert open_book.call_count == 2
        assert state["df_untouched"] is not first
        assert_frame_equal(state["df_untouched"], pd.read_excel(BytesIO(upload.getvalue()), sheet_name=other_sheet))


def test_normalization_and_quality_checks_leave_source_frames_unchanged():
    raw = pd.read_csv(FIXTURES / "agility_malformed_inputs.csv")
    original_raw = raw.copy(deep=True)
    normalized = normalize_uploaded_dataframe(raw)
    original_normalized = normalized.copy(deep=True)
    report = build_upload_quality_report(raw, normalized)
    assert report["warnings"]
    assert_frame_equal(raw, original_raw)
    assert_frame_equal(normalized, original_normalized)


def test_failed_replacement_does_not_mark_previous_data_as_the_new_upload():
    state = {}
    first = Upload(b"Value,Other\n1,2\n", "data.csv", "first")
    state["df_untouched"] = read_uploaded_file(first, session_state=state)
    key = state["_parsed_upload_key"]
    with patch("utils.io.pd.read_csv", side_effect=ValueError("bad file")):
        try:
            read_uploaded_file(Upload(b"bad", "data.csv", "second"), session_state=state)
        except ValueError:
            pass
        else:
            raise AssertionError("Expected upload parsing to fail")
    assert state["_parsed_upload_key"] == key
    assert state["df_untouched"]["Value"].tolist() == [1]
