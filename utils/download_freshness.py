"""Content signatures for the generated downloads kept in memory."""

import hashlib
import json

import pandas as pd


def content_fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def frame_fingerprint(frame: pd.DataFrame) -> str:
    # Ignore the index, which these downloads do not write. Some source
    # columns contain lists or dictionaries that pandas cannot hash directly.
    try:
        values = pd.util.hash_pandas_object(frame, index=False).values.tobytes()
    except TypeError:
        values = frame.to_json(orient="values", date_format="iso", default_handler=str).encode()
    schema = content_fingerprint(list(zip(frame.columns, frame.dtypes.astype(str))))
    return hashlib.sha256(schema.encode() + values).hexdigest()


def workbook_fingerprint(sheets, original_ave_col) -> str:
    parts = []
    for name, frame, _, _ in sheets:
        if name == "EXPORT METADATA":
            frame = frame.loc[~frame["Field"].isin(("Export Timestamp", "Session Duration"))]
        parts.append((name, frame_fingerprint(frame)))
    return content_fingerprint((original_ave_col, parts))
