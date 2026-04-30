from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd


APP_TIMEZONE = ZoneInfo("America/Toronto")


def now_local() -> datetime:
    return datetime.now(APP_TIMEZONE)


def format_local_timestamp(value: object | None = None) -> str:
    if value is None or value == "":
        dt = now_local()
    elif isinstance(value, datetime):
        dt = value if value.tzinfo else value.replace(tzinfo=APP_TIMEZONE)
        dt = dt.astimezone(APP_TIMEZONE)
    else:
        parsed = pd.to_datetime(value, errors="coerce")
        if pd.isna(parsed):
            return str(value)
        if getattr(parsed, "tzinfo", None) is None:
            parsed = parsed.tz_localize(APP_TIMEZONE)
        else:
            parsed = parsed.tz_convert(APP_TIMEZONE)
        dt = parsed.to_pydatetime()

    return f"{dt.strftime('%b')} {dt.day}, {dt.year} at {dt.strftime('%I:%M %p').lstrip('0')}"


def current_timestamp_storage_string() -> str:
    return now_local().isoformat()


def current_timestamp_filename_string() -> str:
    dt = now_local()
    return f"{dt.year:04d}-{dt.month:02d}-{dt.day:02d}-{dt.hour:02d}-{dt.minute:02d}"
