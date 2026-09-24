from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo


def format_local_timestamp(timezone: str = "America/Toronto") -> str:
    return datetime.now(ZoneInfo(timezone)).strftime("%Y-%m-%d %H:%M")
