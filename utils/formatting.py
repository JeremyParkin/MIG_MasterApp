# formatting.py
from __future__ import annotations

NUMERIC_FORMAT_DICT = {
    "AVE": "${0:,.2f}",
    "A": "${0:,.2f}",
    "Domain Authority": "{0:,.0f}",
    "Impressions": "{:,.0f}",
    "Effective Reach": "{:,.0f}",
    "Mentions": "{:,.0f}",
    "Engagements": "{:,.0f}",
}

def format_number(num: float | int) -> str:
    """Format large numbers with K/M/B suffixes."""
    try:
        n = float(num)
    except Exception:
        return str(num)

    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.1f} B"
    elif n >= 1_000_000:
        return f"{n / 1_000_000:.1f} M"
    elif n >= 1_000:
        return f"{n / 1_000:.1f} K"
    else:
        return str(int(n)) if n.is_integer() else str(n)