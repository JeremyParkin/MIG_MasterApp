# mig_functions.py
from __future__ import annotations
import streamlit as st








# ----------------------------
# Display helpers
# ----------------------------
def format_number(num: float | int) -> str:
    """Helper to display large integers with K/M/B suffixes."""
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
        # Preserve int look if it's actually an int
        return str(int(n)) if n.is_integer() else str(n)


# ----------------------------
# OpenAI Usage Metering
# ----------------------------
_OPENAI_PRICES = {
    "gpt-5.4-nano": {"in": 0.20, "out": 1.25},
    "gpt-5.4-mini": {"in": 0.75, "out": 4.50},
    "gpt-5-mini": {"in": 0.25, "out": 2.00},
    "gpt-4.1-mini": {"in": 0.40, "out": 1.60},
    # "gpt-4o-mini": {"in": 0.15, "out": 0.60},
    # "gpt-5": {"in": 1.25, "out": 10.00},
    # Optional extras if you ever use them:
    # "gpt-4o":      {"in": 2.50, "out": 10.00},
    # "gpt-5-nano":  {"in": 0.05, "out": 0.40},
}


def _init_api_meter() -> None:
    """Ensure the session usage meter object exists."""
    if "api_meter" not in st.session_state:
        st.session_state.api_meter = {
            "in_tokens": 0,
            "out_tokens": 0,
            "cost_usd": 0.0,
            "by_model": {},
        }


def add_api_usage(resp, model_name: str) -> None:
    """
    Record usage & cost from an OpenAI response object.

    Usage:
        resp = client.chat.completions.create(...)
        mig.add_api_usage(resp, model_id)
    """
    _init_api_meter()
    usage = getattr(resp, "usage", None)
    if not usage:
        return

    in_t = int(getattr(usage, "prompt_tokens", 0) or 0)
    out_t = int(getattr(usage, "completion_tokens", 0) or 0)

    meter = st.session_state.api_meter
    meter["in_tokens"] += in_t
    meter["out_tokens"] += out_t

    prices = _OPENAI_PRICES.get((model_name or "").lower(), {"in": 0.15, "out": 0.60})
    in_cost = (in_t / 1_000_000) * prices["in"]
    out_cost = (out_t / 1_000_000) * prices["out"]
    meter["cost_usd"] += (in_cost + out_cost)

    # Per-model breakdown
    bym = meter["by_model"].setdefault(model_name, {"in_tokens": 0, "out_tokens": 0, "cost_usd": 0.0})
    bym["in_tokens"] += in_t
    bym["out_tokens"] += out_t
    bym["cost_usd"] += (in_cost + out_cost)


def reset_api_meter() -> None:
    """Reset the usage meter (call this from your 'Start Over' button handler)."""
    st.session_state.api_meter = {
        "in_tokens": 0,
        "out_tokens": 0,
        "cost_usd": 0.0,
        "by_model": {},
    }