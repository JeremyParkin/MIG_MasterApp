from __future__ import annotations

import streamlit as st


# ----------------------------
# Pricing
# ----------------------------
_OPENAI_PRICES = {
    "gpt-5.4-nano": {"in": 0.20, "out": 1.25},
    "gpt-5.4-mini": {"in": 0.75, "out": 4.50},
    "gpt-5-mini": {"in": 0.25, "out": 2.00},
    "gpt-4.1-mini": {"in": 0.40, "out": 1.60},
}


# ----------------------------
# Init / Accessors
# ----------------------------
def init_api_meter() -> None:
    """Ensure session API meter exists."""
    if "api_meter" not in st.session_state:
        st.session_state.api_meter = {
            "in_tokens": 0,
            "out_tokens": 0,
            "cost_usd": 0.0,
            "by_model": {},
        }


def get_api_cost_usd() -> float:
    """Get total session cost."""
    init_api_meter()
    return float(st.session_state.api_meter.get("cost_usd", 0.0) or 0.0)


def get_model_prices(model_name: str) -> dict:
    """Return input/output token pricing for a model."""
    return _OPENAI_PRICES.get(
        (model_name or "").lower(),
        {"in": 0.15, "out": 0.60},
    )


# ----------------------------
# Usage extraction
# ----------------------------
def extract_usage_tokens(resp) -> tuple[int, int]:
    """
    Extract input/output token usage from either:
    - Chat Completions API responses
    - Responses API responses

    Returns:
        (input_tokens, output_tokens)
    """
    usage = getattr(resp, "usage", None)
    if not usage:
        return 0, 0

    # Responses API names
    in_t = int(getattr(usage, "input_tokens", 0) or 0)
    out_t = int(getattr(usage, "output_tokens", 0) or 0)

    # Fallback: Chat Completions API names
    if in_t == 0:
        in_t = int(getattr(usage, "prompt_tokens", 0) or 0)
    if out_t == 0:
        out_t = int(getattr(usage, "completion_tokens", 0) or 0)

    return in_t, out_t


def estimate_cost_usd(in_tokens: int, out_tokens: int, model_name: str) -> float:
    """Estimate USD cost from token counts and model pricing."""
    prices = get_model_prices(model_name)
    in_cost = (int(in_tokens or 0) / 1_000_000) * prices["in"]
    out_cost = (int(out_tokens or 0) / 1_000_000) * prices["out"]
    return in_cost + out_cost


# ----------------------------
# Session application
# ----------------------------
def apply_usage_to_session(in_tokens: int, out_tokens: int, model_name: str) -> None:
    """
    Apply known token usage totals to the session meter.
    Call this on the main thread after a batch completes or after a single request.
    """
    init_api_meter()

    in_tokens = int(in_tokens or 0)
    out_tokens = int(out_tokens or 0)
    total_cost = estimate_cost_usd(in_tokens, out_tokens, model_name)

    meter = st.session_state.api_meter
    meter["in_tokens"] += in_tokens
    meter["out_tokens"] += out_tokens
    meter["cost_usd"] += total_cost

    by_model = meter["by_model"].setdefault(
        model_name,
        {"in_tokens": 0, "out_tokens": 0, "cost_usd": 0.0},
    )
    by_model["in_tokens"] += in_tokens
    by_model["out_tokens"] += out_tokens
    by_model["cost_usd"] += total_cost


def add_api_usage(resp, model_name: str) -> None:
    """
    Convenience helper for single-thread / main-thread usage:
    extract usage from a response object and apply it to session state.
    """
    in_t, out_t = extract_usage_tokens(resp)
    apply_usage_to_session(in_t, out_t, model_name)


def reset_api_meter() -> None:
    """Reset session API meter."""
    st.session_state.api_meter = {
        "in_tokens": 0,
        "out_tokens": 0,
        "cost_usd": 0.0,
        "by_model": {},
    }