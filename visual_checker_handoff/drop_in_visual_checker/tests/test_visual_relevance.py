from __future__ import annotations

from processing.visual_relevance import estimate_cost, likely_url_columns, normalize_url


def test_normalize_url_adds_https_for_domain_like_values():
    assert normalize_url("example.com/story") == "https://example.com/story"


def test_normalize_url_preserves_absolute_urls():
    assert normalize_url("https://example.com/story") == "https://example.com/story"


def test_likely_url_columns_prioritizes_url_and_link_names():
    assert likely_url_columns(["Headline", "Article Link", "URL", "Outlet"])[:2] == ["Article Link", "URL"]


def test_estimate_cost_uses_uncached_and_cached_input_rates():
    usage = {"input_tokens": 1000, "cached_input_tokens": 250, "output_tokens": 500}
    assert estimate_cost("gpt-4.1-mini", usage) == 0.001125
