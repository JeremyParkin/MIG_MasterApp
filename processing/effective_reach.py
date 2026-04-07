from __future__ import annotations

import math
import numpy as np
import pandas as pd

DEFAULT_EPS = 1e-12

DEFAULT_TRAD_MEDIA_PARAMS = {
    "print_generic": {
        "MID_IMPRESSIONS": 65_000,
        "HIGH_IMPRESSIONS": 1_500_000,
        "BENCHMARK_VIS": 0.38,
        "A_SIZE": 0.30,
        "A_MIN": 0.20,
        "A_MAX": 0.40,
        "LOW_IMPRESSIONS": 0,
        "GATE_ANCHOR": 150_000,
        "MIN_VISIBILITY": 0.09,
        "MAX_VISIBILITY": 0.30,
    },
    "print_magazine": {
        "MID_IMPRESSIONS": 75_000,
        "HIGH_IMPRESSIONS": 1_000_000,
        "BENCHMARK_VIS": 0.45,
        "A_SIZE": 0.30,
        "A_MIN": 0.20,
        "A_MAX": 0.40,
        "LOW_IMPRESSIONS": 0,
        "GATE_ANCHOR": 150_000,
        "MIN_VISIBILITY": 0.10,
        "MAX_VISIBILITY": 0.70,
    },
    "print_daily": {
        "MID_IMPRESSIONS": 50_000,
        "HIGH_IMPRESSIONS": 2_000_000,
        "BENCHMARK_VIS": 0.23,
        "A_SIZE": 0.30,
        "A_MIN": 0.20,
        "A_MAX": 0.40,
        "LOW_IMPRESSIONS": 0,
        "GATE_ANCHOR": 50_000,
        "MIN_VISIBILITY": 0.05,
        "MAX_VISIBILITY": 0.40,
    },
    "tv": {
        "MID_IMPRESSIONS": 500_000,
        "HIGH_IMPRESSIONS": 1_000_000,
        "BENCHMARK_VIS": 0.488,
        "A_SIZE": 0.85,
        "A_MIN": 0.05,
        "A_MAX": 0.30,
        "LOW_IMPRESSIONS": 0,
        "GATE_ANCHOR": 750_000,
        "MIN_VISIBILITY": 0.15,
        "MAX_VISIBILITY": 0.65,
    },
    "radio": {
        "MID_IMPRESSIONS": 100_000,
        "HIGH_IMPRESSIONS": 500_000,
        "BENCHMARK_VIS": 0.375,
        "A_SIZE": 0.25,
        "A_MIN": 0.05,
        "A_MAX": 0.30,
        "LOW_IMPRESSIONS": 0,
        "GATE_ANCHOR": 150_000,
        "MIN_VISIBILITY": 0.10,
        "MAX_VISIBILITY": 0.55,
    },
    "online": {
        "DAILY_VISITOR_RATE": 0.30,
        "PAGES_PER_VISIT": 2.0,
        "ONLINE_DENOMINATOR_COEFF": 2.3746,
        "ONLINE_DENOMINATOR_EXPONENT": 0.2009,
    },
}

DEFAULT_PLATFORM_PARAMS = {
    "x": {
        "MID_FOLLOWERS": 70_500,
        "HIGH_FOLLOWERS": 5_000_000,
        "BENCHMARK_VIS": 0.015,
        "EXPECTED_ENG_RATE": 0.0012,
        "PERF_EXPONENT": 0.50,
        "PERF_FLOOR": 0.60,
        "PERF_CAP": 1.80,
        "GATE_FOLLOWERS_ANCHOR": 200_000,
        "MAX_VISIBILITY": 0.20,
    },
    "bluesky": {
        "MID_FOLLOWERS": 15_000,
        "HIGH_FOLLOWERS": 500_000,
        "BENCHMARK_VIS": 0.04,
        "EXPECTED_ENG_RATE": 0.0012,
        "PERF_EXPONENT": 0.50,
        "PERF_FLOOR": 0.60,
        "PERF_CAP": 1.80,
        "GATE_FOLLOWERS_ANCHOR": 40_000,
        "MAX_VISIBILITY": 0.22,
    },
    "instagram": {
        "MID_FOLLOWERS": 497_900,
        "HIGH_FOLLOWERS": 10_000_000,
        "BENCHMARK_VIS": 0.076,
        "EXPECTED_ENG_RATE": 0.0048,
        "PERF_EXPONENT": 0.50,
        "PERF_FLOOR": 0.60,
        "PERF_CAP": 1.80,
        "GATE_FOLLOWERS_ANCHOR": 1_100_000,
        "MAX_VISIBILITY": 0.20,
    },
    "facebook": {
        "MID_FOLLOWERS": 346_300,
        "HIGH_FOLLOWERS": 10_000_000,
        "BENCHMARK_VIS": 0.043,
        "EXPECTED_ENG_RATE": 0.0015,
        "PERF_EXPONENT": 0.50,
        "PERF_FLOOR": 0.60,
        "PERF_CAP": 1.80,
        "GATE_FOLLOWERS_ANCHOR": 850_000,
        "MAX_VISIBILITY": 0.15,
    },
    "linkedin": {
        "MID_FOLLOWERS": 26_500,
        "HIGH_FOLLOWERS": 1_000_000,
        "BENCHMARK_VIS": 0.121,
        "EXPECTED_ENG_RATE": 0.0060,
        "PERF_EXPONENT": 0.50,
        "PERF_FLOOR": 0.60,
        "PERF_CAP": 1.90,
        "GATE_FOLLOWERS_ANCHOR": 50_000,
        "MAX_VISIBILITY": 0.40,
    },
    "tiktok": {
        "MID_FOLLOWERS": 46_900,
        "HIGH_FOLLOWERS": 10_000_000,
        "BENCHMARK_VIS": 0.25,
        "EXPECTED_ENG_RATE": 0.037,
        "PERF_EXPONENT": 0.60,
        "PERF_FLOOR": 0.55,
        "PERF_CAP": 2.50,
        "GATE_FOLLOWERS_ANCHOR": 110_000,
        "MAX_VISIBILITY": 2.0,
    },
    "youtube": {
        "MID_FOLLOWERS": 68_800,
        "HIGH_FOLLOWERS": 10_000_000,
        "BENCHMARK_VIS": 0.12,
        "EXPECTED_ENG_RATE": 0.0030,
        "PERF_EXPONENT": 0.60,
        "PERF_FLOOR": 0.55,
        "PERF_CAP": 2.50,
        "GATE_FOLLOWERS_ANCHOR": 150_000,
        "MAX_VISIBILITY": 5.0,
    },
    "reddit": {
        "MID_FOLLOWERS": 25_000,
        "HIGH_FOLLOWERS": 2_000_000,
        "BENCHMARK_VIS": 0.08,
        "EXPECTED_ENG_RATE": 0.01,
        "PERF_EXPONENT": 0.70,
        "PERF_FLOOR": 0.50,
        "PERF_CAP": 2.50,
        "GATE_FOLLOWERS_ANCHOR": 75_000,
        "MAX_VISIBILITY": 0.60,
    },
}


def clamp(x: float, lo: float, hi: float) -> float:
    return min(max(x, lo), hi)


def safe_ln1p(x: float) -> float:
    return math.log1p(max(0.0, float(x)))


def normalize_platform(value) -> str | None:
    if pd.isna(value):
        return None

    s = str(value).strip().lower()
    aliases = {
        "twitter": "x",
        "x.com": "x",
        "x (twitter)": "x",
        "ig": "instagram",
        "insta": "instagram",
        "fb": "facebook",
        "meta": "facebook",
        "li": "linkedin",
        "ln": "linkedin",
        "tt": "tiktok",
        "tik tok": "tiktok",
        "yt": "youtube",
        "you tube": "youtube",
    }
    return aliases.get(s, s)


def normalize_trad_media_type(media_type: str) -> str | None:
    s = str(media_type).strip().lower()
    mapping = {
        "online": "online",
        "print": "print_generic",
        "tv": "tv",
        "radio": "radio",
    }
    return mapping.get(s)


def trad_size_percentile_ln(impressions: float, high_i: float) -> float:
    den = safe_ln1p(high_i)
    if den <= 0:
        return 0.0
    return clamp(safe_ln1p(impressions) / den, 0.0, 1.0)


def compute_trad_vis_size(impressions: float, p: dict) -> float:
    mid_i = float(p["MID_IMPRESSIONS"])
    high_i = float(p["HIGH_IMPRESSIONS"])
    base_vis = float(p["BENCHMARK_VIS"])

    p_size = trad_size_percentile_ln(impressions, high_i)
    p0 = trad_size_percentile_ln(mid_i, high_i)
    dist = abs(p_size - p0)

    a_used = clamp(
        float(p["A_SIZE"]) * (1.0 + dist),
        float(p["A_MIN"]),
        float(p["A_MAX"]),
    )

    return base_vis * math.exp(-a_used * (p_size - p0))


def compute_trad_gate(impressions: float, p: dict) -> float:
    anchor = float(p["GATE_ANCHOR"])
    impressions = max(float(impressions), 0.0)

    if impressions <= anchor:
        return 1.0

    gate = 1.0 / (1.0 + safe_ln1p(impressions / anchor - 1.0))
    return clamp(gate, 0.55, 1.0)


def compute_online_single(impressions: float, p: dict) -> int:
    impressions = float(impressions)
    if impressions <= 0:
        return 0

    eff_reach_raw = (
        impressions
        * float(p["DAILY_VISITOR_RATE"])
        * float(p["PAGES_PER_VISIT"])
    ) / (
        float(p["ONLINE_DENOMINATOR_COEFF"])
        * (impressions ** float(p["ONLINE_DENOMINATOR_EXPONENT"]))
    )

    return int(round(eff_reach_raw))


def compute_trad_single(media_type, impressions, trad_params) -> int | None:
    key = normalize_trad_media_type(media_type)
    if key is None:
        return None

    if key == "online":
        return compute_online_single(impressions, trad_params["online"])

    if key not in trad_params:
        return None

    p = trad_params[key]
    impressions = float(impressions)
    if impressions <= 0:
        return 0

    vis_final = clamp(
        compute_trad_vis_size(impressions, p) * compute_trad_gate(impressions, p),
        float(p["MIN_VISIBILITY"]),
        float(p["MAX_VISIBILITY"]),
    )
    return int(round(impressions * vis_final))


def compute_vis_size(followers, p):
    f = max(float(followers), 1.0)
    z = (
        (np.log10(f) - np.log10(float(p["MID_FOLLOWERS"])))
        / (np.log10(float(p["HIGH_FOLLOWERS"])) - np.log10(float(p["MID_FOLLOWERS"])) + DEFAULT_EPS)
    )
    z = clamp(float(z), 0.0, 1.0)

    vis = float(p["BENCHMARK_VIS"]) * ((1 - z) * 1.35 + z * 0.92)
    return clamp(vis, 0.0, float(p["MAX_VISIBILITY"]))


def compute_perf_index(followers, engagements, p):
    eng_rate = max(float(engagements), 0.0) / (max(float(followers), 1.0) + DEFAULT_EPS)
    perf_ratio = eng_rate / (float(p["EXPECTED_ENG_RATE"]) + DEFAULT_EPS)
    raw = perf_ratio ** float(p["PERF_EXPONENT"])
    return clamp(raw, float(p["PERF_FLOOR"]), float(p["PERF_CAP"]))


def compute_gate(followers, p):
    f = max(float(followers), 1.0)
    z = (
        (np.log10(f) - np.log10(float(p["GATE_FOLLOWERS_ANCHOR"])))
        / (np.log10(float(p["HIGH_FOLLOWERS"])) - np.log10(float(p["GATE_FOLLOWERS_ANCHOR"])) + DEFAULT_EPS)
    )
    z = clamp(float(z), 0.0, 1.0)
    return clamp(1.0 - 0.35 * z, 0.50, 1.00)


def compute_social_single(platform, followers, engagements, platform_params) -> int | None:
    key = normalize_platform(platform)
    if key is None or key not in platform_params:
        return None

    followers = float(followers)
    if followers <= 0:
        return 0

    p = platform_params[key]
    vis_final = clamp(
        compute_vis_size(followers, p)
        * compute_perf_index(followers, engagements, p)
        * compute_gate(followers, p),
        0.0,
        float(p["MAX_VISIBILITY"]),
    )
    return int(round(followers * vis_final))


def apply_effective_reach_traditional(
    df: pd.DataFrame,
    trad_params: dict | None = None,
) -> pd.DataFrame:
    df = df.copy()
    trad_params = trad_params or DEFAULT_TRAD_MEDIA_PARAMS

    if "Type" not in df.columns or "Impressions" not in df.columns:
        df["Effective Reach"] = np.nan
        return df

    df["Impressions"] = pd.to_numeric(df["Impressions"], errors="coerce").fillna(0)

    df["Effective Reach"] = [
        compute_trad_single(row.get("Type", ""), row.get("Impressions", 0), trad_params)
        for _, row in df.iterrows()
    ]
    return df


def apply_effective_reach_social(
    df: pd.DataFrame,
    platform_params: dict | None = None,
) -> pd.DataFrame:
    df = df.copy()
    platform_params = platform_params or DEFAULT_PLATFORM_PARAMS

    if "Type" not in df.columns or "Impressions" not in df.columns or "Engagements" not in df.columns:
        df["Effective Reach"] = np.nan
        return df

    df["Impressions"] = pd.to_numeric(df["Impressions"], errors="coerce").fillna(0)
    df["Engagements"] = pd.to_numeric(df["Engagements"], errors="coerce").fillna(0)

    df["Effective Reach"] = [
        compute_social_single(
            row.get("Type", ""),
            row.get("Impressions", 0),
            row.get("Engagements", 0),
            platform_params,
        )
        for _, row in df.iterrows()
    ]
    return df