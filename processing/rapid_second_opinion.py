from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable

import pandas as pd
from openai import OpenAI

from processing.ai_sentiment import CHAT_COMPLETIONS_TOOL_REASONING_EFFORT
from processing.ai_tagging import RESERVED_OTHER_TAG, normalize_tag_assignment
from processing.jev_sentiment import (
    DEFAULT_JEV_TAGGING_MODE,
    JEV_TAGGING_MODES,
    build_jev_tag_question_map,
    normalize_jev_tagging_mode,
)
from utils.api_meter import estimate_cost_usd, extract_usage_tokens


RAPID_REVIEW_MODEL = "gpt-5.6-luna"
RAPID_REVIEW_MAX_RETRIES = 2
RAPID_REVIEW_MAX_WORKERS = 4
RAPID_REVIEW_DEFAULT_BATCH_SIZE = 50
RAPID_REVIEW_CONFIDENCE_LOW = 0.60
RAPID_REVIEW_PROBABILITY_LOW = 0.60
RAPID_REVIEW_CLOSE_MARGIN = 0.15
RAPID_REVIEW_HIGH_MIXTURE = 2.0
RAPID_REVIEW_MANY_TAGS = 4
RAPID_REVIEW_RELEVANCE_STRONG = 0.75
RAPID_REVIEW_RELEVANCE_WEAK = 0.25

ANCHOR_OUTCOMES = ("NOT_RELEVANT", "-7", "-5", "-3", "-1", "0", "1", "3", "5", "7")
ANCHOR_SCORES = {-7, -5, -3, -1, 0, 1, 3, 5, 7}
NOT_RELEVANT_LABEL = "NOT RELEVANT"

RAPID_REVIEW_COLUMNS = [
    "Rapid Review Sentiment Outcome",
    "Rapid Review Sentiment Score",
    "Rapid Review Sentiment Confidence",
    "Rapid Review Sentiment Rationale",
    "Rapid Review 3-Way Sentiment",
    "Rapid Review 5-Way Sentiment",
    "Rapid Review Best Tag",
    "Rapid Review Tags",
    "Rapid Review Tag Confidence",
    "Rapid Review Tag Rationale",
    "Rapid Review Status",
    "Rapid Review Priority Tier",
    "Rapid Review Priority Reason",
    "Rapid Sentiment 3-Way Agreement",
    "Rapid Sentiment 5-Way Agreement",
    "Rapid Sentiment Score Distance",
    "Rapid Tag Best-Fit Agreement",
    "Rapid Tag Set Agreement",
    "Rapid Review Error",
    "Rapid Review Model",
    "Rapid Review Input Tokens",
    "Rapid Review Output Tokens",
    "Rapid Review Cost USD",
    "Rapid Review Raw Response",
]


def get_openai_api_key(secrets: Any | None = None) -> str:
    if secrets is not None:
        try:
            value = secrets.get("key", "")
        except Exception:
            value = ""
        if str(value or "").strip():
            return str(value).strip()
        try:
            value = secrets["key"]
        except Exception:
            value = ""
        if str(value or "").strip():
            return str(value).strip()
    return str(os.environ.get("OPENAI_API_KEY", "") or "").strip()


def ensure_rapid_review_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame() if df is None else df.copy()
    for column in RAPID_REVIEW_COLUMNS:
        if column not in out.columns:
            out[column] = pd.NA
    return out


def clear_rapid_review_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame() if df is None else df.copy()
    for column in RAPID_REVIEW_COLUMNS:
        if column in out.columns:
            out[column] = pd.NA
    return out


def resolve_rapid_review_batch_size(
    stored_batch_size: Any,
    remaining_eligible_count: int,
    *,
    default: int = RAPID_REVIEW_DEFAULT_BATCH_SIZE,
) -> int:
    """Return the practical execution chunk size for the remaining review pool."""
    remaining = max(0, int(remaining_eligible_count or 0))
    if remaining == 0:
        return 0
    try:
        selected = int(stored_batch_size)
    except (TypeError, ValueError):
        selected = default
    if selected < 1:
        selected = default
    return min(selected, remaining)


def _clean_label(value: Any) -> str:
    text = str(value or "").strip().upper()
    return " ".join(text.replace("_", " ").split())


def _safe_text(value: Any) -> str:
    try:
        missing = pd.isna(value)
    except Exception:
        missing = False
    if isinstance(missing, bool) and missing:
        return ""
    return str(value or "").strip()


def _to_float(value: Any) -> float | None:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return None
    return float(numeric)


def _to_probability(value: Any) -> float | None:
    numeric = _to_float(value)
    if numeric is None:
        return None
    if numeric > 1.0:
        numeric = numeric / 100.0
    return max(0.0, min(1.0, float(numeric)))


def derive_three_way_from_score(score: Any, *, not_relevant: bool = False) -> str:
    if not_relevant:
        return NOT_RELEVANT_LABEL
    numeric = _to_float(score)
    if numeric is None:
        return ""
    if numeric <= -2:
        return "NEGATIVE"
    if numeric >= 2:
        return "POSITIVE"
    return "NEUTRAL"


def derive_five_way_from_score(score: Any, *, not_relevant: bool = False) -> str:
    if not_relevant:
        return NOT_RELEVANT_LABEL
    numeric = _to_float(score)
    if numeric is None:
        return ""
    if numeric <= -4:
        return "VERY NEGATIVE"
    if numeric <= -2:
        return "SOMEWHAT NEGATIVE"
    if numeric < 2:
        return "NEUTRAL"
    if numeric < 4:
        return "SOMEWHAT POSITIVE"
    return "VERY POSITIVE"


def normalize_sentiment_outcome(value: Any) -> tuple[str, float | None]:
    raw = str(value or "").strip().upper().replace(" ", "_")
    if raw in {"NOT_RELEVANT", "NOT-RELEVANT", "NOT_RELEVANT."}:
        return "NOT_RELEVANT", None
    raw = raw.replace("+", "")
    numeric = _to_float(raw)
    if numeric is None or int(numeric) != numeric or int(numeric) not in ANCHOR_SCORES:
        raise ValueError(f"Invalid Rapid review sentiment outcome: {value or 'missing'}")
    score = int(numeric)
    return str(score), float(score)


def normalize_first_pass_sentiment(row: pd.Series | dict[str, Any]) -> dict[str, Any]:
    series = pd.Series(row)
    three_way = _clean_label(series.get("Jev Sentiment", ""))
    five_way = _clean_label(series.get("Jev 5-Way Sentiment", ""))
    raw_score = _to_float(series.get("Jev Sentiment Score"))
    relevance_probability = _to_probability(series.get("Jev Score Relevant Probability"))

    three_nr = three_way == NOT_RELEVANT_LABEL
    five_nr = five_way == NOT_RELEVANT_LABEL
    contradiction = bool(three_nr != five_nr)

    if three_nr and five_nr:
        outcome = "NOT_RELEVANT"
        comparable_score = None
        derived_three = NOT_RELEVANT_LABEL
        derived_five = NOT_RELEVANT_LABEL
    elif not three_nr and not five_nr and raw_score is not None:
        outcome = "SCORE"
        comparable_score = raw_score
        derived_three = derive_three_way_from_score(raw_score)
        derived_five = derive_five_way_from_score(raw_score)
    else:
        outcome = "CONTRADICTION"
        comparable_score = raw_score
        derived_three = derive_three_way_from_score(raw_score) if raw_score is not None else ""
        derived_five = derive_five_way_from_score(raw_score) if raw_score is not None else ""

    return {
        "outcome": outcome,
        "score": comparable_score,
        "raw_score": raw_score,
        "raw_3_way": three_way,
        "raw_5_way": five_way,
        "derived_3_way": derived_three,
        "derived_5_way": derived_five,
        "relevance_probability": relevance_probability,
        "internal_contradiction": contradiction,
    }


def _json_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if str(item).strip()]
        except Exception:
            pass
        return [part.strip() for part in stripped.replace(";", ",").split(",") if part.strip()]
    return []


def _first_pass_independent_yes_tags(row: pd.Series, tag_definitions: dict[str, str] | None) -> list[str]:
    tags = []
    for item in build_jev_tag_question_map(tag_definitions):
        tag = item["tag"]
        if _clean_label(row.get(f"Jev Tag [{tag}]", "")) == "YES":
            tags.append(tag)
    return tags


def _normalize_tag(value: Any, allowed_tags: dict[str, str], *, allow_other: bool = True) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    for tag in allowed_tags:
        if text.casefold() == tag.casefold():
            return tag
    if allow_other and text.casefold() == RESERVED_OTHER_TAG.casefold():
        return RESERVED_OTHER_TAG
    raise ValueError(f"Invalid Rapid review tag: {text}")


def normalize_applicable_tags(value: Any, allowed_tags: dict[str, str]) -> list[str]:
    explicit_allowed = {
        tag: rationale
        for tag, rationale in allowed_tags.items()
        if tag.casefold() != RESERVED_OTHER_TAG.casefold()
    }
    raw_tags = _json_list(value)
    normalized = []
    for raw_tag in raw_tags:
        tag = _normalize_tag(raw_tag, allowed_tags, allow_other=True)
        if tag and tag not in normalized:
            normalized.append(tag)
    explicit = [tag for tag in normalized if tag.casefold() != RESERVED_OTHER_TAG.casefold()]
    if explicit:
        return normalize_tag_assignment(explicit, explicit_allowed)
    return [RESERVED_OTHER_TAG]


def _probabilities_for_prefix(row: pd.Series, prefix: str) -> list[float]:
    values = []
    for column in row.index:
        if str(column).startswith(prefix):
            prob = _to_probability(row.get(column))
            if prob is not None:
                values.append(prob)
    return values


def _close_probability_competition(probabilities: list[float]) -> bool:
    probs = sorted([p for p in probabilities if p is not None], reverse=True)
    return len(probs) >= 2 and (probs[0] - probs[1]) <= RAPID_REVIEW_CLOSE_MARGIN


def _near_boundary(score: float | None) -> bool:
    if score is None:
        return False
    for boundary in (-4, -2, 2, 4):
        if abs(score - boundary) <= 0.35:
            return True
    return False


def _has_successful_first_pass(row: pd.Series, tag_definitions: dict[str, str] | None) -> bool:
    if _safe_text(row.get("Jev Error", "")):
        return False
    required = ["Jev Sentiment", "Jev 5-Way Sentiment", "Jev Sentiment Score", "Jev Mixture Label"]
    if any(not _safe_text(row.get(column, "")) for column in required):
        return False
    if build_jev_tag_question_map(tag_definitions):
        return bool(_safe_text(row.get("Jev Best Tag", "")) and _safe_text(row.get("Jev Tags", "")))
    return True


def _impact_thresholds(df: pd.DataFrame) -> dict[str, float]:
    thresholds = {}
    for column in ["Group Count", "Effective Reach", "Mentions", "Impressions"]:
        if column not in df.columns:
            thresholds[column] = float("inf")
            continue
        numeric = pd.to_numeric(df[column], errors="coerce").fillna(0)
        if numeric.empty or numeric.max() <= 0:
            thresholds[column] = float("inf")
        else:
            thresholds[column] = float(numeric.quantile(0.75))
    return thresholds


def _impact_reasons(row: pd.Series, thresholds: dict[str, float]) -> list[str]:
    reasons = []
    labels = {
        "Group Count": "high Group Count",
        "Effective Reach": "high Effective Reach",
        "Mentions": "high Mentions",
        "Impressions": "high Impressions",
    }
    for column, label in labels.items():
        value = _to_float(row.get(column)) or 0.0
        threshold = thresholds.get(column, float("inf"))
        if threshold != float("inf") and value > 0 and value > threshold:
            reasons.append(label)
    return reasons


def _is_negative_label(label: str) -> bool:
    return label in {"NEGATIVE", "SOMEWHAT NEGATIVE", "VERY NEGATIVE"}


def compute_rapid_review_candidate_reasons(
    row: pd.Series | dict[str, Any],
    *,
    tag_definitions: dict[str, str] | None = None,
    impact_thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    series = pd.Series(row)
    tag_definitions = tag_definitions or {}
    impact_thresholds = impact_thresholds or {}
    first_pass = normalize_first_pass_sentiment(series)

    contradictions: list[str] = []
    impact = _impact_reasons(series, impact_thresholds)
    uncertainty: list[str] = []
    contextual: list[str] = []
    malformed: list[str] = []

    if first_pass["internal_contradiction"]:
        contradictions.append("one categorical sentiment says NOT RELEVANT and the other does not")

    if first_pass["outcome"] == "CONTRADICTION":
        contradictions.append("first-pass relevance cannot be normalized cleanly")

    raw_three = first_pass["raw_3_way"]
    raw_five = first_pass["raw_5_way"]
    derived_three = first_pass["derived_3_way"]
    derived_five = first_pass["derived_5_way"]
    if raw_three and derived_three and raw_three != derived_three:
        contradictions.append("raw 3-way conflicts with score-derived 3-way")
    if raw_five and derived_five and raw_five != derived_five:
        contradictions.append("raw 5-way conflicts with score-derived 5-way")

    relevance_probability = first_pass["relevance_probability"]
    if relevance_probability is not None:
        categorical_relevant = first_pass["outcome"] == "SCORE"
        categorical_not_relevant = first_pass["outcome"] == "NOT_RELEVANT"
        if categorical_relevant and relevance_probability <= RAPID_REVIEW_RELEVANCE_WEAK:
            contradictions.append("independent relevance evidence strongly conflicts with categorical relevance")
        elif categorical_not_relevant and relevance_probability >= RAPID_REVIEW_RELEVANCE_STRONG:
            contradictions.append("independent relevance evidence strongly conflicts with categorical NOT RELEVANT")
        elif RAPID_REVIEW_RELEVANCE_WEAK < relevance_probability < RAPID_REVIEW_RELEVANCE_STRONG:
            uncertainty.append("questionable relevance probability")

    if _safe_text(series.get("AI Entity Match Conflict", "")).upper() == "YES" and (
        first_pass["internal_contradiction"]
        or (relevance_probability is not None and RAPID_REVIEW_RELEVANCE_WEAK < relevance_probability < RAPID_REVIEW_RELEVANCE_STRONG)
    ):
        contradictions.append("entity lexical conflict plus uncertain semantic relevance")

    if not raw_three or not raw_five or first_pass["raw_score"] is None:
        malformed.append("missing or malformed Rapid sentiment output")

    three_probs = _probabilities_for_prefix(series, "Jev Probability ")
    five_probs = _probabilities_for_prefix(series, "Jev 5-Way Probability ")
    selected_probs = [
        _to_probability(series.get("Jev Selected Probability")),
        _to_probability(series.get("Jev 5-Way Selected Probability")),
    ]
    if any(prob is not None and prob < RAPID_REVIEW_PROBABILITY_LOW for prob in selected_probs):
        uncertainty.append("low selected sentiment probability")
    if _close_probability_competition(three_probs) or _close_probability_competition(five_probs):
        uncertainty.append("close competing sentiment probabilities")
    if _near_boundary(first_pass["raw_score"]):
        uncertainty.append("fractional score near a derived sentiment boundary")

    mixture_score = _to_float(series.get("Jev Mixture Score"))
    if mixture_score is not None and mixture_score >= RAPID_REVIEW_HIGH_MIXTURE:
        uncertainty.append("high sentiment mixture")

    tag_map = build_jev_tag_question_map(tag_definitions)
    yes_tags = _first_pass_independent_yes_tags(series, tag_definitions)
    best_tag = _safe_text(series.get("Jev Best Tag", ""))
    if tag_map:
        best_prob = _to_probability(series.get("Jev Best Tag Selected Probability"))
        if best_prob is not None and best_prob < RAPID_REVIEW_PROBABILITY_LOW:
            uncertainty.append("low best-tag probability")
        if best_tag and best_tag.casefold() != RESERVED_OTHER_TAG.casefold() and best_tag not in yes_tags:
            contradictions.append("best-fit tag is not among independent YES tags")
        if len(yes_tags) == 0 and best_tag and best_tag.casefold() != RESERVED_OTHER_TAG.casefold():
            uncertainty.append("zero explicit YES tags when best-fit is not Other")
        if len(yes_tags) >= RAPID_REVIEW_MANY_TAGS:
            uncertainty.append("unusually many independent YES tags")
    elif any(_safe_text(series.get(column, "")) for column in ["Jev Best Tag", "Jev Tags"]):
        malformed.append("tag output exists without configured tag definitions")

    comparable_label = derived_three or raw_three
    negative = _is_negative_label(comparable_label) or _is_negative_label(derived_five)
    if negative and impact:
        contextual.append("negative + high impact")
    if negative and any("low" in reason for reason in uncertainty):
        contextual.append("negative + low confidence")
    if negative and contradictions:
        contextual.append("negative + sentiment contradiction")
    if negative and any("relevance" in reason for reason in uncertainty + contradictions):
        contextual.append("negative + questionable relevance")

    contradiction_reasons = list(dict.fromkeys(malformed + contradictions))
    uncertainty_reasons = list(dict.fromkeys(uncertainty))
    impact_reasons = list(dict.fromkeys(impact))
    contextual_reasons = list(dict.fromkeys(contextual))

    if contradiction_reasons:
        tier = 1
        tier_name = "Tier 1 - Contradictions / malformed outputs"
    elif impact_reasons:
        tier = 2
        tier_name = "Tier 2 - Impact"
    elif uncertainty_reasons:
        tier = 3
        tier_name = "Tier 3 - Uncertainty"
    elif contextual_reasons:
        tier = 4
        tier_name = "Tier 4 - Contextual combination"
    else:
        tier = 5
        tier_name = "Tier 5 - Routine"

    ordered_reasons = contradiction_reasons + impact_reasons + uncertainty_reasons + contextual_reasons
    return {
        "tier": tier,
        "tier_name": tier_name,
        "reason": "; ".join(ordered_reasons) if ordered_reasons else "routine processed story",
        "contradiction_count": len(contradiction_reasons),
        "impact_count": len(impact_reasons),
        "uncertainty_count": len(uncertainty_reasons),
        "contextual_count": len(contextual_reasons),
        "first_pass": first_pass,
    }


def build_rapid_review_candidates(
    df_unique: pd.DataFrame,
    *,
    tag_definitions: dict[str, str] | None = None,
) -> pd.DataFrame:
    if df_unique is None or df_unique.empty:
        return pd.DataFrame()

    working = ensure_rapid_review_columns(df_unique)
    processed_mask = working.apply(lambda row: _has_successful_first_pass(row, tag_definitions), axis=1)
    status = working["Rapid Review Status"].fillna("").astype(str).str.strip().str.upper()
    completed_mask = status.eq("COMPLETED")
    pool = working[processed_mask & ~completed_mask].copy()
    if pool.empty:
        return pool

    thresholds = _impact_thresholds(pool)
    reason_rows = []
    for idx, row in pool.iterrows():
        reasons = compute_rapid_review_candidate_reasons(
            row,
            tag_definitions=tag_definitions,
            impact_thresholds=thresholds,
        )
        reason_rows.append((idx, reasons))
        pool.loc[idx, "Rapid Review Priority Tier"] = reasons["tier_name"]
        pool.loc[idx, "Rapid Review Priority Reason"] = reasons["reason"]
        pool.loc[idx, "_rapid_review_tier_num"] = reasons["tier"]
        pool.loc[idx, "_rapid_review_contradictions"] = reasons["contradiction_count"]
        pool.loc[idx, "_rapid_review_impact_count"] = reasons["impact_count"]
        pool.loc[idx, "_rapid_review_uncertainty_count"] = reasons["uncertainty_count"]
        pool.loc[idx, "_rapid_review_contextual_count"] = reasons["contextual_count"]

    for column in ["Group Count", "Effective Reach", "Mentions", "Impressions"]:
        pool[column] = pd.to_numeric(pool.get(column, 0), errors="coerce").fillna(0)
    pool["_rapid_review_original_index"] = pool.index
    pool = pool.sort_values(
        [
            "_rapid_review_tier_num",
            "_rapid_review_contradictions",
            "Group Count",
            "Effective Reach",
            "Mentions",
            "Impressions",
            "_rapid_review_uncertainty_count",
            "_rapid_review_original_index",
        ],
        ascending=[True, False, False, False, False, False, False, True],
    ).reset_index(drop=False)
    return pool


def build_rapid_review_schema(tag_definitions: dict[str, str] | None = None) -> list[dict[str, Any]]:
    explicit_tags = [item["tag"] for item in build_jev_tag_question_map(tag_definitions)]
    allowed_tags = [*explicit_tags, RESERVED_OTHER_TAG]
    properties: dict[str, Any] = {
        "sentiment_outcome": {"type": "string", "enum": list(ANCHOR_OUTCOMES)},
        "sentiment_confidence": {"type": "integer", "minimum": 0, "maximum": 100},
        "sentiment_rationale": {"type": "string"},
    }
    required = ["sentiment_outcome", "sentiment_confidence", "sentiment_rationale"]
    if explicit_tags:
        properties.update(
            {
                "best_tag": {"type": "string", "enum": allowed_tags},
                "applicable_tags": {
                    "type": "array",
                    "items": {"type": "string", "enum": allowed_tags},
                },
                "tag_confidence": {"type": "integer", "minimum": 0, "maximum": 100},
                "tag_rationale": {"type": "string"},
            }
        )
        required.extend(["best_tag", "applicable_tags", "tag_confidence", "tag_rationale"])
    return [
        {
            "name": "rapid_second_opinion",
            "description": "Return a Rapid Labeling second opinion for one grouped story.",
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }
    ]


def build_rapid_review_prompt(
    row: pd.Series | dict[str, Any],
    analysis_payload: dict[str, Any],
    *,
    tag_definitions: dict[str, str] | None = None,
) -> str:
    series = pd.Series(row)
    explicit_tags = {
        item["tag"]: item["rationale"]
        for item in build_jev_tag_question_map(tag_definitions)
    }
    story_text = (
        _safe_text(series.get("Full Text", ""))
        or _safe_text(series.get("Article Text", ""))
        or _safe_text(series.get("Example Snippet", ""))
        or _safe_text(series.get("Snippet", ""))
    )
    context = {
        "primary_entity": analysis_payload.get("primary_name", ""),
        "aliases": analysis_payload.get("alternate_names", []),
        "spokespeople": analysis_payload.get("spokespeople", []),
        "products_subbrands_programs": analysis_payload.get("products", []),
        "general_guidance": analysis_payload.get("general_guidance", ""),
        "sentiment_guidance": analysis_payload.get("sentiment_guidance", ""),
    }
    sentiment_anchors = {
        "NOT_RELEVANT": "The monitored collective entity is not meaningfully present.",
        "-7": "Extremely negative toward the collective entity.",
        "-5": "Strongly negative toward the collective entity.",
        "-3": "Moderately negative toward the collective entity.",
        "-1": "Slightly negative, but still near neutral for category purposes.",
        "0": "Neutral, factual, balanced, incidental, or no directional takeaway.",
        "1": "Slightly positive, but still near neutral for category purposes.",
        "3": "Moderately positive toward the collective entity.",
        "5": "Strongly positive toward the collective entity.",
        "7": "Extremely positive toward the collective entity.",
    }
    parts = [
        "You are providing a second opinion for a media-analysis labeling workflow.",
        "Judge the monitored collective entity, not the broader topic.",
        "The monitored collective entity includes the primary entity, aliases, listed spokespeople acting for the entity, and configured products, sub-brands, or programs.",
        "The parent organization does not need to be explicitly named for the story to be relevant; substantive coverage of any configured member of the collective is in scope.",
        "A passing or incidental mention of the collective entity may be neutral, but should not become NOT_RELEVANT merely because it is brief.",
        "Use NOT_RELEVANT only when the configured client collective is genuinely absent, mismatched, or not actually the subject of the apparent match.",
        "Negative subject matter alone is not negative sentiment toward the entity.",
        "Ordinary factual/background context does not make a story mixed or negative.",
        "Do not treat an entity term as a true entity mention when it appears only as part of a different organization, product, or proper name, unless the story clearly refers to the monitored collective entity.",
        "",
        "Entity and analysis context:",
        json.dumps(context, ensure_ascii=False, indent=2),
        "",
        "Sentiment outcome anchors:",
        json.dumps(sentiment_anchors, ensure_ascii=False, indent=2),
    ]
    if explicit_tags:
        parts.extend(
            [
                "",
                "Rapid tag definitions. Choose one best_tag from these explicit tags or Other. For applicable_tags, include every explicit tag that substantively applies; use Other only when no explicit tag applies, and never with explicit tags.",
                json.dumps({**explicit_tags, RESERVED_OTHER_TAG: "Use only when no explicit tag applies."}, ensure_ascii=False, indent=2),
            ]
        )
    parts.extend(
        [
            "",
            "Story:",
            f"Headline: {_safe_text(series.get('Headline', ''))}",
            f"Outlet: {_safe_text(series.get('Outlet', ''))}",
            f"Type: {_safe_text(series.get('Type', ''))}",
            f"Text: {story_text}",
        ]
    )
    return "\n".join(parts)


def parse_rapid_review_response(
    args: dict[str, Any],
    *,
    tag_definitions: dict[str, str] | None = None,
    tagging_mode: str = DEFAULT_JEV_TAGGING_MODE,
) -> dict[str, Any]:
    outcome, score = normalize_sentiment_outcome(args.get("sentiment_outcome"))
    confidence = _to_probability(args.get("sentiment_confidence"))
    if confidence is None:
        raise ValueError("Rapid review sentiment confidence is missing.")
    sentiment_rationale = _safe_text(args.get("sentiment_rationale", ""))
    if not sentiment_rationale:
        raise ValueError("Rapid review sentiment rationale is missing.")

    parsed = {
        "sentiment_outcome": outcome,
        "sentiment_score": score,
        "sentiment_confidence": confidence,
        "sentiment_rationale": sentiment_rationale,
        "review_3_way": derive_three_way_from_score(score, not_relevant=outcome == "NOT_RELEVANT"),
        "review_5_way": derive_five_way_from_score(score, not_relevant=outcome == "NOT_RELEVANT"),
    }

    if build_jev_tag_question_map(tag_definitions):
        allowed = {item["tag"]: item["rationale"] for item in build_jev_tag_question_map(tag_definitions)}
        allowed[RESERVED_OTHER_TAG] = "Fallback"
        best_tag = _normalize_tag(args.get("best_tag"), allowed, allow_other=True)
        applicable_tags = normalize_applicable_tags(args.get("applicable_tags", []), allowed)
        tag_confidence = _to_probability(args.get("tag_confidence"))
        if tag_confidence is None:
            raise ValueError("Rapid review tag confidence is missing.")
        tag_rationale = _safe_text(args.get("tag_rationale", ""))
        if not tag_rationale:
            raise ValueError("Rapid review tag rationale is missing.")
        mode = normalize_jev_tagging_mode(tagging_mode)
        if mode == "Single best tag":
            official_tags = normalize_tag_assignment(best_tag, allowed)
        else:
            official_tags = applicable_tags
        parsed.update(
            {
                "best_tag": best_tag,
                "applicable_tags": applicable_tags,
                "official_tags": official_tags,
                "tag_confidence": tag_confidence,
                "tag_rationale": tag_rationale,
            }
        )
    return parsed


def call_rapid_review_luna(
    prompt: str,
    api_key: str,
    *,
    model: str = RAPID_REVIEW_MODEL,
    tag_definitions: dict[str, str] | None = None,
) -> tuple[dict[str, Any], int, int, str]:
    client = OpenAI(api_key=api_key)
    functions = build_rapid_review_schema(tag_definitions)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a careful media-labeling second-opinion analyst."},
            {"role": "user", "content": prompt},
        ],
        tools=[{"type": "function", "function": function} for function in functions],
        tool_choice={"type": "function", "function": {"name": "rapid_second_opinion"}},
        reasoning_effort=CHAT_COMPLETIONS_TOOL_REASONING_EFFORT,
    )
    message = response.choices[0].message
    tool_calls = getattr(message, "tool_calls", None) or []
    args_text = tool_calls[0].function.arguments if tool_calls else None
    if not args_text:
        function_call = getattr(message, "function_call", None)
        args_text = getattr(function_call, "arguments", None) if function_call else None
    if not args_text:
        raise ValueError("Rapid review model returned no function-call result.")
    args = json.loads(args_text)
    in_tokens, out_tokens = extract_usage_tokens(response)
    return args, in_tokens, out_tokens, json.dumps(args, ensure_ascii=False, sort_keys=True)


def _run_rapid_review_row(
    row_dict: dict[str, Any],
    analysis_payload: dict[str, Any],
    api_key: str,
    model: str,
    tag_definitions: dict[str, str] | None,
    tagging_mode: str,
    call_fn: Callable[..., tuple[dict[str, Any], int, int, str]],
) -> dict[str, Any]:
    row = pd.Series(row_dict)
    original_index = int(row.get("index", row.get("_rapid_review_original_index", 0)))
    prompt = build_rapid_review_prompt(row, analysis_payload, tag_definitions=tag_definitions)
    last_error = ""
    for attempt in range(RAPID_REVIEW_MAX_RETRIES + 1):
        try:
            args, in_tokens, out_tokens, raw_response = call_fn(
                prompt,
                api_key,
                model=model,
                tag_definitions=tag_definitions,
            )
            parsed = parse_rapid_review_response(
                args,
                tag_definitions=tag_definitions,
                tagging_mode=tagging_mode,
            )
            return {
                "original_index": original_index,
                "group_id": row.get("Group ID", original_index),
                "parsed": parsed,
                "input_tokens": int(in_tokens or 0),
                "output_tokens": int(out_tokens or 0),
                "cost_usd": estimate_cost_usd(in_tokens, out_tokens, model),
                "raw_response": raw_response,
                "model": model,
                "error": "",
            }
        except Exception as exc:
            last_error = str(exc)
            if attempt < RAPID_REVIEW_MAX_RETRIES:
                time.sleep(min(2 ** attempt, 4))
                continue
    return {
        "original_index": original_index,
        "group_id": row.get("Group ID", original_index),
        "parsed": {},
        "input_tokens": 0,
        "output_tokens": 0,
        "cost_usd": 0.0,
        "raw_response": "",
        "model": model,
        "error": last_error or "Rapid review failed.",
    }


def apply_rapid_review_result_to_df(
    df_unique: pd.DataFrame,
    original_index: int,
    row_result: dict[str, Any],
    *,
    tag_definitions: dict[str, str] | None = None,
) -> pd.DataFrame:
    out = ensure_rapid_review_columns(df_unique)
    parsed = row_result.get("parsed", {}) if isinstance(row_result.get("parsed"), dict) else {}
    error = _safe_text(row_result.get("error", ""))
    if error:
        out.loc[original_index, "Rapid Review Status"] = "Error"
        out.loc[original_index, "Rapid Review Error"] = error
        out.loc[original_index, "Rapid Review Model"] = row_result.get("model")
        return out

    row = out.loc[original_index]
    first_pass = normalize_first_pass_sentiment(row)
    review_score = parsed.get("sentiment_score")
    review_not_relevant = parsed.get("sentiment_outcome") == "NOT_RELEVANT"
    review_three = parsed.get("review_3_way")
    review_five = parsed.get("review_5_way")

    first_three = first_pass["derived_3_way"]
    first_five = first_pass["derived_5_way"]
    if first_pass["outcome"] == "NOT_RELEVANT":
        first_three = NOT_RELEVANT_LABEL
        first_five = NOT_RELEVANT_LABEL
    score_distance = pd.NA
    if first_pass["score"] is not None and review_score is not None:
        score_distance = abs(float(first_pass["score"]) - float(review_score))

    out.loc[original_index, "Rapid Review Sentiment Outcome"] = parsed.get("sentiment_outcome")
    out.loc[original_index, "Rapid Review Sentiment Score"] = pd.NA if review_not_relevant else review_score
    out.loc[original_index, "Rapid Review Sentiment Confidence"] = parsed.get("sentiment_confidence")
    out.loc[original_index, "Rapid Review Sentiment Rationale"] = parsed.get("sentiment_rationale")
    out.loc[original_index, "Rapid Review 3-Way Sentiment"] = review_three
    out.loc[original_index, "Rapid Review 5-Way Sentiment"] = review_five
    out.loc[original_index, "Rapid Sentiment 3-Way Agreement"] = "Match" if first_three and first_three == review_three else "Disagree"
    out.loc[original_index, "Rapid Sentiment 5-Way Agreement"] = "Match" if first_five and first_five == review_five else "Disagree"
    out.loc[original_index, "Rapid Sentiment Score Distance"] = score_distance

    if build_jev_tag_question_map(tag_definitions):
        first_best = _safe_text(row.get("Jev Best Tag", ""))
        first_tags = set(normalize_tag_assignment(_json_list(row.get("Jev Tags", "")), {item["tag"]: item["rationale"] for item in build_jev_tag_question_map(tag_definitions)}))
        review_tags = parsed.get("official_tags", [])
        out.loc[original_index, "Rapid Review Best Tag"] = parsed.get("best_tag")
        out.loc[original_index, "Rapid Review Tags"] = "; ".join(review_tags)
        out.loc[original_index, "Rapid Review Tag Confidence"] = parsed.get("tag_confidence")
        out.loc[original_index, "Rapid Review Tag Rationale"] = parsed.get("tag_rationale")
        out.loc[original_index, "Rapid Tag Best-Fit Agreement"] = "Match" if first_best.casefold() == str(parsed.get("best_tag", "")).casefold() else "Disagree"
        out.loc[original_index, "Rapid Tag Set Agreement"] = "Match" if first_tags == set(review_tags) else "Disagree"

    out.loc[original_index, "Rapid Review Status"] = "Completed"
    out.loc[original_index, "Rapid Review Error"] = pd.NA
    out.loc[original_index, "Rapid Review Model"] = row_result.get("model")
    out.loc[original_index, "Rapid Review Input Tokens"] = row_result.get("input_tokens")
    out.loc[original_index, "Rapid Review Output Tokens"] = row_result.get("output_tokens")
    out.loc[original_index, "Rapid Review Cost USD"] = row_result.get("cost_usd")
    out.loc[original_index, "Rapid Review Raw Response"] = row_result.get("raw_response")
    return out


def run_rapid_review_batch(
    df_unique: pd.DataFrame,
    candidates_df: pd.DataFrame,
    analysis_payload: dict[str, Any],
    api_key: str,
    *,
    model: str = RAPID_REVIEW_MODEL,
    tag_definitions: dict[str, str] | None = None,
    tagging_mode: str = DEFAULT_JEV_TAGGING_MODE,
    limit: int = 50,
    max_workers: int = RAPID_REVIEW_MAX_WORKERS,
    call_fn: Callable[..., tuple[dict[str, Any], int, int, str]] = call_rapid_review_luna,
    progress_callback: Callable[[int, int], None] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    updated = ensure_rapid_review_columns(df_unique)
    working = candidates_df.head(max(0, int(limit or 0))).copy()
    if working.empty:
        return updated, {"done": 0, "successful": 0, "errors": [], "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}

    rows = [row.to_dict() for _, row in working.iterrows()]
    worker_count = max(1, min(int(max_workers or 1), len(rows)))
    errors: list[str] = []
    successful = 0
    total_in = 0
    total_out = 0
    total_cost = 0.0
    finished = 0

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(
                _run_rapid_review_row,
                row_dict,
                analysis_payload,
                api_key,
                model,
                tag_definitions,
                normalize_jev_tagging_mode(tagging_mode),
                call_fn,
            ): int(row_dict.get("index", row_dict.get("_rapid_review_original_index", fallback_index)))
            for fallback_index, row_dict in enumerate(rows)
        }
        for future in as_completed(futures):
            original_index = futures[future]
            try:
                row_result = future.result()
                original_index = int(row_result.get("original_index", original_index))
                updated = apply_rapid_review_result_to_df(
                    updated,
                    original_index,
                    row_result,
                    tag_definitions=tag_definitions,
                )
                if row_result.get("error"):
                    gid = row_result.get("group_id", original_index)
                    errors.append(f"Group {gid}: {row_result['error']}")
                else:
                    successful += 1
                    total_in += int(row_result.get("input_tokens", 0) or 0)
                    total_out += int(row_result.get("output_tokens", 0) or 0)
                    total_cost += float(row_result.get("cost_usd", 0.0) or 0.0)
            except Exception as exc:
                errors.append(f"Story {original_index + 1}: {exc}")
            finally:
                finished += 1
                if progress_callback is not None:
                    progress_callback(finished, len(rows))

    return updated, {
        "done": len(rows),
        "successful": successful,
        "errors": errors,
        "input_tokens": total_in,
        "output_tokens": total_out,
        "cost_usd": total_cost,
    }
