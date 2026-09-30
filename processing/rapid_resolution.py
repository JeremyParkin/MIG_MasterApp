from __future__ import annotations

from collections import Counter
from typing import Any

import pandas as pd

from processing.ai_tagging import RESERVED_OTHER_TAG, normalize_tag_assignment
from processing.jev_sentiment import build_jev_tag_question_map


STATUS_INVALID = "Invalid/incomplete"
STATUS_FIRST_PASS_ONLY = "First pass only"
STATUS_INTERNAL_CONFLICT = "First-pass internal conflict"
STATUS_AGREEMENT = "Cross-model agreement"
STATUS_PARTIAL_AGREEMENT = "Cross-model partial agreement"
STATUS_DISAGREEMENT = "Cross-model disagreement"
STATUS_SECOND_OPINION_ERROR = "Second opinion error"

SOURCE_RAPID_FIRST_PASS = "Rapid first pass"
SOURCE_HUMAN_ASSIGNED = "Human assigned"
SOURCE_HUMAN_ACCEPTED_MACHINE = "Human accepted machine"

RELEVANCE_RELEVANT = "RELEVANT"
RELEVANCE_NOT_RELEVANT = "NOT_RELEVANT"
RELEVANCE_CONFLICT = "CONFLICT"

HUMAN_STATE_ACCEPTED = "Accepted machine"
HUMAN_STATE_ASSIGNED = "Assigned"
HUMAN_SENTIMENT_SCALE_3_WAY = "3-way"
HUMAN_SENTIMENT_SCALE_5_WAY = "5-way"
RAPID_TAG_REVIEW_MODE_BEST = "Single best fit"
RAPID_TAG_REVIEW_MODE_APPLICABLE = "All applicable"
LEGACY_RAPID_TAGGING_MODE_BEST = "Single best tag"
LEGACY_RAPID_TAGGING_MODE_APPLICABLE = "Multiple applicable tags"

RAPID_HUMAN_REVIEW_COLUMNS = [
    "Rapid Human Sentiment Review State",
    "Rapid Human Sentiment Scale",
    "Rapid Human Sentiment",
    "Rapid Human Sentiment Source",
    "Rapid Accepted Machine Relevance",
    "Rapid Accepted Machine Sentiment 3-Way",
    "Rapid Accepted Machine Sentiment 5-Way",
    "Rapid Accepted Machine Sentiment Score",
    "Rapid Human Tag Review State",
    "Rapid Human Tag Assignment",
    "Rapid Human Tagging Mode",
    "Rapid Human Tag Source",
    "Rapid Human Best Tag Review State",
    "Rapid Human Best Tag Assignment",
    "Rapid Human Best Tag Source",
    "Rapid Human Applicable Tags Review State",
    "Rapid Human Applicable Tags Assignment",
    "Rapid Human Applicable Tags Source",
    "Rapid Accepted Machine Best Tag",
    "Rapid Accepted Machine Tags",
]

EFFECTIVE_SENTIMENT_COLUMNS = [
    "Effective Rapid Relevance",
    "Effective Rapid Sentiment 3-Way",
    "Effective Rapid Sentiment 5-Way",
    "Effective Rapid Sentiment Score",
    "Effective Rapid Sentiment Source",
    "Rapid Sentiment Resolution Status",
    "Rapid Sentiment Score Distance",
]

EFFECTIVE_TAG_COLUMNS = [
    "Effective Rapid Best Tag",
    "Effective Rapid Tags",
    "Effective Rapid Tag Source",
    "Rapid Tag Resolution Status",
]

FINAL_SENTIMENT_COLUMNS = [
    "Final Rapid Relevance",
    "Final Rapid Sentiment 3-Way",
    "Final Rapid Sentiment 5-Way",
    "Final Rapid Sentiment Score",
    "Final Rapid Sentiment Source",
]

FINAL_TAG_COLUMNS = [
    "Final Rapid Best Tag",
    "Final Rapid Tags",
    "Final Rapid Tag Source",
]


def _safe_text(value: Any) -> str:
    try:
        missing = pd.isna(value)
    except Exception:
        missing = False
    if isinstance(missing, bool) and missing:
        return ""
    if value is None:
        return ""
    return str(value).strip()


def _clean_label(value: Any) -> str:
    return " ".join(_safe_text(value).replace("_", " ").upper().split())


def _to_float(value: Any) -> float | None:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return None
    return float(numeric)


def format_rapid_numeric(value: Any, *, decimals: int = 2) -> str:
    numeric = _to_float(value)
    if numeric is None:
        return ""
    return f"{numeric:.{decimals}f}"


def _is_not_relevant(value: Any) -> bool:
    return _clean_label(value) == "NOT RELEVANT"


def ensure_rapid_human_review_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for column in RAPID_HUMAN_REVIEW_COLUMNS:
        if column not in out.columns:
            out[column] = pd.NA
    return out


def build_rapid_highlight_keywords(analysis_payload: dict[str, Any] | None) -> list[str]:
    payload = analysis_payload or {}
    raw_items: list[Any] = []
    raw_items.append(payload.get("primary_name", ""))
    raw_items.extend(payload.get("alternate_names", []) or [])
    raw_items.extend(payload.get("spokespeople", []) or [])
    raw_items.extend(payload.get("products", []) or [])
    raw_items.extend(payload.get("highlight_keywords", []) or [])

    seen: set[str] = set()
    out: list[str] = []
    for item in raw_items:
        text = _safe_text(item)
        key = text.casefold()
        if not text or key in seen:
            continue
        seen.add(key)
        out.append(text)
    return out


def rapid_sentiment_acceptance_available(row: pd.Series | dict[str, Any], scale: str) -> bool:
    series = pd.Series(row)
    column = (
        "Effective Rapid Sentiment 3-Way"
        if scale == HUMAN_SENTIMENT_SCALE_3_WAY
        else "Effective Rapid Sentiment 5-Way"
        if scale == HUMAN_SENTIMENT_SCALE_5_WAY
        else ""
    )
    if not column:
        return False
    return bool(_safe_text(series.get(column, "")))


def rapid_tag_acceptance_available(row: pd.Series | dict[str, Any], tagging_mode: str) -> bool:
    series = pd.Series(row)
    if tagging_mode in {RAPID_TAG_REVIEW_MODE_APPLICABLE, LEGACY_RAPID_TAGGING_MODE_APPLICABLE}:
        return bool(_safe_text(series.get("Effective Rapid Tags", "")))
    return bool(_safe_text(series.get("Effective Rapid Best Tag", "")))


def _is_applicable_tag_mode(mode: str) -> bool:
    return _safe_text(mode) in {RAPID_TAG_REVIEW_MODE_APPLICABLE, LEGACY_RAPID_TAGGING_MODE_APPLICABLE}


def apply_rapid_tag_human_selection(
    df: pd.DataFrame,
    group_id: int,
    *,
    assignment: Any,
    tagging_mode: str,
    tag_definitions: dict[str, str] | None = None,
) -> pd.DataFrame:
    out = ensure_rapid_human_review_columns(df)
    if out.empty or "Group ID" not in out.columns:
        return out
    mask = out["Group ID"] == group_id
    if not mask.any():
        return out

    effective = build_effective_rapid_tag_frame(out, tag_definitions=tag_definitions)
    idx = out.index[mask][0]
    selected_tags, selected_invalid = _normalize_tag_set(assignment, tag_definitions)
    if selected_invalid or not selected_tags:
        selected_tags = normalize_tag_assignment(assignment, _tag_definitions_from_config(tag_definitions))
    selected_assignment = "; ".join(selected_tags)

    machine_best = _safe_text(effective.loc[idx, "Effective Rapid Best Tag"])
    machine_tags, machine_invalid = _normalize_tag_set(effective.loc[idx, "Effective Rapid Tags"], tag_definitions)
    accepted = False
    applicable_mode = _is_applicable_tag_mode(tagging_mode)
    if applicable_mode:
        accepted = bool(machine_tags) and not machine_invalid and _tags_equal(selected_tags, machine_tags)
    else:
        selected_best = selected_tags[0] if selected_tags else ""
        accepted = bool(machine_best) and selected_best.casefold() == machine_best.casefold()

    if accepted:
        if applicable_mode:
            out.loc[mask, "Rapid Human Applicable Tags Review State"] = HUMAN_STATE_ACCEPTED
            out.loc[mask, "Rapid Human Applicable Tags Assignment"] = pd.NA
            out.loc[mask, "Rapid Human Applicable Tags Source"] = "HUMAN"
            out.loc[mask, "Rapid Accepted Machine Tags"] = effective.loc[idx, "Effective Rapid Tags"]
        else:
            out.loc[mask, "Rapid Human Best Tag Review State"] = HUMAN_STATE_ACCEPTED
            out.loc[mask, "Rapid Human Best Tag Assignment"] = pd.NA
            out.loc[mask, "Rapid Human Best Tag Source"] = "HUMAN"
            out.loc[mask, "Rapid Accepted Machine Best Tag"] = effective.loc[idx, "Effective Rapid Best Tag"]
        return out

    if applicable_mode:
        out.loc[mask, "Rapid Human Applicable Tags Review State"] = HUMAN_STATE_ASSIGNED
        out.loc[mask, "Rapid Human Applicable Tags Assignment"] = selected_assignment
        out.loc[mask, "Rapid Human Applicable Tags Source"] = "HUMAN"
        out.loc[mask, "Rapid Accepted Machine Tags"] = pd.NA
    else:
        out.loc[mask, "Rapid Human Best Tag Review State"] = HUMAN_STATE_ASSIGNED
        out.loc[mask, "Rapid Human Best Tag Assignment"] = selected_tags[0] if selected_tags else selected_assignment
        out.loc[mask, "Rapid Human Best Tag Source"] = "HUMAN"
        out.loc[mask, "Rapid Accepted Machine Best Tag"] = pd.NA
    return out


def apply_rapid_sentiment_human_selection(
    df: pd.DataFrame,
    group_id: int,
    *,
    scale: str,
    label: str,
) -> pd.DataFrame:
    out = ensure_rapid_human_review_columns(df)
    if out.empty or "Group ID" not in out.columns:
        return out
    mask = out["Group ID"] == group_id
    if not mask.any():
        return out

    effective = build_effective_rapid_sentiment_frame(out)
    idx = out.index[mask][0]
    active_column = (
        "Effective Rapid Sentiment 3-Way"
        if scale == HUMAN_SENTIMENT_SCALE_3_WAY
        else "Effective Rapid Sentiment 5-Way"
        if scale == HUMAN_SENTIMENT_SCALE_5_WAY
        else ""
    )
    active_machine_label = _clean_label(effective.loc[idx, active_column]) if active_column else ""
    selected_label = _clean_label(label)

    if active_machine_label and active_machine_label == selected_label:
        out.loc[mask, "Rapid Human Sentiment Review State"] = HUMAN_STATE_ACCEPTED
        out.loc[mask, "Rapid Human Sentiment Scale"] = pd.NA
        out.loc[mask, "Rapid Human Sentiment"] = pd.NA
        out.loc[mask, "Rapid Human Sentiment Source"] = "HUMAN"
        out.loc[mask, "Rapid Accepted Machine Relevance"] = effective.loc[idx, "Effective Rapid Relevance"]
        out.loc[mask, "Rapid Accepted Machine Sentiment 3-Way"] = effective.loc[idx, "Effective Rapid Sentiment 3-Way"]
        out.loc[mask, "Rapid Accepted Machine Sentiment 5-Way"] = effective.loc[idx, "Effective Rapid Sentiment 5-Way"]
        out.loc[mask, "Rapid Accepted Machine Sentiment Score"] = effective.loc[idx, "Effective Rapid Sentiment Score"]
        return out

    out.loc[mask, "Rapid Human Sentiment Review State"] = HUMAN_STATE_ASSIGNED
    out.loc[mask, "Rapid Human Sentiment Scale"] = scale
    out.loc[mask, "Rapid Human Sentiment"] = selected_label
    out.loc[mask, "Rapid Human Sentiment Source"] = "HUMAN"
    for column in [
        "Rapid Accepted Machine Relevance",
        "Rapid Accepted Machine Sentiment 3-Way",
        "Rapid Accepted Machine Sentiment 5-Way",
        "Rapid Accepted Machine Sentiment Score",
    ]:
        out.loc[mask, column] = pd.NA
    return out


def _review_has_completed(row: pd.Series) -> bool:
    return _safe_text(row.get("Rapid Review Status", "")).upper() == "COMPLETED"


def _review_has_error(row: pd.Series) -> bool:
    status = _safe_text(row.get("Rapid Review Status", "")).upper()
    error = _safe_text(row.get("Rapid Review Error", ""))
    return status == "ERROR" or bool(error)


def _sentiment_base_from_jev(row: pd.Series) -> dict[str, Any]:
    three = _clean_label(row.get("Jev Sentiment", ""))
    five = _clean_label(row.get("Jev 5-Way Sentiment", ""))
    if not three or not five:
        return {
            "valid": False,
            "relevance": "",
            "three_way": "",
            "five_way": "",
            "score": pd.NA,
            "source": "",
            "status": STATUS_INVALID,
        }

    three_nr = three == "NOT RELEVANT"
    five_nr = five == "NOT RELEVANT"
    if three_nr and five_nr:
        return {
            "valid": True,
            "relevance": RELEVANCE_NOT_RELEVANT,
            "three_way": "NOT RELEVANT",
            "five_way": "NOT RELEVANT",
            "score": pd.NA,
            "source": SOURCE_RAPID_FIRST_PASS,
            "status": STATUS_FIRST_PASS_ONLY,
        }
    if three_nr != five_nr:
        return {
            "valid": False,
            "relevance": RELEVANCE_CONFLICT,
            "three_way": "",
            "five_way": "",
            "score": pd.NA,
            "source": "",
            "status": STATUS_INTERNAL_CONFLICT,
        }

    score = _to_float(row.get("Jev Sentiment Score"))
    return {
        "valid": True,
        "relevance": RELEVANCE_RELEVANT,
        "three_way": three,
        "five_way": five,
        "score": score if score is not None else pd.NA,
        "source": SOURCE_RAPID_FIRST_PASS,
        "status": STATUS_FIRST_PASS_ONLY,
    }


def _review_sentiment_from_luna(row: pd.Series) -> dict[str, Any]:
    three = _clean_label(row.get("Rapid Review 3-Way Sentiment", ""))
    five = _clean_label(row.get("Rapid Review 5-Way Sentiment", ""))
    outcome = _safe_text(row.get("Rapid Review Sentiment Outcome", "")).upper()
    if not three or not five or not outcome:
        return {"valid": False}
    if outcome == "NOT_RELEVANT" or (three == "NOT RELEVANT" and five == "NOT RELEVANT"):
        return {
            "valid": True,
            "relevance": RELEVANCE_NOT_RELEVANT,
            "three_way": "NOT RELEVANT",
            "five_way": "NOT RELEVANT",
            "score": None,
        }
    if three == "NOT RELEVANT" or five == "NOT RELEVANT":
        return {"valid": False}
    return {
        "valid": True,
        "relevance": RELEVANCE_RELEVANT,
        "three_way": three,
        "five_way": five,
        "score": _to_float(row.get("Rapid Review Sentiment Score")),
    }


def resolve_rapid_sentiment_row(row: pd.Series | dict[str, Any]) -> dict[str, Any]:
    series = pd.Series(row)
    base = _sentiment_base_from_jev(series)
    result = {
        "Effective Rapid Relevance": base["relevance"],
        "Effective Rapid Sentiment 3-Way": base["three_way"],
        "Effective Rapid Sentiment 5-Way": base["five_way"],
        "Effective Rapid Sentiment Score": base["score"],
        "Effective Rapid Sentiment Source": base["source"],
        "Rapid Sentiment Resolution Status": base["status"],
        "Rapid Sentiment Score Distance": pd.NA,
    }

    if base["status"] in {STATUS_INVALID, STATUS_INTERNAL_CONFLICT}:
        return result

    if _review_has_error(series):
        result["Rapid Sentiment Resolution Status"] = STATUS_SECOND_OPINION_ERROR
        return result

    if not _review_has_completed(series):
        return result

    review = _review_sentiment_from_luna(series)
    if not review.get("valid"):
        result["Rapid Sentiment Resolution Status"] = STATUS_SECOND_OPINION_ERROR
        return result

    if base["relevance"] == RELEVANCE_RELEVANT and review["relevance"] == RELEVANCE_RELEVANT:
        base_score = _to_float(base["score"])
        review_score = _to_float(review.get("score"))
        if base_score is not None and review_score is not None:
            result["Rapid Sentiment Score Distance"] = abs(base_score - review_score)

    relevance_agrees = base["relevance"] == review["relevance"]
    three_agrees = _clean_label(base["three_way"]) == _clean_label(review["three_way"])
    five_agrees = _clean_label(base["five_way"]) == _clean_label(review["five_way"])

    if relevance_agrees and three_agrees and five_agrees:
        status = STATUS_AGREEMENT
    elif not relevance_agrees or (not three_agrees and not five_agrees):
        status = STATUS_DISAGREEMENT
    else:
        status = STATUS_PARTIAL_AGREEMENT
    result["Rapid Sentiment Resolution Status"] = status
    return result


def _tag_definitions_from_config(tag_definitions: dict[str, str] | None) -> dict[str, str]:
    definitions = {
        item["tag"]: item["rationale"]
        for item in build_jev_tag_question_map(tag_definitions)
    }
    definitions[RESERVED_OTHER_TAG] = "Fallback when no configured tag applies."
    return definitions


def _split_tag_list(value: Any) -> list[str]:
    if isinstance(value, list):
        raw_items = value
    else:
        try:
            missing = pd.isna(value)
        except Exception:
            missing = False
        if isinstance(missing, bool) and missing:
            raw_items = []
        elif value is None:
            raw_items = []
        else:
            raw_items = str(value).replace(";", ",").split(",")
    cleaned: list[str] = []
    seen: set[str] = set()
    for item in raw_items:
        try:
            item_missing = pd.isna(item)
        except Exception:
            item_missing = False
        if isinstance(item_missing, bool) and item_missing:
            continue
        if item is None:
            continue
        tag = str(item).strip()
        if not tag:
            continue
        key = tag.casefold()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(tag)
    return cleaned


def _normalize_tag_set(value: Any, tag_definitions: dict[str, str] | None) -> tuple[list[str], bool]:
    definitions = _tag_definitions_from_config(tag_definitions)
    if not definitions:
        return [], False
    raw_tags = _split_tag_list(value)
    invalid = False
    if not raw_tags:
        return [], invalid
    raw_has_other = any(tag.casefold() == RESERVED_OTHER_TAG.casefold() for tag in raw_tags)
    raw_has_explicit = any(tag.casefold() != RESERVED_OTHER_TAG.casefold() for tag in raw_tags)
    if raw_has_other and raw_has_explicit:
        invalid = True
    for tag in raw_tags:
        if tag.casefold() not in {defined.casefold() for defined in definitions}:
            invalid = True
    normalized = normalize_tag_assignment(raw_tags, definitions)
    has_other = any(tag.casefold() == RESERVED_OTHER_TAG.casefold() for tag in normalized)
    has_explicit = any(tag.casefold() != RESERVED_OTHER_TAG.casefold() for tag in normalized)
    if has_other and has_explicit:
        invalid = True
    return normalized, invalid


def _normalize_one_tag(value: Any, tag_definitions: dict[str, str] | None) -> tuple[str, bool]:
    text = _safe_text(value)
    if not text:
        return "", True
    definitions = _tag_definitions_from_config(tag_definitions)
    by_key = {tag.casefold(): tag for tag in definitions}
    tag = by_key.get(text.casefold())
    return (tag or text, tag is None)


def _tags_equal(left: list[str], right: list[str]) -> bool:
    return {tag.casefold() for tag in left} == {tag.casefold() for tag in right}


def _tag_base_from_jev(row: pd.Series, tag_definitions: dict[str, str] | None) -> dict[str, Any]:
    best_tag, best_invalid = _normalize_one_tag(row.get("Jev Best Tag", ""), tag_definitions)
    tags, tags_invalid = _normalize_tag_set(row.get("Jev Tags", ""), tag_definitions)
    if not best_tag or not tags:
        return {
            "valid": False,
            "best_tag": "",
            "tags": [],
            "source": "",
            "status": STATUS_INVALID,
        }

    best_is_other = best_tag.casefold() == RESERVED_OTHER_TAG.casefold()
    tags_have_other = any(tag.casefold() == RESERVED_OTHER_TAG.casefold() for tag in tags)
    tags_have_explicit = any(tag.casefold() != RESERVED_OTHER_TAG.casefold() for tag in tags)
    internal_conflict = best_invalid or tags_invalid
    if not best_is_other and best_tag.casefold() not in {tag.casefold() for tag in tags}:
        internal_conflict = True
    if tags_have_other and tags_have_explicit:
        internal_conflict = True

    return {
        "valid": not internal_conflict,
        "best_tag": best_tag if not internal_conflict else "",
        "tags": tags if not internal_conflict else [],
        "source": SOURCE_RAPID_FIRST_PASS if not internal_conflict else "",
        "status": STATUS_INTERNAL_CONFLICT if internal_conflict else STATUS_FIRST_PASS_ONLY,
    }


def _tag_review_from_luna(row: pd.Series, tag_definitions: dict[str, str] | None) -> dict[str, Any]:
    best_tag, best_invalid = _normalize_one_tag(row.get("Rapid Review Best Tag", ""), tag_definitions)
    tags, tags_invalid = _normalize_tag_set(row.get("Rapid Review Tags", ""), tag_definitions)
    if not best_tag or not tags or best_invalid or tags_invalid:
        return {"valid": False}
    return {"valid": True, "best_tag": best_tag, "tags": tags}


def resolve_rapid_tag_row(
    row: pd.Series | dict[str, Any],
    *,
    tag_definitions: dict[str, str] | None = None,
) -> dict[str, Any]:
    series = pd.Series(row)
    base = _tag_base_from_jev(series, tag_definitions)
    result = {
        "Effective Rapid Best Tag": base["best_tag"],
        "Effective Rapid Tags": "; ".join(base["tags"]),
        "Effective Rapid Tag Source": base["source"],
        "Rapid Tag Resolution Status": base["status"],
    }
    if base["status"] in {STATUS_INVALID, STATUS_INTERNAL_CONFLICT}:
        return result
    if _review_has_error(series):
        result["Rapid Tag Resolution Status"] = STATUS_SECOND_OPINION_ERROR
        return result
    if not _review_has_completed(series):
        return result

    review = _tag_review_from_luna(series, tag_definitions)
    if not review.get("valid"):
        result["Rapid Tag Resolution Status"] = STATUS_SECOND_OPINION_ERROR
        return result

    best_agrees = base["best_tag"].casefold() == review["best_tag"].casefold()
    set_agrees = _tags_equal(base["tags"], review["tags"])
    if best_agrees and set_agrees:
        status = STATUS_AGREEMENT
    elif best_agrees or set_agrees:
        status = STATUS_PARTIAL_AGREEMENT
    else:
        status = STATUS_DISAGREEMENT
    result["Rapid Tag Resolution Status"] = status
    return result


def build_effective_rapid_sentiment_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=EFFECTIVE_SENTIMENT_COLUMNS)
    rows = [resolve_rapid_sentiment_row(row) for _, row in df.iterrows()]
    return pd.DataFrame(rows, index=df.index).reindex(columns=EFFECTIVE_SENTIMENT_COLUMNS)


def build_effective_rapid_tag_frame(
    df: pd.DataFrame,
    *,
    tag_definitions: dict[str, str] | None = None,
) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=EFFECTIVE_TAG_COLUMNS)
    rows = [resolve_rapid_tag_row(row, tag_definitions=tag_definitions) for _, row in df.iterrows()]
    return pd.DataFrame(rows, index=df.index).reindex(columns=EFFECTIVE_TAG_COLUMNS)


def _human_sentiment_assignment(row: pd.Series) -> dict[str, Any] | None:
    state = _safe_text(row.get("Rapid Human Sentiment Review State", ""))
    if state == HUMAN_STATE_ACCEPTED:
        relevance = _safe_text(row.get("Rapid Accepted Machine Relevance", ""))
        three = _clean_label(row.get("Rapid Accepted Machine Sentiment 3-Way", ""))
        five = _clean_label(row.get("Rapid Accepted Machine Sentiment 5-Way", ""))
        score = _to_float(row.get("Rapid Accepted Machine Sentiment Score"))
        return {
            "relevance": relevance,
            "three_way": three,
            "five_way": five,
            "score": pd.NA if score is None else score,
            "source": SOURCE_HUMAN_ACCEPTED_MACHINE,
        }

    if state != HUMAN_STATE_ASSIGNED:
        return None

    scale = _safe_text(row.get("Rapid Human Sentiment Scale", ""))
    label = _clean_label(row.get("Rapid Human Sentiment", ""))
    if not scale or not label:
        return None

    machine = resolve_rapid_sentiment_row(row)
    machine_score = machine.get("Effective Rapid Sentiment Score", pd.NA)
    not_relevant = label == "NOT RELEVANT"
    if not_relevant:
        return {
            "relevance": RELEVANCE_NOT_RELEVANT,
            "three_way": "NOT RELEVANT",
            "five_way": "NOT RELEVANT",
            "score": pd.NA,
            "source": SOURCE_HUMAN_ASSIGNED,
        }

    relevance = RELEVANCE_RELEVANT
    three = machine.get("Effective Rapid Sentiment 3-Way", "")
    five = machine.get("Effective Rapid Sentiment 5-Way", "")
    if scale == HUMAN_SENTIMENT_SCALE_3_WAY:
        three = label
    elif scale == HUMAN_SENTIMENT_SCALE_5_WAY:
        five = label
    else:
        return None
    return {
        "relevance": relevance,
        "three_way": three,
        "five_way": five,
        "score": machine_score,
        "source": SOURCE_HUMAN_ASSIGNED,
    }


def resolve_final_rapid_sentiment_row(row: pd.Series | dict[str, Any]) -> dict[str, Any]:
    series = pd.Series(row)
    human = _human_sentiment_assignment(series)
    if human is not None:
        return {
            "Final Rapid Relevance": human["relevance"],
            "Final Rapid Sentiment 3-Way": human["three_way"],
            "Final Rapid Sentiment 5-Way": human["five_way"],
            "Final Rapid Sentiment Score": human["score"],
            "Final Rapid Sentiment Source": human["source"],
        }

    machine = resolve_rapid_sentiment_row(series)
    return {
        "Final Rapid Relevance": machine["Effective Rapid Relevance"],
        "Final Rapid Sentiment 3-Way": machine["Effective Rapid Sentiment 3-Way"],
        "Final Rapid Sentiment 5-Way": machine["Effective Rapid Sentiment 5-Way"],
        "Final Rapid Sentiment Score": machine["Effective Rapid Sentiment Score"],
        "Final Rapid Sentiment Source": machine["Effective Rapid Sentiment Source"],
    }


def _legacy_human_tag_state(row: pd.Series, expected_mode: str) -> tuple[str, str]:
    mode = _safe_text(row.get("Rapid Human Tagging Mode", ""))
    state = _safe_text(row.get("Rapid Human Tag Review State", ""))
    assignment = _safe_text(row.get("Rapid Human Tag Assignment", ""))
    if not state:
        return "", ""
    if expected_mode == RAPID_TAG_REVIEW_MODE_APPLICABLE:
        if mode in {LEGACY_RAPID_TAGGING_MODE_APPLICABLE, RAPID_TAG_REVIEW_MODE_APPLICABLE}:
            return state, assignment
        if state == HUMAN_STATE_ACCEPTED and not mode and _safe_text(row.get("Rapid Accepted Machine Tags", "")):
            return state, assignment
    else:
        if mode in {LEGACY_RAPID_TAGGING_MODE_BEST, RAPID_TAG_REVIEW_MODE_BEST}:
            return state, assignment
        if state == HUMAN_STATE_ACCEPTED and not mode and _safe_text(row.get("Rapid Accepted Machine Best Tag", "")):
            return state, assignment
    return "", ""


def _human_best_tag_assignment(
    row: pd.Series,
    *,
    tag_definitions: dict[str, str] | None = None,
) -> tuple[str, str] | None:
    state = _safe_text(row.get("Rapid Human Best Tag Review State", ""))
    assignment = _safe_text(row.get("Rapid Human Best Tag Assignment", ""))
    if not state:
        state, assignment = _legacy_human_tag_state(row, RAPID_TAG_REVIEW_MODE_BEST)
    if state == HUMAN_STATE_ACCEPTED:
        best = _safe_text(row.get("Rapid Accepted Machine Best Tag", ""))
        if not best:
            return None
        return best, SOURCE_HUMAN_ACCEPTED_MACHINE

    if state != HUMAN_STATE_ASSIGNED:
        return None

    if not assignment:
        return None
    best_tag, invalid = _normalize_one_tag(assignment, tag_definitions)
    if invalid or not best_tag:
        return None
    return best_tag, SOURCE_HUMAN_ASSIGNED


def _human_applicable_tags_assignment(
    row: pd.Series,
    *,
    tag_definitions: dict[str, str] | None = None,
) -> tuple[str, str] | None:
    state = _safe_text(row.get("Rapid Human Applicable Tags Review State", ""))
    assignment = _safe_text(row.get("Rapid Human Applicable Tags Assignment", ""))
    if not state:
        state, assignment = _legacy_human_tag_state(row, RAPID_TAG_REVIEW_MODE_APPLICABLE)
    if state == HUMAN_STATE_ACCEPTED:
        tags, invalid = _normalize_tag_set(row.get("Rapid Accepted Machine Tags", ""), tag_definitions)
        if not tags:
            return None
        return ("; ".join(tags) if not invalid else _safe_text(row.get("Rapid Accepted Machine Tags", ""))), SOURCE_HUMAN_ACCEPTED_MACHINE

    if state != HUMAN_STATE_ASSIGNED:
        return None

    if not assignment:
        return None
    normalized, invalid = _normalize_tag_set(assignment, tag_definitions)
    if invalid or not normalized:
        return None
    return "; ".join(normalized), SOURCE_HUMAN_ASSIGNED


def resolve_final_rapid_tag_row(
    row: pd.Series | dict[str, Any],
    *,
    tag_definitions: dict[str, str] | None = None,
) -> dict[str, Any]:
    series = pd.Series(row)
    machine = resolve_rapid_tag_row(series, tag_definitions=tag_definitions)
    best = _human_best_tag_assignment(series, tag_definitions=tag_definitions)
    tags = _human_applicable_tags_assignment(series, tag_definitions=tag_definitions)
    best_value = best[0] if best is not None else machine["Effective Rapid Best Tag"]
    tags_value = tags[0] if tags is not None else machine["Effective Rapid Tags"]
    sources = []
    if best is not None:
        sources.append(f"Best fit: {best[1]}")
    if tags is not None:
        sources.append(f"Applicable tags: {tags[1]}")
    source = "; ".join(sources) if sources else machine["Effective Rapid Tag Source"]
    return {
        "Final Rapid Best Tag": best_value,
        "Final Rapid Tags": tags_value,
        "Final Rapid Tag Source": source,
    }


def build_final_rapid_sentiment_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=FINAL_SENTIMENT_COLUMNS)
    rows = [resolve_final_rapid_sentiment_row(row) for _, row in ensure_rapid_human_review_columns(df).iterrows()]
    return pd.DataFrame(rows, index=df.index).reindex(columns=FINAL_SENTIMENT_COLUMNS)


def build_final_rapid_tag_frame(
    df: pd.DataFrame,
    *,
    tag_definitions: dict[str, str] | None = None,
) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=FINAL_TAG_COLUMNS)
    working = ensure_rapid_human_review_columns(df)
    rows = [resolve_final_rapid_tag_row(row, tag_definitions=tag_definitions) for _, row in working.iterrows()]
    return pd.DataFrame(rows, index=df.index).reindex(columns=FINAL_TAG_COLUMNS)


def get_effective_rapid_sentiment(
    df: pd.DataFrame,
) -> pd.DataFrame:
    return build_effective_rapid_sentiment_frame(df)


def get_effective_rapid_tags(
    df: pd.DataFrame,
    *,
    tag_definitions: dict[str, str] | None = None,
) -> pd.DataFrame:
    return build_effective_rapid_tag_frame(df, tag_definitions=tag_definitions)


def summarize_rapid_label_provenance(
    df: pd.DataFrame,
    *,
    tag_definitions: dict[str, str] | None = None,
) -> dict[str, int]:
    if df is None or df.empty:
        return {
            "rapid_labeled_stories": 0,
            "effective_sentiment_available": 0,
            "effective_tags_available": 0,
            "first_pass_only": 0,
            "cross_model_agreement": 0,
            "cross_model_partial_agreement": 0,
            "cross_model_disagreement": 0,
            "first_pass_internal_conflict": 0,
            "second_opinion_errors": 0,
        }

    sentiment = build_effective_rapid_sentiment_frame(df)
    tags = build_effective_rapid_tag_frame(df, tag_definitions=tag_definitions)
    sent_status = Counter(sentiment["Rapid Sentiment Resolution Status"].fillna("").astype(str))
    tag_status = Counter(tags["Rapid Tag Resolution Status"].fillna("").astype(str))
    rapid_labeled = int(
        df.get("Jev Sentiment", pd.Series(index=df.index, dtype="object"))
        .fillna("")
        .astype(str)
        .str.strip()
        .ne("")
        .sum()
    )
    return {
        "rapid_labeled_stories": rapid_labeled,
        "effective_sentiment_available": int(sentiment["Effective Rapid Sentiment 3-Way"].fillna("").astype(str).str.strip().ne("").sum()),
        "effective_tags_available": int(tags["Effective Rapid Best Tag"].fillna("").astype(str).str.strip().ne("").sum()),
        "first_pass_only": sent_status[STATUS_FIRST_PASS_ONLY] + tag_status[STATUS_FIRST_PASS_ONLY],
        "cross_model_agreement": sent_status[STATUS_AGREEMENT] + tag_status[STATUS_AGREEMENT],
        "cross_model_partial_agreement": sent_status[STATUS_PARTIAL_AGREEMENT] + tag_status[STATUS_PARTIAL_AGREEMENT],
        "cross_model_disagreement": sent_status[STATUS_DISAGREEMENT] + tag_status[STATUS_DISAGREEMENT],
        "first_pass_internal_conflict": sent_status[STATUS_INTERNAL_CONFLICT] + tag_status[STATUS_INTERNAL_CONFLICT],
        "second_opinion_errors": sent_status[STATUS_SECOND_OPINION_ERROR] + tag_status[STATUS_SECOND_OPINION_ERROR],
    }
