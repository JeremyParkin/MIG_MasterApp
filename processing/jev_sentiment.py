from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable

import pandas as pd
import requests

from processing.ai_tagging import (
    RESERVED_OTHER_DEFINITION,
    RESERVED_OTHER_TAG,
    clean_text,
    ensure_canonical_tag_definitions,
    normalize_tag_assignment,
    parse_tag_definitions,
)


OPENROUTER_DECISIONS_API_URL = "https://openrouter.ai/api/alpha/decisions"
DEFAULT_JEV_MODEL = "typesafe/jev-1.13"
JEV_MAX_RETRIES = 2
JEV_REQUEST_TIMEOUT = 30
DEFAULT_JEV_MAX_WORKERS = 4
DEFAULT_JEV_TAGGING_MODE = "Single best tag"
JEV_TAGGING_MODES = ("Single best tag", "Multiple applicable tags")
JEV_3WAY_SENTIMENT_ORDER = ["POSITIVE", "NEUTRAL", "NEGATIVE", "NOT RELEVANT"]
JEV_5WAY_SENTIMENT_ORDER = [
    "VERY POSITIVE",
    "SOMEWHAT POSITIVE",
    "NEUTRAL",
    "SOMEWHAT NEGATIVE",
    "VERY NEGATIVE",
    "NOT RELEVANT",
]

JEV_TAG_COLOR_PALETTE = [
    "#6366F1",
    "#14B8A6",
    "#F59E0B",
    "#EC4899",
    "#22C55E",
    "#3B82F6",
    "#A855F7",
    "#EF4444",
    "#06B6D4",
    "#84CC16",
    "#F97316",
    "#8B5CF6",
]
JEV_OTHER_TAG_COLOR = "#6B7280"
RAPID_LABELING_SAMPLE_SHEET_NAME = "RAPID LABELING SAMPLE"
RAPID_LABELING_GROUPED_RESULTS_SHEET_NAME = "RAPID LABELING GROUPED RESULTS"

JEV_TO_RAPID_COLUMN_RENAMES = {
    "Jev Sentiment": "Rapid Sentiment",
    "Jev Selected Probability": "Rapid Sentiment Selected Probability",
    "Jev Confidence": "Rapid Sentiment Confidence",
    "Jev 5-Way Sentiment": "Rapid Sentiment - 5 Way",
    "Jev 5-Way Selected Probability": "Rapid Sentiment - 5 Way Selected Probability",
    "Jev 5-Way Confidence": "Rapid Sentiment - 5 Way Confidence",
    "Jev Sentiment Score": "Rapid Sentiment Score",
    "Jev Score Native Score": "Rapid Sentiment Score Native Score",
    "Jev Score Confidence": "Rapid Sentiment Score Confidence",
    "Jev Score Relevant Probability": "Rapid Relevance Probability",
    "Jev Score Legend": "Rapid Sentiment Score Legend",
    "Jev Score Probabilities": "Rapid Sentiment Score Probabilities",
    "Jev Mixture Score": "Rapid Sentiment Mixture Score",
    "Jev Mixture Label": "Rapid Sentiment Mixture Label",
    "Jev Mixture Confidence": "Rapid Sentiment Mixture Confidence",
    "Jev Mixture Probabilities": "Rapid Sentiment Mixture Probabilities",
    "Jev Mixture Legend": "Rapid Sentiment Mixture Legend",
    "Jev Tagging Mode": "Rapid Tagging Mode",
    "Jev Tags": "Rapid Tags",
    "Jev Tag Count": "Rapid Tag Count",
    "Jev Best Tag": "Rapid Best Tag",
    "Jev Best Tag Selected Probability": "Rapid Best Tag Selected Probability",
    "Jev Best Tag Confidence": "Rapid Best Tag Confidence",
    "Jev Tag Details": "Rapid Tag Details",
    "Jev Model": "Rapid Labeling Model",
    "Jev Input Tokens": "Rapid Labeling Input Tokens",
    "Jev Cost USD": "Rapid Labeling Cost USD",
    "Jev Error": "Rapid Labeling Error",
    "Jev Raw Response": "Rapid Labeling Raw Response",
}


@dataclass(frozen=True)
class JevSentimentScheme:
    name: str
    display_name: str
    storage_prefix: str
    method_type: str
    question_id: str
    labels: tuple[str, ...]
    criteria: dict[str, str]
    instructions: str
    score_levels: tuple[tuple[int, str], ...] = ()
    relevance_question_id: str = ""
    relevance_instructions: str = ""


JEV_SENTIMENT_3WAY_SCHEME = JevSentimentScheme(
    name="3-way",
    display_name="3-way + Not Relevant",
    storage_prefix="Jev",
    method_type="choice",
    question_id="sentiment",
    labels=("POSITIVE", "NEUTRAL", "NEGATIVE", "NOT RELEVANT"),
    instructions="What sentiment does the story convey toward the collective entity?",
    criteria={
        "POSITIVE": (
            "Praise, favorable framing, or beneficial outcomes credited to the collective entity."
        ),
        "NEUTRAL": (
            "Factual, procedural, balanced, incidental, or passing coverage with no clear positive or negative "
            "judgment toward the collective entity. Use this when the entity is mentioned in a normal role such as "
            "researching, reporting, forecasting, warning, hosting, responding, teaching, commemorating, or running "
            "a program, unless the coverage clearly praises or criticizes how the entity did it."
        ),
        "NEGATIVE": (
            "Criticism, unfavorable framing, blame, failure, wrongdoing, poor judgment, harm, scandal, hypocrisy, "
            "incompetence, reputational damage, or negative outcomes attributed to the collective entity itself. "
            "Do not choose this merely because the broader topic is negative."
        ),
        "NOT RELEVANT": (
            "The collective entity is not present at all in the story text. Do not choose this if the primary entity, "
            "an alias, a spokesperson acting for the entity, or a product/sub-brand/program is directly mentioned."
        ),
    },
)


JEV_SENTIMENT_5WAY_SCHEME = JevSentimentScheme(
    name="5-way",
    display_name="5-way + Not Relevant",
    storage_prefix="Jev 5-Way",
    method_type="choice",
    question_id="sentiment_5_way",
    labels=(
        "VERY POSITIVE",
        "SOMEWHAT POSITIVE",
        "NEUTRAL",
        "SOMEWHAT NEGATIVE",
        "VERY NEGATIVE",
        "NOT RELEVANT",
    ),
    instructions="What sentiment intensity does the story convey toward the collective entity?",
    criteria={
        "VERY POSITIVE": (
            "Strong praise, highly favorable framing, or substantial positive impact clearly credited to the collective entity."
        ),
        "SOMEWHAT POSITIVE": (
            "Moderate praise, mildly favorable framing, or limited positive outcomes credited to the collective entity."
        ),
        "NEUTRAL": (
            "Factual, procedural, balanced, incidental, or passing coverage with no clear positive or negative "
            "judgment toward the collective entity. Use this when the entity is mentioned in a normal role such as "
            "researching, reporting, forecasting, warning, hosting, responding, teaching, commemorating, or running "
            "a program, unless the coverage clearly praises or criticizes how the entity did it."
        ),
        "SOMEWHAT NEGATIVE": (
            "Mild criticism, somewhat unfavorable framing, limited reputational concern, or limited negative impact "
            "attributed to the collective entity itself. Do not choose this merely because the broader topic is negative."
        ),
        "VERY NEGATIVE": (
            "Strong criticism, blame, wrongdoing, serious failure, harm, scandal, hypocrisy, incompetence, substantial "
            "reputational damage, or substantial negative outcomes attributed to the collective entity itself."
        ),
        "NOT RELEVANT": (
            "The collective entity is not present at all in the story text. Do not choose this if the primary entity, "
            "an alias, a spokesperson acting for the entity, or a product/sub-brand/program is directly mentioned."
        ),
    },
)


JEV_SENTIMENT_SCORE_SCHEME = JevSentimentScheme(
    name="score-7",
    display_name="Score -7 to +7",
    storage_prefix="Jev Score",
    method_type="score",
    question_id="sentiment_score",
    labels=(),
    instructions=(
        "Place the story on the ordered sentiment score rubric according to the directional reputational takeaway "
        "toward the collective entity."
    ),
    criteria={},
    score_levels=(
        (-7, "Extremely negative toward the collective entity: severe blame, scandal, serious harm, or major reputational damage."),
        (-5, "Strongly negative toward the collective entity: clear criticism, failure, unfairness, or meaningful reputational harm."),
        (-3, "Moderately negative toward the collective entity: unfavorable framing or criticism is present but not severe."),
        (-1, "Slightly negative toward the collective entity: mild concern, limited criticism, or weakly unfavorable framing."),
        (0, "Neutral, factual, balanced, incidental, or no meaningful directional reputational takeaway toward the entity."),
        (1, "Slightly positive toward the collective entity: mild favorable framing or limited positive credit."),
        (3, "Moderately positive toward the collective entity: clear favorable framing or positive outcomes credited to the entity."),
        (5, "Strongly positive toward the collective entity: strong praise or meaningful positive impact credited to the entity."),
        (7, "Extremely positive toward the collective entity: exceptional praise or major positive impact clearly credited to the entity."),
    ),
    relevance_question_id="sentiment_relevant",
    relevance_instructions=(
        "Is the collective entity present in the story through the primary entity, an alias, a spokesperson acting for "
        "the entity, or a product, sub-brand, or program in the entity portfolio?"
    ),
)

JEV_SENTIMENT_MIXTURE_SCHEME = JevSentimentScheme(
    name="mixture",
    display_name="Sentiment Mixture",
    storage_prefix="Jev Mixture",
    method_type="score",
    question_id="sentiment_mixture",
    labels=(),
    instructions=(
        "How internally mixed is the sentiment toward the collective entity within the story? Judge the actual "
        "coexistence of differently valenced treatment of the collective entity, not model uncertainty or the "
        "probability spread of any other sentiment question. Neutral/background material should only increase "
        "mixedness when it meaningfully changes how the collective entity is being treated."
    ),
    criteria={},
    score_levels=(
        (
            0,
            "VERY CONSISTENT: Sentiment toward the collective entity is essentially uniform throughout. This includes "
            "consistently positive, consistently negative, or consistently neutral/factual treatment. Ordinary "
            "factual/background material does not make a story mixed.",
        ),
        (
            1,
            "MOSTLY CONSISTENT: One sentiment direction clearly dominates. Any differently valenced treatment of the "
            "collective entity is minor, brief, or incidental.",
        ),
        (
            2,
            "SOMEWHAT MIXED: One sentiment direction still dominates, but meaningful differently valenced treatment "
            "of the collective entity is present.",
        ),
        (
            3,
            "SUBSTANTIALLY MIXED: Multiple sentiment directions toward the collective entity are prominent and "
            "materially shape the story.",
        ),
        (
            4,
            "HIGHLY MIXED: Strongly contrasting positive, negative, and/or neutral treatment of the collective entity "
            "coexists, with no simple uniform treatment.",
        ),
    ),
)


JEV_SENTIMENT_SCHEMES: dict[str, JevSentimentScheme] = {
    JEV_SENTIMENT_3WAY_SCHEME.name: JEV_SENTIMENT_3WAY_SCHEME,
    JEV_SENTIMENT_5WAY_SCHEME.name: JEV_SENTIMENT_5WAY_SCHEME,
    JEV_SENTIMENT_SCORE_SCHEME.name: JEV_SENTIMENT_SCORE_SCHEME,
    JEV_SENTIMENT_MIXTURE_SCHEME.name: JEV_SENTIMENT_MIXTURE_SCHEME,
}

JEV_COMBINED_SCHEMES = (
    JEV_SENTIMENT_3WAY_SCHEME,
    JEV_SENTIMENT_5WAY_SCHEME,
    JEV_SENTIMENT_SCORE_SCHEME,
    JEV_SENTIMENT_MIXTURE_SCHEME,
)

JEV_SHARED_RESULT_COLUMNS = [
    "Jev Model",
    "Jev Input Tokens",
    "Jev Cost USD",
    "Jev Error",
    "Jev Raw Response",
]

JEV_TAG_CORE_COLUMNS = [
    "Jev Tagging Mode",
    "Jev Tags",
    "Jev Tag Count",
    "Jev Best Tag",
    "Jev Best Tag Selected Probability",
    "Jev Best Tag Confidence",
    "Jev Tag Details",
]


def get_jev_sentiment_scheme(name: str | None) -> JevSentimentScheme:
    return JEV_SENTIMENT_SCHEMES.get(str(name or "").strip(), JEV_SENTIMENT_3WAY_SCHEME)


def parse_jev_tag_definitions(tags_text: str) -> dict[str, str]:
    raw = str(tags_text or "")
    if not raw.strip():
        return {}

    seen: dict[str, int] = {}
    validation_errors: list[str] = []
    for line_number, raw_line in enumerate(raw.splitlines(), start=1):
        line = str(raw_line or "").strip()
        if not line:
            continue
        if ":" not in line:
            validation_errors.append(f"Line {line_number}: use `Tag Name: rationale`.")
            continue
        tag, rationale = line.split(":", 1)
        cleaned_tag = clean_text(tag)
        cleaned_rationale = clean_text(rationale)
        if not cleaned_tag:
            validation_errors.append(f"Line {line_number}: tag name is required.")
            continue
        if not cleaned_rationale:
            validation_errors.append(f"Line {line_number}: rationale is required.")
            continue
        tag_key = cleaned_tag.casefold()
        if tag_key == RESERVED_OTHER_TAG.casefold():
            validation_errors.append(
                f"Line {line_number}: {RESERVED_OTHER_TAG} is protected and added automatically."
            )
            continue
        if tag_key in seen:
            validation_errors.append(
                f"Line {line_number}: duplicate tag `{cleaned_tag}` also appears on line {seen[tag_key]}."
            )
            continue
        seen[tag_key] = line_number

    if validation_errors:
        raise ValueError("\n".join(validation_errors))

    definitions = parse_tag_definitions(raw)
    explicit = get_explicit_jev_tag_definitions(definitions)
    if not explicit:
        return {}
    return ensure_canonical_tag_definitions(explicit)


def get_explicit_jev_tag_definitions(tag_definitions: dict[str, str] | None) -> dict[str, str]:
    definitions = dict(tag_definitions or {})
    return {
        str(tag).strip(): str(rationale).strip()
        for tag, rationale in definitions.items()
        if str(tag).strip()
        and str(rationale).strip()
        and str(tag).strip().casefold() != RESERVED_OTHER_TAG.casefold()
    }


def build_jev_tag_question_map(tag_definitions: dict[str, str] | None) -> list[dict[str, str]]:
    explicit = get_explicit_jev_tag_definitions(tag_definitions)
    ordered = sorted(explicit.items(), key=lambda item: (item[0].casefold(), item[0]))
    return [
        {
            "question_id": f"tag_{index:03d}",
            "tag": tag,
            "rationale": rationale,
        }
        for index, (tag, rationale) in enumerate(ordered, start=1)
    ]


def build_jev_tag_config_fingerprint(tag_definitions: dict[str, str] | None) -> str:
    mapping = build_jev_tag_question_map(tag_definitions)
    payload = [
        {
            "tag_key": item["tag"].casefold(),
            "tag": item["tag"],
            "rationale": item["rationale"],
        }
        for item in mapping
    ]
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def normalize_jev_tagging_mode(mode: str | None) -> str:
    mode_text = str(mode or "").strip()
    return mode_text if mode_text in JEV_TAGGING_MODES else DEFAULT_JEV_TAGGING_MODE


def get_jev_tag_result_columns(tag_definitions: dict[str, str] | None) -> list[str]:
    mapping = build_jev_tag_question_map(tag_definitions)
    if not mapping:
        return []
    columns = list(JEV_TAG_CORE_COLUMNS)
    for item in mapping:
        tag = item["tag"]
        columns.extend(
            [
                f"Jev Best Tag Probability [{tag}]",
                f"Jev Tag [{tag}]",
                f"Jev Tag Probability [{tag}]",
                f"Jev Tag Confidence [{tag}]",
            ]
        )
    columns.append(f"Jev Best Tag Probability [{RESERVED_OTHER_TAG}]")
    return list(dict.fromkeys(columns))


def get_jev_result_columns(scheme: JevSentimentScheme = JEV_SENTIMENT_3WAY_SCHEME) -> list[str]:
    prefix = scheme.storage_prefix
    if scheme.method_type == "score":
        if scheme.name == "mixture":
            return [
                f"{prefix} Score",
                f"{prefix} Label",
                f"{prefix} Confidence",
                f"{prefix} Probabilities",
                f"{prefix} Legend",
            ]
        return [
            "Jev Sentiment Score",
            f"{prefix} Native Score",
            f"{prefix} Confidence",
            f"{prefix} Relevant Probability",
            f"{prefix} Legend",
            f"{prefix} Probabilities",
        ]
    return [
        f"{prefix} Sentiment",
        f"{prefix} Selected Probability",
        f"{prefix} Confidence",
        *[f"{prefix} Probability {label}" for label in scheme.labels],
    ]


def get_jev_combined_result_columns(tag_definitions: dict[str, str] | None = None) -> list[str]:
    columns = list(JEV_SHARED_RESULT_COLUMNS)
    for scheme in JEV_COMBINED_SCHEMES:
        columns.extend(get_jev_result_columns(scheme))
    columns.extend(get_jev_tag_result_columns(tag_definitions))
    return list(dict.fromkeys(columns))


def get_openrouter_api_key(secrets: Any | None = None) -> str:
    if secrets is not None:
        try:
            value = secrets["openrouter_key"]
            if str(value or "").strip():
                return str(value).strip()
        except Exception:
            pass
    return str(os.environ.get("OPENROUTER_API_KEY", "") or "").strip()


def ensure_jev_sentiment_columns(
    df: pd.DataFrame,
    scheme: JevSentimentScheme = JEV_SENTIMENT_3WAY_SCHEME,
    tag_definitions: dict[str, str] | None = None,
) -> pd.DataFrame:
    out = df.copy()
    for column in get_jev_combined_result_columns(tag_definitions):
        if column not in out.columns:
            out[column] = pd.NA
    return out


def get_remaining_jev_sentiment_rows(
    df_unique: pd.DataFrame,
    scheme: JevSentimentScheme = JEV_SENTIMENT_3WAY_SCHEME,
    tag_definitions: dict[str, str] | None = None,
) -> pd.DataFrame:
    if df_unique is None or df_unique.empty:
        return pd.DataFrame()

    working = ensure_jev_sentiment_columns(df_unique, scheme, tag_definitions=tag_definitions)
    error = working["Jev Error"].astype("string").fillna("").str.strip()
    required = ["Jev Sentiment", "Jev 5-Way Sentiment", "Jev Sentiment Score", "Jev Mixture Label"]
    if build_jev_tag_question_map(tag_definitions):
        required.extend(["Jev Best Tag", "Jev Tags"])
    completed = pd.Series(True, index=working.index)
    for column in required:
        completed = completed & working[column].astype("string").fillna("").str.strip().ne("")
    remaining_mask = (~completed) & (error == "")
    return working.loc[remaining_mask].reset_index(drop=False)


def build_jev_sentiment_state(
    row: pd.Series | dict[str, Any],
    analysis_payload: dict[str, Any],
) -> dict[str, Any]:
    row_series = pd.Series(row)
    primary_name = _safe_text(analysis_payload.get("primary_name", ""))
    aliases = _clean_list(analysis_payload.get("alternate_names", []))
    spokespeople = _clean_list(analysis_payload.get("spokespeople", []))
    products = _clean_list(analysis_payload.get("products", []))
    shared_guidance = _safe_text(analysis_payload.get("general_guidance", analysis_payload.get("guidance", "")))
    sentiment_guidance = _safe_text(analysis_payload.get("sentiment_guidance", ""))

    headline = _safe_text(row_series.get("Headline", ""))
    body = _safe_text(row_series.get("Example Snippet", row_series.get("Snippet", "")))

    return {
        "entity_context": {
            "primary_entity": primary_name,
            "collective_entity_definition": [
                "the primary entity",
                "aliases or alternate names",
                "named spokespeople when acting on behalf of the entity",
                "products, sub-brands, or programs when discussed as part of the entity's activity or portfolio",
            ],
            "aliases": aliases,
            "spokespeople": spokespeople,
            "products_subbrands_programs": products,
        },
        "sentiment_rules": {
            "judge": "Net sentiment conveyed to a typical reader/viewer about the collective entity.",
            "entity_vs_topic": (
                "Judge sentiment toward the collective entity itself, not toward the broader topic, event, market "
                "condition, social problem, or historic issue being discussed."
            ),
            "direct_mention_scope": (
                "If the collective entity is directly mentioned anywhere in the headline, body, or transcript, "
                "NOT RELEVANT is not appropriate."
            ),
            "configured_member_scope": (
                "The parent organization does not need to be explicitly named for the story to be relevant. "
                "Substantive coverage of any configured alias, spokesperson acting for the entity, product, "
                "sub-brand, or program is in scope for sentiment."
            ),
            "incidental_mentions": (
                "If the entity is mentioned but coverage is only incidental or passing and does not express judgment, "
                "choose NEUTRAL."
            ),
            "negative_subject_matter": (
                "Negative subject matter does not automatically mean negative sentiment toward the collective entity."
            ),
            "normal_entity_activities": (
                "Researching, reporting, forecasting, warning, hosting, teaching, commemorating, responding, or other "
                "normal entity activities are generally NEUTRAL unless coverage clearly praises or criticizes how the "
                "entity did it."
            ),
            "attribution": [
                f"When a spokesperson acts explicitly for {primary_name}, attribute their stance to the entity.",
                f"When a product, sub-brand, or program is discussed, attribute sentiment to {primary_name} unless clearly unrelated.",
                "Do not infer sentiment toward the entity from third parties or adjacent topics.",
            ],
            "tie_breakers": [
                "Use the audience takeaway when positive and negative elements coexist.",
                "Prefer explicit attributions, direct quotes, headlines, and framing to infer stance.",
            ],
        },
        "analysis_guidance": {
            "shared_guidance": shared_guidance,
            "sentiment_specific_guidance": sentiment_guidance,
            "sentiment_specific_guidance_priority": (
                "HIGH-PRIORITY: Apply sentiment-specific guidance whenever relevant, not only in gray areas. If it "
                "conflicts with default heuristics, follow the workflow-specific guidance."
            )
            if sentiment_guidance
            else "",
        },
        "story": {
            "headline": headline,
            "body": body,
            "outlet": _safe_text(row_series.get("Outlet", "")),
            "media_type": _safe_text(row_series.get("Type", "")),
            "date": _safe_json_scalar(row_series.get("Date", "")),
        },
    }


def build_jev_sentiment_question(
    scheme: JevSentimentScheme = JEV_SENTIMENT_3WAY_SCHEME,
) -> dict[str, Any]:
    if scheme.method_type == "score":
        return {
            "type": "score",
            "instructions": scheme.instructions,
            "criteria": [description for _, description in scheme.score_levels],
        }
    return {
        "type": "choice",
        "instructions": scheme.instructions,
        "criteria": {label: scheme.criteria[label] for label in scheme.labels},
    }


def build_jev_relevance_question(
    scheme: JevSentimentScheme = JEV_SENTIMENT_SCORE_SCHEME,
) -> dict[str, Any]:
    return {
        "type": "noul",
        "instructions": scheme.relevance_instructions,
    }


def build_jev_tag_questions(tag_definitions: dict[str, str] | None) -> dict[str, Any]:
    mapping = build_jev_tag_question_map(tag_definitions)
    if not mapping:
        return {}

    full_definitions = ensure_canonical_tag_definitions(get_explicit_jev_tag_definitions(tag_definitions))
    criteria = {
        item["tag"]: item["rationale"]
        for item in mapping
    }
    criteria[RESERVED_OTHER_TAG] = RESERVED_OTHER_DEFINITION

    questions: dict[str, Any] = {
        "tag_best_fit": {
            "type": "choice",
            "instructions": (
                "Which single tag best fits this story? Choose exactly one option. Use Other when none of the "
                "explicit tags is a meaningful fit."
            ),
            "criteria": criteria,
        }
    }

    for item in mapping:
        tag = item["tag"]
        rationale = full_definitions.get(tag, item["rationale"])
        questions[item["question_id"]] = {
            "type": "choice",
            "instructions": (
                f"Does this story qualify for the tag `{tag}`? Judge semantic fit to the analyst rationale, "
                "not keyword presence alone."
            ),
            "criteria": {
                "YES": f"The story meaningfully qualifies for `{tag}`. Rationale: {rationale}",
                "NO": f"The story does not meaningfully qualify for `{tag}`.",
            },
        }
    return questions


def build_jev_sentiment_payload(
    row: pd.Series | dict[str, Any],
    analysis_payload: dict[str, Any],
    *,
    model: str = DEFAULT_JEV_MODEL,
    scheme: JevSentimentScheme = JEV_SENTIMENT_3WAY_SCHEME,
) -> dict[str, Any]:
    questions = {
        scheme.question_id: build_jev_sentiment_question(scheme),
    }
    if scheme.method_type == "score":
        questions[scheme.relevance_question_id] = build_jev_relevance_question(scheme)
    return {
        "model": model,
        "state": build_jev_sentiment_state(row, analysis_payload),
        "questions": questions,
    }


def build_jev_combined_sentiment_payload(
    row: pd.Series | dict[str, Any],
    analysis_payload: dict[str, Any],
    *,
    model: str = DEFAULT_JEV_MODEL,
    tag_definitions: dict[str, str] | None = None,
) -> dict[str, Any]:
    questions: dict[str, Any] = {}
    for scheme in JEV_COMBINED_SCHEMES:
        questions[scheme.question_id] = build_jev_sentiment_question(scheme)
        if scheme.method_type == "score" and scheme.relevance_question_id:
            questions[scheme.relevance_question_id] = build_jev_relevance_question(scheme)
    questions.update(build_jev_tag_questions(tag_definitions))
    return {
        "model": model,
        "state": build_jev_sentiment_state(row, analysis_payload),
        "questions": questions,
    }


def call_jev_sentiment(
    payload: dict[str, Any],
    api_key: str,
    *,
    api_url: str = OPENROUTER_DECISIONS_API_URL,
    timeout: int = JEV_REQUEST_TIMEOUT,
    max_retries: int = JEV_MAX_RETRIES,
) -> dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            response = requests.post(api_url, headers=headers, json=payload, timeout=timeout)
            if response.status_code in {429, 529, 500, 502, 503, 504} and attempt < max_retries:
                retry_after = _parse_retry_after(response)
                time.sleep(retry_after if retry_after is not None else min(2 ** attempt, 8))
                continue
            response.raise_for_status()
            parsed = response.json()
            if not isinstance(parsed, dict):
                raise ValueError("OpenRouter Decisions response was not a JSON object.")
            return parsed
        except Exception as exc:
            last_error = exc
            if attempt < max_retries:
                time.sleep(min(2 ** attempt, 8))
                continue
            raise
    raise RuntimeError(f"OpenRouter Decisions request failed: {last_error}")


def parse_jev_sentiment_response(
    response_json: dict[str, Any],
    *,
    scheme: JevSentimentScheme = JEV_SENTIMENT_3WAY_SCHEME,
) -> dict[str, Any]:
    answers = response_json.get("answers")
    if not isinstance(answers, dict):
        raise ValueError("OpenRouter Decisions response missing answers.")

    answer = answers.get(scheme.question_id)
    if not isinstance(answer, dict):
        raise ValueError(f"OpenRouter Decisions response missing `{scheme.question_id}` answer.")

    if scheme.method_type == "score":
        return _parse_jev_score_response(response_json, answer, answers, scheme)

    if answer.get("type") != "choice":
        raise ValueError("Rapid sentiment answer was not a choice answer.")

    choice = str(answer.get("choice", "") or "").strip().upper()
    if choice not in scheme.labels:
        raise ValueError(f"Unexpected Rapid sentiment label: {choice or 'missing'}")

    probabilities = answer.get("probabilities")
    if not isinstance(probabilities, dict):
        raise ValueError("Rapid sentiment answer missing probabilities.")

    normalized_probabilities = {
        label: _coerce_probability(probabilities.get(label, 0.0))
        for label in scheme.labels
    }
    selected_probability = normalized_probabilities.get(choice, 0.0)
    confidence = _coerce_probability(answer.get("confidence", 0.0))
    usage = response_json.get("usage") if isinstance(response_json.get("usage"), dict) else {}
    input_tokens = int(usage.get("input_tokens", usage.get("inputTokens", 0)) or 0)
    cost_usd = _coerce_float(usage.get("cost", None))
    if cost_usd is None:
        cost_usd = 0.0

    return {
        "sentiment": choice,
        "selected_probability": selected_probability,
        "confidence": confidence,
        "probabilities": normalized_probabilities,
        "model": str(response_json.get("model", "") or "").strip(),
        "input_tokens": input_tokens,
        "cost_usd": float(cost_usd or 0.0),
        "raw_response": json.dumps(response_json, ensure_ascii=False, sort_keys=True),
    }


def parse_jev_combined_sentiment_response(
    response_json: dict[str, Any],
    *,
    tag_definitions: dict[str, str] | None = None,
    tagging_mode: str = DEFAULT_JEV_TAGGING_MODE,
) -> dict[str, Any]:
    usage = response_json.get("usage") if isinstance(response_json.get("usage"), dict) else {}
    input_tokens = int(usage.get("input_tokens", usage.get("inputTokens", 0)) or 0)
    cost_usd = _coerce_float(usage.get("cost", None))
    if cost_usd is None:
        cost_usd = 0.0

    parsed = {
        "model": str(response_json.get("model", "") or "").strip(),
        "input_tokens": input_tokens,
        "cost_usd": float(cost_usd or 0.0),
        "raw_response": json.dumps(response_json, ensure_ascii=False, sort_keys=True),
        "methods": {},
    }
    for scheme in JEV_COMBINED_SCHEMES:
        parsed["methods"][scheme.name] = parse_jev_sentiment_response(response_json, scheme=scheme)
    if build_jev_tag_question_map(tag_definitions):
        parsed["tagging"] = parse_jev_tagging_response(
            response_json,
            tag_definitions=tag_definitions,
            tagging_mode=tagging_mode,
        )
    return parsed


def parse_jev_tagging_response(
    response_json: dict[str, Any],
    *,
    tag_definitions: dict[str, str] | None,
    tagging_mode: str = DEFAULT_JEV_TAGGING_MODE,
) -> dict[str, Any]:
    mapping = build_jev_tag_question_map(tag_definitions)
    if not mapping:
        return {}

    answers = response_json.get("answers")
    if not isinstance(answers, dict):
        raise ValueError("OpenRouter Decisions response missing answers.")

    explicit_tags = [item["tag"] for item in mapping]
    allowed_best_tags = [*explicit_tags, RESERVED_OTHER_TAG]
    best_answer = answers.get("tag_best_fit")
    if not isinstance(best_answer, dict) or best_answer.get("type") != "choice":
        raise ValueError("OpenRouter Decisions response missing `tag_best_fit` choice answer.")

    best_tag = _match_allowed_label(best_answer.get("choice", ""), allowed_best_tags)
    if not best_tag:
        raise ValueError(f"Unexpected Rapid best-fit tag: {best_answer.get('choice', '') or 'missing'}")
    best_probabilities_raw = best_answer.get("probabilities")
    if not isinstance(best_probabilities_raw, dict):
        raise ValueError("Rapid best-fit tag answer missing probabilities.")
    best_probabilities = {
        tag: _coerce_probability(best_probabilities_raw.get(tag, 0.0))
        for tag in allowed_best_tags
    }

    independent: dict[str, dict[str, Any]] = {}
    for item in mapping:
        tag = item["tag"]
        question_id = item["question_id"]
        answer = answers.get(question_id)
        if not isinstance(answer, dict) or answer.get("type") != "choice":
            raise ValueError(f"OpenRouter Decisions response missing `{question_id}` tag choice answer.")
        choice = str(answer.get("choice", "") or "").strip().upper()
        if choice not in {"YES", "NO"}:
            raise ValueError(f"Unexpected Rapid tag applicability label for `{tag}`: {choice or 'missing'}")
        probabilities = answer.get("probabilities")
        if not isinstance(probabilities, dict):
            raise ValueError(f"Rapid tag applicability answer missing probabilities for `{tag}`.")
        independent[tag] = {
            "question_id": question_id,
            "choice": choice,
            "yes_probability": _coerce_probability(probabilities.get("YES", 0.0)),
            "no_probability": _coerce_probability(probabilities.get("NO", 0.0)),
            "confidence": _coerce_probability(answer.get("confidence", 0.0)),
            "probabilities": {
                "YES": _coerce_probability(probabilities.get("YES", 0.0)),
                "NO": _coerce_probability(probabilities.get("NO", 0.0)),
            },
        }

    official_tags = compute_jev_official_tags(
        best_tag=best_tag,
        independent=independent,
        tagging_mode=tagging_mode,
        tag_definitions=tag_definitions,
    )

    return {
        "tagging_mode": normalize_jev_tagging_mode(tagging_mode),
        "tags": official_tags,
        "tag_count": len(official_tags),
        "best_tag": best_tag,
        "best_selected_probability": best_probabilities.get(best_tag, 0.0),
        "best_confidence": _coerce_probability(best_answer.get("confidence", 0.0)),
        "best_probabilities": best_probabilities,
        "independent": independent,
        "details_json": json.dumps(
            {
                "best_fit": {
                    "tag": best_tag,
                    "selected_probability": best_probabilities.get(best_tag, 0.0),
                    "confidence": _coerce_probability(best_answer.get("confidence", 0.0)),
                    "probabilities": best_probabilities,
                },
                "independent": independent,
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
    }


def compute_jev_official_tags(
    *,
    best_tag: str,
    independent: dict[str, dict[str, Any]],
    tagging_mode: str,
    tag_definitions: dict[str, str] | None,
) -> list[str]:
    mode = normalize_jev_tagging_mode(tagging_mode)
    if mode == "Single best tag":
        return normalize_tag_assignment(best_tag, ensure_canonical_tag_definitions(get_explicit_jev_tag_definitions(tag_definitions)))

    yes_tags = [
        tag
        for tag, result in independent.items()
        if str(result.get("choice", "") or "").strip().upper() == "YES"
    ]
    return normalize_tag_assignment(yes_tags, ensure_canonical_tag_definitions(get_explicit_jev_tag_definitions(tag_definitions)))


def _parse_jev_score_response(
    response_json: dict[str, Any],
    answer: dict[str, Any],
    answers: dict[str, Any],
    scheme: JevSentimentScheme,
) -> dict[str, Any]:
    if answer.get("type") != "score":
        raise ValueError("Rapid sentiment score answer was not a score answer.")

    native_score = _coerce_float(answer.get("score", None))
    if native_score is None:
        raise ValueError("Rapid sentiment score answer missing score.")

    probabilities = answer.get("probabilities")
    if not isinstance(probabilities, dict):
        raise ValueError("Rapid sentiment score answer missing probabilities.")

    legend = answer.get("legend")
    if not isinstance(legend, dict):
        raise ValueError("Rapid sentiment score answer missing legend.")

    relevant_probability = None
    if scheme.relevance_question_id:
        relevance_answer = answers.get(scheme.relevance_question_id)
        if not isinstance(relevance_answer, dict) or relevance_answer.get("type") != "noul":
            raise ValueError(f"OpenRouter Decisions response missing `{scheme.relevance_question_id}` noul answer.")
        relevant_probability = _coerce_probability(relevance_answer.get("noul", 0.0))
    numeric_probabilities = {
        str(index): _coerce_probability(probabilities.get(str(index), probabilities.get(index, 0.0)))
        for index in range(len(scheme.score_levels))
    }
    max_probability_index = max(
        numeric_probabilities,
        key=lambda key: numeric_probabilities.get(key, 0.0),
    )
    max_index = int(max_probability_index)
    label = _score_level_label(scheme.score_levels[max_index][1])

    if scheme.name == "mixture":
        score_value = float(native_score)
    else:
        score_value = 0.0
        for index, (level_value, _) in enumerate(scheme.score_levels):
            score_value += float(level_value) * numeric_probabilities.get(str(index), 0.0)

    confidence = _coerce_probability(answer.get("confidence", 0.0))
    usage = response_json.get("usage") if isinstance(response_json.get("usage"), dict) else {}
    input_tokens = int(usage.get("input_tokens", usage.get("inputTokens", 0)) or 0)
    cost_usd = _coerce_float(usage.get("cost", None))
    if cost_usd is None:
        cost_usd = 0.0

    return {
        "score": score_value,
        "native_score": float(native_score),
        "label": label,
        "confidence": confidence,
        "relevant_probability": relevant_probability,
        "legend": json.dumps(legend, ensure_ascii=False, sort_keys=True),
        "score_probabilities": json.dumps(numeric_probabilities, ensure_ascii=False, sort_keys=True),
        "model": str(response_json.get("model", "") or "").strip(),
        "input_tokens": input_tokens,
        "cost_usd": float(cost_usd or 0.0),
        "raw_response": json.dumps(response_json, ensure_ascii=False, sort_keys=True),
    }


def apply_jev_sentiment_result_to_unique_df(
    df_unique: pd.DataFrame,
    original_index: int,
    result: dict[str, Any],
    *,
    scheme: JevSentimentScheme = JEV_SENTIMENT_3WAY_SCHEME,
) -> pd.DataFrame:
    out = ensure_jev_sentiment_columns(df_unique, scheme)
    if scheme.method_type == "score":
        prefix = scheme.storage_prefix
        out.loc[original_index, f"{prefix} Score"] = result.get("score")
        out.loc[original_index, f"{prefix} Native Score"] = result.get("native_score")
        out.loc[original_index, f"{prefix} Confidence"] = result.get("confidence")
        out.loc[original_index, f"{prefix} Relevant Probability"] = result.get("relevant_probability")
        out.loc[original_index, f"{prefix} Legend"] = result.get("legend")
        out.loc[original_index, f"{prefix} Probabilities"] = result.get("score_probabilities")
        out.loc[original_index, f"{prefix} Model"] = result.get("model")
        out.loc[original_index, f"{prefix} Input Tokens"] = result.get("input_tokens")
        out.loc[original_index, f"{prefix} Cost USD"] = result.get("cost_usd")
        out.loc[original_index, f"{prefix} Error"] = pd.NA
        out.loc[original_index, f"{prefix} Raw Response"] = result.get("raw_response")
        return out

    probabilities = result.get("probabilities", {}) if isinstance(result.get("probabilities"), dict) else {}
    prefix = scheme.storage_prefix

    out.loc[original_index, f"{prefix} Sentiment"] = result.get("sentiment")
    out.loc[original_index, f"{prefix} Selected Probability"] = result.get("selected_probability")
    out.loc[original_index, f"{prefix} Confidence"] = result.get("confidence")
    for label in scheme.labels:
        out.loc[original_index, f"{prefix} Probability {label}"] = probabilities.get(label, 0.0)
    out.loc[original_index, f"{prefix} Model"] = result.get("model")
    out.loc[original_index, f"{prefix} Input Tokens"] = result.get("input_tokens")
    out.loc[original_index, f"{prefix} Cost USD"] = result.get("cost_usd")
    out.loc[original_index, f"{prefix} Error"] = pd.NA
    out.loc[original_index, f"{prefix} Raw Response"] = result.get("raw_response")
    return out


def apply_jev_combined_result_to_unique_df(
    df_unique: pd.DataFrame,
    original_index: int,
    result: dict[str, Any],
    *,
    tag_definitions: dict[str, str] | None = None,
    tagging_mode: str = DEFAULT_JEV_TAGGING_MODE,
) -> pd.DataFrame:
    out = ensure_jev_sentiment_columns(df_unique, tag_definitions=tag_definitions)
    out.loc[original_index, "Jev Model"] = result.get("model")
    out.loc[original_index, "Jev Input Tokens"] = result.get("input_tokens")
    out.loc[original_index, "Jev Cost USD"] = result.get("cost_usd")
    out.loc[original_index, "Jev Error"] = pd.NA
    out.loc[original_index, "Jev Raw Response"] = result.get("raw_response")

    methods = result.get("methods", {}) if isinstance(result.get("methods"), dict) else {}
    for scheme in JEV_COMBINED_SCHEMES:
        method_result = methods.get(scheme.name, {})
        if scheme.method_type == "score":
            if scheme.name == "mixture":
                out.loc[original_index, "Jev Mixture Score"] = method_result.get("score")
                out.loc[original_index, "Jev Mixture Label"] = method_result.get("label")
                out.loc[original_index, "Jev Mixture Confidence"] = method_result.get("confidence")
                out.loc[original_index, "Jev Mixture Probabilities"] = method_result.get("score_probabilities")
                out.loc[original_index, "Jev Mixture Legend"] = method_result.get("legend")
                continue
            out.loc[original_index, "Jev Sentiment Score"] = method_result.get("score")
            out.loc[original_index, f"{scheme.storage_prefix} Native Score"] = method_result.get("native_score")
            out.loc[original_index, f"{scheme.storage_prefix} Confidence"] = method_result.get("confidence")
            out.loc[original_index, f"{scheme.storage_prefix} Relevant Probability"] = method_result.get("relevant_probability")
            out.loc[original_index, f"{scheme.storage_prefix} Legend"] = method_result.get("legend")
            out.loc[original_index, f"{scheme.storage_prefix} Probabilities"] = method_result.get("score_probabilities")
            continue

        probabilities = (
            method_result.get("probabilities", {})
            if isinstance(method_result.get("probabilities"), dict)
            else {}
        )
        prefix = scheme.storage_prefix
        out.loc[original_index, f"{prefix} Sentiment"] = method_result.get("sentiment")
        out.loc[original_index, f"{prefix} Selected Probability"] = method_result.get("selected_probability")
        out.loc[original_index, f"{prefix} Confidence"] = method_result.get("confidence")
        for label in scheme.labels:
            out.loc[original_index, f"{prefix} Probability {label}"] = probabilities.get(label, 0.0)
    tagging_result = result.get("tagging", {}) if isinstance(result.get("tagging"), dict) else {}
    if tagging_result:
        out = apply_jev_tagging_result_to_unique_df(
            out,
            original_index,
            tagging_result,
            tag_definitions=tag_definitions,
            tagging_mode=tagging_mode,
        )
    return out


def apply_jev_tagging_result_to_unique_df(
    df_unique: pd.DataFrame,
    original_index: int,
    tagging_result: dict[str, Any],
    *,
    tag_definitions: dict[str, str] | None,
    tagging_mode: str = DEFAULT_JEV_TAGGING_MODE,
) -> pd.DataFrame:
    out = ensure_jev_sentiment_columns(df_unique, tag_definitions=tag_definitions)
    mode = normalize_jev_tagging_mode(tagging_mode)
    tags = tagging_result.get("tags", [])
    if not isinstance(tags, list):
        tags = normalize_tag_assignment(tags, ensure_canonical_tag_definitions(get_explicit_jev_tag_definitions(tag_definitions)))

    out.loc[original_index, "Jev Tagging Mode"] = mode
    out.loc[original_index, "Jev Tags"] = "; ".join(str(tag) for tag in tags)
    out.loc[original_index, "Jev Tag Count"] = len(tags)
    out.loc[original_index, "Jev Best Tag"] = tagging_result.get("best_tag")
    out.loc[original_index, "Jev Best Tag Selected Probability"] = tagging_result.get("best_selected_probability")
    out.loc[original_index, "Jev Best Tag Confidence"] = tagging_result.get("best_confidence")
    out.loc[original_index, "Jev Tag Details"] = tagging_result.get("details_json")

    best_probabilities = (
        tagging_result.get("best_probabilities", {})
        if isinstance(tagging_result.get("best_probabilities"), dict)
        else {}
    )
    independent = (
        tagging_result.get("independent", {})
        if isinstance(tagging_result.get("independent"), dict)
        else {}
    )
    for item in build_jev_tag_question_map(tag_definitions):
        tag = item["tag"]
        tag_result = independent.get(tag, {}) if isinstance(independent.get(tag, {}), dict) else {}
        out.loc[original_index, f"Jev Best Tag Probability [{tag}]"] = best_probabilities.get(tag, 0.0)
        out.loc[original_index, f"Jev Tag [{tag}]"] = tag_result.get("choice")
        out.loc[original_index, f"Jev Tag Probability [{tag}]"] = tag_result.get("yes_probability")
        out.loc[original_index, f"Jev Tag Confidence [{tag}]"] = tag_result.get("confidence")
    out.loc[original_index, f"Jev Best Tag Probability [{RESERVED_OTHER_TAG}]"] = best_probabilities.get(RESERVED_OTHER_TAG, 0.0)
    return out


def recompute_jev_official_tag_assignments(
    df_unique: pd.DataFrame,
    *,
    tag_definitions: dict[str, str] | None,
    tagging_mode: str = DEFAULT_JEV_TAGGING_MODE,
) -> pd.DataFrame:
    if df_unique is None or df_unique.empty or not build_jev_tag_question_map(tag_definitions):
        return pd.DataFrame() if df_unique is None else df_unique.copy()

    out = ensure_jev_sentiment_columns(df_unique, tag_definitions=tag_definitions)
    mode = normalize_jev_tagging_mode(tagging_mode)
    for index, row in out.iterrows():
        best_tag = _safe_text(row.get("Jev Best Tag", ""))
        if not best_tag:
            continue
        independent = {}
        for item in build_jev_tag_question_map(tag_definitions):
            tag = item["tag"]
            independent[tag] = {
                "choice": _safe_text(row.get(f"Jev Tag [{tag}]", "")).upper(),
            }
        tags = compute_jev_official_tags(
            best_tag=best_tag,
            independent=independent,
            tagging_mode=mode,
            tag_definitions=tag_definitions,
        )
        out.loc[index, "Jev Tagging Mode"] = mode
        out.loc[index, "Jev Tags"] = "; ".join(tags)
        out.loc[index, "Jev Tag Count"] = len(tags)
    return out


def apply_jev_sentiment_error_to_unique_df(
    df_unique: pd.DataFrame,
    original_index: int,
    error_message: str,
    *,
    scheme: JevSentimentScheme = JEV_SENTIMENT_3WAY_SCHEME,
    tag_definitions: dict[str, str] | None = None,
) -> pd.DataFrame:
    out = ensure_jev_sentiment_columns(df_unique, scheme, tag_definitions=tag_definitions)
    out.loc[original_index, "Jev Error"] = str(error_message or "Unknown Rapid Labeling error").strip()
    return out


def _run_jev_sentiment_row(
    row_dict: dict[str, Any],
    analysis_payload: dict[str, Any],
    api_key: str,
    model: str,
    call_fn: Callable[[dict[str, Any], str], dict[str, Any]],
    tag_definitions: dict[str, str] | None = None,
    tagging_mode: str = DEFAULT_JEV_TAGGING_MODE,
) -> dict[str, Any]:
    row = pd.Series(row_dict)
    original_index = int(row.get("index", row.name if row.name is not None else 0))
    payload = build_jev_combined_sentiment_payload(
        row,
        analysis_payload,
        model=model,
        tag_definitions=tag_definitions,
    )
    response_json = call_fn(payload, api_key)
    result = parse_jev_combined_sentiment_response(
        response_json,
        tag_definitions=tag_definitions,
        tagging_mode=tagging_mode,
    )
    return {
        "original_index": original_index,
        "group_id": row.get("Group ID", original_index),
        "result": result,
    }


def run_jev_sentiment_batch(
    df_unique: pd.DataFrame,
    batch_df: pd.DataFrame,
    analysis_payload: dict[str, Any],
    api_key: str,
    *,
    model: str = DEFAULT_JEV_MODEL,
    scheme: JevSentimentScheme = JEV_SENTIMENT_3WAY_SCHEME,
    call_fn: Callable[[dict[str, Any], str], dict[str, Any]] = call_jev_sentiment,
    max_workers: int = DEFAULT_JEV_MAX_WORKERS,
    progress_callback: Callable[[int, int], None] | None = None,
    tag_definitions: dict[str, str] | None = None,
    tagging_mode: str = DEFAULT_JEV_TAGGING_MODE,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    updated = ensure_jev_sentiment_columns(df_unique, tag_definitions=tag_definitions)
    errors: list[str] = []
    completed = 0
    total_input_tokens = 0
    total_cost = 0.0
    total = len(batch_df)

    if total == 0:
        return updated, {
            "done": 0,
            "successful": 0,
            "errors": [],
            "input_tokens": 0,
            "cost_usd": 0.0,
        }

    rows = [row.to_dict() for _, row in batch_df.iterrows()]
    worker_count = max(1, min(int(max_workers or 1), total))
    finished = 0

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(
                _run_jev_sentiment_row,
                row_dict,
                analysis_payload,
                api_key,
                model,
                call_fn,
                tag_definitions,
                tagging_mode,
            ): int(row_dict.get("index", fallback_index))
            for fallback_index, row_dict in enumerate(rows)
        }

        for future in as_completed(futures):
            original_index = futures[future]
            try:
                row_result = future.result()
                original_index = int(row_result["original_index"])
                result = row_result["result"]
                updated = apply_jev_combined_result_to_unique_df(
                    updated,
                    original_index,
                    result,
                    tag_definitions=tag_definitions,
                    tagging_mode=tagging_mode,
                )
                total_input_tokens += int(result.get("input_tokens", 0) or 0)
                total_cost += float(result.get("cost_usd", 0.0) or 0.0)
                completed += 1
            except Exception as exc:
                message = str(exc)
                updated = apply_jev_sentiment_error_to_unique_df(
                    updated,
                    original_index,
                    message,
                    scheme=scheme,
                    tag_definitions=tag_definitions,
                )
                errors.append(f"Story {original_index + 1}: {message}")
            finally:
                finished += 1
                if progress_callback is not None:
                    progress_callback(finished, total)

    return updated, {
        "done": total,
        "successful": completed,
        "errors": errors,
        "input_tokens": total_input_tokens,
        "cost_usd": total_cost,
    }


def build_jev_results_display(
    df_unique: pd.DataFrame,
    scheme: JevSentimentScheme = JEV_SENTIMENT_3WAY_SCHEME,
    tag_definitions: dict[str, str] | None = None,
) -> pd.DataFrame:
    if df_unique is None or df_unique.empty:
        return pd.DataFrame()

    working = ensure_jev_sentiment_columns(df_unique, scheme, tag_definitions=tag_definitions)
    columns = [
        "Group ID",
        "Headline",
        "Outlet",
        "Type",
        "Group Count",
        "Final Sentiment",
        "AI Sentiment",
        "AI Sentiment Confidence",
        "Review AI Sentiment",
        "Assigned Sentiment",
        "Jev Sentiment",
        "Jev Selected Probability",
        "Jev Confidence",
        "Jev 5-Way Sentiment",
        "Jev 5-Way Selected Probability",
        "Jev 5-Way Confidence",
        "Jev Sentiment Score",
        "Jev Score Confidence",
        "Jev Score Relevant Probability",
        "Jev Mixture Label",
        "Jev Mixture Score",
        "Jev Mixture Confidence",
        "Jev Tags",
        "Jev Tag Count",
        "Jev Best Tag",
        "Jev Model",
        "Jev Input Tokens",
        "Jev Cost USD",
        "Jev Error",
    ]
    for item in build_jev_tag_question_map(tag_definitions):
        tag = item["tag"]
        columns.extend([f"Jev Tag [{tag}]", f"Jev Tag Probability [{tag}]"])
    existing = [column for column in columns if column in working.columns]
    return working[existing].copy()


def rapid_labeling_column_name(column: str) -> str:
    name = str(column)
    if name in JEV_TO_RAPID_COLUMN_RENAMES:
        return JEV_TO_RAPID_COLUMN_RENAMES[name]
    if name.startswith("Jev Probability "):
        label = name.removeprefix("Jev Probability ").strip()
        return f"Rapid Sentiment Probability [{label}]"
    if name.startswith("Jev 5-Way Probability "):
        label = name.removeprefix("Jev 5-Way Probability ").strip()
        return f"Rapid Sentiment - 5 Way Probability [{label}]"
    if name.startswith("Jev Tag ["):
        return "Rapid" + name.removeprefix("Jev")
    if name.startswith("Jev Tag Probability ["):
        return "Rapid" + name.removeprefix("Jev")
    if name.startswith("Jev Tag Confidence ["):
        return "Rapid" + name.removeprefix("Jev")
    if name.startswith("Jev Best Tag Probability ["):
        return "Rapid" + name.removeprefix("Jev")
    return name


def rename_jev_columns_to_rapid(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or not isinstance(df, pd.DataFrame):
        return pd.DataFrame() if df is None else df
    return df.rename(columns={column: rapid_labeling_column_name(column) for column in df.columns})


def build_jev_sentiment_distribution(
    df_unique: pd.DataFrame,
    *,
    column: str,
    order: list[str],
) -> pd.DataFrame:
    if df_unique is None or df_unique.empty:
        return pd.DataFrame({"Sentiment": order, "Count": [0] * len(order), "Grouped Stories": [0] * len(order), "Share": [0.0] * len(order)})

    working = df_unique.copy()
    if column not in working.columns:
        return pd.DataFrame({"Sentiment": order, "Count": [0] * len(order), "Grouped Stories": [0] * len(order), "Share": [0.0] * len(order)})

    labels = working[column].fillna("").astype(str).str.strip().str.upper()
    processed = labels.ne("")
    working = working.loc[processed].copy()
    labels = labels.loc[processed]
    group_counts = _group_count_series(working)

    distribution = pd.DataFrame({"Sentiment": labels, "Count": group_counts})
    count_df = distribution.groupby("Sentiment", dropna=False)["Count"].sum().reset_index()
    group_df = labels.value_counts().rename_axis("Sentiment").reset_index(name="Grouped Stories")

    base = pd.DataFrame({"Sentiment": order})
    out = base.merge(count_df, on="Sentiment", how="left").merge(group_df, on="Sentiment", how="left")
    out["Count"] = out["Count"].fillna(0).astype(int)
    out["Grouped Stories"] = out["Grouped Stories"].fillna(0).astype(int)
    total = int(out["Count"].sum())
    out["Share"] = out["Count"] / total if total > 0 else 0.0
    return out


def build_jev_best_tag_distribution(df_unique: pd.DataFrame) -> pd.DataFrame:
    if df_unique is None or df_unique.empty or "Jev Best Tag" not in df_unique.columns:
        return pd.DataFrame(columns=["Tag", "Count", "Grouped Stories", "Share"])

    working = df_unique.copy()
    tags = working["Jev Best Tag"].fillna("").astype(str).str.strip()
    processed = tags.ne("")
    working = working.loc[processed].copy()
    tags = tags.loc[processed]
    if working.empty:
        return pd.DataFrame(columns=["Tag", "Count", "Grouped Stories", "Share"])

    group_counts = _group_count_series(working)
    count_df = pd.DataFrame({"Tag": tags, "Count": group_counts}).groupby("Tag", dropna=False)["Count"].sum().reset_index()
    group_df = tags.value_counts().rename_axis("Tag").reset_index(name="Grouped Stories")
    out = count_df.merge(group_df, on="Tag", how="left")
    out["Grouped Stories"] = out["Grouped Stories"].fillna(0).astype(int)
    out["Count"] = out["Count"].fillna(0).astype(int)
    out = out.sort_values(["Count", "Grouped Stories", "Tag"], ascending=[False, False, True]).reset_index(drop=True)
    total = int(out["Count"].sum())
    out["Share"] = out["Count"] / total if total > 0 else 0.0
    return out


def build_jev_all_applicable_tag_distribution(
    df_unique: pd.DataFrame,
    *,
    tag_definitions: dict[str, str] | None,
) -> pd.DataFrame:
    mapping = build_jev_tag_question_map(tag_definitions)
    if df_unique is None or df_unique.empty or not mapping:
        return pd.DataFrame(columns=["Tag", "Count", "Grouped Stories", "Processed Grouped Stories", "Grouped Story Share"])

    working = ensure_jev_sentiment_columns(df_unique, tag_definitions=tag_definitions)
    processed_mask = pd.Series(True, index=working.index)
    for item in mapping:
        column = f"Jev Tag [{item['tag']}]"
        processed_mask = processed_mask & working[column].fillna("").astype(str).str.strip().ne("")
    working = working.loc[processed_mask].copy()
    if working.empty:
        return pd.DataFrame(columns=["Tag", "Count", "Grouped Stories", "Processed Grouped Stories", "Grouped Story Share"])

    processed_grouped_stories = len(working)
    tag_counter: dict[str, int] = {}
    grouped_story_counter: dict[str, int] = {}
    for _, row in working.iterrows():
        group_count = int(pd.to_numeric(pd.Series([row.get("Group Count", 1)]), errors="coerce").fillna(1).iloc[0] or 1)
        yes_tags = [
            item["tag"]
            for item in mapping
            if str(row.get(f"Jev Tag [{item['tag']}]", "") or "").strip().upper() == "YES"
        ]
        tags = yes_tags or [RESERVED_OTHER_TAG]
        for tag in tags:
            tag_counter[tag] = tag_counter.get(tag, 0) + group_count
            grouped_story_counter[tag] = grouped_story_counter.get(tag, 0) + 1

    out = pd.Series(tag_counter).rename_axis("Tag").reset_index(name="Count")
    out["Grouped Stories"] = out["Tag"].map(grouped_story_counter).fillna(0).astype(int)
    out["Processed Grouped Stories"] = processed_grouped_stories
    out["Grouped Story Share"] = out["Grouped Stories"] / processed_grouped_stories if processed_grouped_stories else 0.0
    return out.sort_values(["Count", "Grouped Stories", "Tag"], ascending=[False, False, True]).reset_index(drop=True)


def filter_jev_sentiment_distribution(
    distribution: pd.DataFrame,
    *,
    order: list[str],
    include_not_relevant: bool,
) -> pd.DataFrame:
    working = distribution.copy()
    if not include_not_relevant:
        working = working[working["Sentiment"] != "NOT RELEVANT"].copy()
        order = [item for item in order if item != "NOT RELEVANT"]
    total = int(pd.to_numeric(working.get("Count", 0), errors="coerce").fillna(0).sum())
    working["Share"] = working["Count"] / total if total > 0 else 0.0
    working["Sentiment"] = pd.Categorical(working["Sentiment"], categories=order, ordered=True)
    return working.sort_values("Sentiment").reset_index(drop=True)


def filter_jev_best_tag_distribution(distribution: pd.DataFrame, *, include_other: bool) -> pd.DataFrame:
    working = distribution.copy()
    if not include_other:
        working = working[working["Tag"].fillna("").astype(str).str.casefold() != RESERVED_OTHER_TAG.casefold()].copy()
    total = int(pd.to_numeric(working.get("Count", 0), errors="coerce").fillna(0).sum())
    working["Share"] = working["Count"] / total if total > 0 else 0.0
    return order_jev_tag_distribution_other_last(working)


def filter_jev_all_applicable_tag_distribution(distribution: pd.DataFrame, *, include_other: bool) -> pd.DataFrame:
    working = distribution.copy()
    if not include_other:
        working = working[working["Tag"].fillna("").astype(str).str.casefold() != RESERVED_OTHER_TAG.casefold()].copy()
    return order_jev_tag_distribution_other_last(working)


def order_jev_tag_distribution_other_last(distribution: pd.DataFrame) -> pd.DataFrame:
    if distribution is None or distribution.empty or "Tag" not in distribution.columns:
        return distribution.copy() if distribution is not None else pd.DataFrame()
    working = distribution.copy()
    is_other = working["Tag"].fillna("").astype(str).str.casefold() == RESERVED_OTHER_TAG.casefold()
    return pd.concat([working.loc[~is_other], working.loc[is_other]], ignore_index=True)


def build_jev_tag_color_scale(tags: list[str], *, include_other: bool = True) -> tuple[list[str], list[str]]:
    domain: list[str] = []
    seen: set[str] = set()
    for tag in tags:
        label = str(tag or "").strip()
        if not label:
            continue
        key = label.casefold()
        if key == RESERVED_OTHER_TAG.casefold():
            continue
        if key in seen:
            continue
        seen.add(key)
        domain.append(label)
    if include_other:
        domain.append(RESERVED_OTHER_TAG)

    colors = [
        JEV_TAG_COLOR_PALETTE[index % len(JEV_TAG_COLOR_PALETTE)]
        for index, _ in enumerate(domain)
    ]
    if include_other and colors:
        colors[-1] = JEV_OTHER_TAG_COLOR
    return domain, colors


def _group_count_series(df: pd.DataFrame) -> pd.Series:
    return (
        pd.to_numeric(df.get("Group Count", pd.Series(1, index=df.index)), errors="coerce")
        .fillna(1)
        .clip(lower=0)
        .astype(int)
    )


def _clean_list(values: list[str] | None) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for value in values or []:
        text = _safe_text(value)
        if not text:
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(text)
    return cleaned


def _safe_json_scalar(value: Any) -> str:
    try:
        missing = pd.isna(value)
    except Exception:
        missing = False
    if isinstance(missing, bool) and missing:
        return ""
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            pass
    return str(value).strip()


def _safe_text(value: Any) -> str:
    try:
        missing = pd.isna(value)
    except Exception:
        missing = False
    if isinstance(missing, bool) and missing:
        return ""
    return str(value).strip()


def _coerce_probability(value: Any) -> float:
    numeric = _coerce_float(value)
    if numeric is None:
        return 0.0
    return float(max(0.0, min(1.0, numeric)))


def _score_level_label(description: str) -> str:
    text = str(description or "").strip()
    if not text:
        return ""
    return text.split(":", 1)[0].strip()


def _match_allowed_label(value: Any, allowed_labels: list[str]) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    by_key = {label.casefold(): label for label in allowed_labels}
    return by_key.get(raw.casefold(), "")


def _coerce_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        numeric = float(value)
        if pd.isna(numeric):
            return None
        return numeric
    except Exception:
        return None


def _parse_retry_after(response: requests.Response) -> float | None:
    for header in ["retry-after-ms", "Retry-After-Ms"]:
        value = response.headers.get(header)
        numeric = _coerce_float(value)
        if numeric is not None:
            return max(0.0, numeric / 1000.0)
    for header in ["retry-after", "Retry-After"]:
        value = response.headers.get(header)
        numeric = _coerce_float(value)
        if numeric is not None:
            return max(0.0, numeric)
    return None
