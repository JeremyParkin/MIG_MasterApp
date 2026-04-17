from __future__ import annotations

import json
import re
import unicodedata
from typing import Any

from openai import OpenAI

from utils.api_meter import add_api_usage, extract_usage_tokens


DEFAULT_ANALYSIS_CONTEXT_MODEL = "gpt-5.4-mini"

ORG_LIKE_MARKERS = {
    "agency",
    "alliance",
    "association",
    "board",
    "bureau",
    "campaign",
    "city",
    "commission",
    "committee",
    "council",
    "department",
    "destination",
    "division",
    "foundation",
    "government",
    "group",
    "industry",
    "initiative",
    "institute",
    "media",
    "ministry",
    "network",
    "office",
    "organization",
    "province",
    "region",
    "secretariat",
    "society",
    "strategy",
    "tourism",
}


def _match_key(value: str) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.casefold()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _clean_list(values: list[str] | None) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values or []:
        cleaned = str(value or "").strip()
        if not cleaned:
            continue
        key = _match_key(cleaned)
        if key in seen:
            continue
        seen.add(key)
        out.append(cleaned)
    return out


def _clean_suggestion_items(items: list[dict[str, Any]] | None) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in items or []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "") or "").strip()
        detail = str(item.get("detail", "") or "").strip()
        if not name:
            continue
        key = _match_key(name)
        if key in seen:
            continue
        seen.add(key)
        out.append({"name": name, "detail": detail})
    return out


def _looks_like_person_name(value: str) -> bool:
    words = [part for part in re.split(r"\s+", str(value or "").strip()) if part]
    if len(words) < 2 or len(words) > 4:
        return False
    lowered = {_match_key(word) for word in words}
    if any(word in ORG_LIKE_MARKERS for word in lowered):
        return False
    alpha_chars = sum(1 for ch in value if ch.isalpha())
    non_space_chars = sum(1 for ch in value if not ch.isspace())
    if alpha_chars == 0 or non_space_chars == 0:
        return False
    return alpha_chars / non_space_chars >= 0.7


def _filter_alias_items(
    items: list[dict[str, str]],
    client_name: str,
    primary_name: str,
) -> list[dict[str, str]]:
    client_key = _match_key(client_name)
    primary_key = _match_key(primary_name)
    primary_tokens = {token for token in primary_key.split() if len(token) >= 4}
    filtered: list[dict[str, str]] = []

    for item in items:
        name = item["name"]
        key = _match_key(name)
        if not key or key in {client_key, primary_key}:
            continue
        if len(key.split()) == 1 and key in primary_tokens:
            continue
        filtered.append(item)

    return filtered


def _extract_response_text(response) -> str:
    if hasattr(response, "output_text") and response.output_text:
        return response.output_text

    try:
        parts = []
        for item in response.output:
            if getattr(item, "type", None) != "message":
                continue
            for content in getattr(item, "content", []):
                if getattr(content, "type", None) == "output_text":
                    parts.append(content.text)
        return "\n".join(parts).strip()
    except Exception:
        return ""


def build_analysis_context_discovery_prompt(
    client_name: str,
    primary_name: str,
    alternate_names: list[str] | None = None,
    spokespeople: list[str] | None = None,
    products: list[str] | None = None,
    guidance: str = "",
) -> str:
    client = str(client_name or "").strip() or "[Client name]"
    primary = str(primary_name or "").strip() or client
    aliases_text = "; ".join(_clean_list(alternate_names))
    spokespeople_text = "; ".join(_clean_list(spokespeople))
    products_text = "; ".join(_clean_list(products))
    guidance_text = str(guidance or "").strip()
    return (
        "Help prepare analysis-context inputs for a media-analysis workflow.\n\n"
        f"Client / organization: {client}\n"
        f"Primary topic or entity of interest: {primary}\n"
        f"Existing alternate names / aliases: {aliases_text or '[none yet]'}\n"
        f"Existing key spokespeople: {spokespeople_text or '[none yet]'}\n"
        f"Existing products / sub-brands / initiatives: {products_text or '[none yet]'}\n"
        f"Existing analytical guidance: {guidance_text or '[none yet]'}\n\n"
        "Task:\n"
        "- Suggest practical additional items for alternate names / aliases, key spokespeople, and products / sub-brands / initiatives.\n"
        "- Be selective rather than exhaustive.\n"
        "- Aim for stability and high confidence rather than variety.\n"
        "- Prefer public-facing names that are likely to appear in media coverage.\n"
        "- Avoid duplicates of existing items.\n"
        "- If a category is weak or uncertain, keep it sparse and say so in the assessment.\n"
        "- Preserve native spelling / accents when they are part of the real public-facing name.\n"
        "- Do not repeat the exact client name or the exact primary topic as an alias unless there is a materially different public-facing variant.\n"
        "- Keep categories cleanly separated.\n"
        "- Do not force an item into a bucket just because it seems related.\n\n"
        "Output discipline:\n"
        "- Prefer 0 to 5 items per category.\n"
        "- Order items from strongest/most useful to weakest.\n"
        "- If a plausible item is not clearly tied to the client/topic, leave it out.\n\n"
        "Category definitions:\n"
        "- Alternate names / aliases: alternate public-facing names, abbreviations, or common coverage labels for the same entity/topic.\n"
        "  Use this only for true alternate references to the exact client/topic.\n"
        "  Do not include broader place names, adjacent organizations, partner groups, destinations, departments, or generic language translations unless they are genuinely used as alternate labels for the same thing.\n"
        "- Key spokespeople: actual named people who are likely to be quoted or mentioned as representatives.\n"
        "  Do not include organizations, brands, destinations, ministries, agencies, offices, or unnamed roles by themselves.\n"
        "- Products / sub-brands / initiatives: named programs, campaigns, product lines, initiatives, or sub-brands directly tied to the client/topic.\n"
        "  Do not include unrelated organizations, umbrella industry groups, or broad place names unless they function as a real branded initiative.\n\n"
        "Bad bucket examples to avoid:\n"
        '- Do not put organization names like "Alliance de l’industrie touristique du Québec" or destination labels like "Destination Québec cité" into Key spokespeople.\n'
        '- Do not put broad labels like "Quebec" or "Québec" into Alternate names / aliases just because they overlap with the topic.\n'
        '- Do not put the same item into multiple categories unless there is a truly compelling reason, which is rare here.\n\n'
        "When uncertain:\n"
        "- prefer fewer, higher-confidence items\n"
        "- explain uncertainty in the assessment rather than padding the lists\n"
    )


def generate_analysis_context_suggestions(
    client_name: str,
    primary_name: str,
    alternate_names: list[str],
    spokespeople: list[str],
    products: list[str],
    guidance: str,
    api_key: str,
    model: str = DEFAULT_ANALYSIS_CONTEXT_MODEL,
) -> tuple[dict[str, Any], int, int]:
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "assessment": {"type": "string"},
            "aliases": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "name": {"type": "string"},
                        "detail": {"type": "string"},
                    },
                    "required": ["name", "detail"],
                },
            },
            "spokespeople": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "name": {"type": "string"},
                        "detail": {"type": "string"},
                    },
                    "required": ["name", "detail"],
                },
            },
            "products": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "name": {"type": "string"},
                        "detail": {"type": "string"},
                    },
                    "required": ["name", "detail"],
                },
            },
        },
        "required": ["assessment", "aliases", "spokespeople", "products"],
    }

    client = OpenAI(api_key=api_key)
    prompt = build_analysis_context_discovery_prompt(
        client_name=client_name,
        primary_name=primary_name,
        alternate_names=alternate_names,
        spokespeople=spokespeople,
        products=products,
        guidance=guidance,
    )
    response = client.responses.create(
        model=model,
        temperature=0,
        input=[
            {
                "role": "system",
                "content": (
                    "You are a precise media intelligence analyst. "
                    "Return only structured JSON that helps configure analysis context fields."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        text={
            "verbosity": "low",
            "format": {
                "type": "json_schema",
                "name": "analysis_context_suggestions",
                "strict": True,
                "schema": schema,
            },
        },
    )

    add_api_usage(response, model)
    in_tok, out_tok = extract_usage_tokens(response)
    raw_text = _extract_response_text(response).strip()
    if not raw_text:
        raise ValueError("No structured output text was returned.")

    parsed = json.loads(raw_text)
    cleaned_aliases = _filter_alias_items(
        _clean_suggestion_items(parsed.get("aliases", [])),
        client_name=client_name,
        primary_name=primary_name,
    )
    cleaned_spokespeople = [
        item
        for item in _clean_suggestion_items(parsed.get("spokespeople", []))
        if _looks_like_person_name(item["name"])
    ]
    cleaned_products = _clean_suggestion_items(parsed.get("products", []))

    return {
        "assessment": str(parsed.get("assessment", "") or "").strip(),
        "aliases": cleaned_aliases,
        "spokespeople": cleaned_spokespeople,
        "products": cleaned_products,
    }, in_tok, out_tok


def apply_analysis_context_suggestions(session_state, suggestions: dict[str, Any]) -> None:
    payload = get_analysis_context_payload(session_state)
    client_key = _match_key(payload["client_name"])
    primary_key = _match_key(payload["primary_name"])

    alias_names = [
        item["name"]
        for item in _clean_suggestion_items(suggestions.get("aliases", []))
        if _match_key(item["name"]) not in {client_key, primary_key}
    ]
    spokesperson_names = [
        item["name"]
        for item in _clean_suggestion_items(suggestions.get("spokespeople", []))
        if " " in item["name"].strip()
    ]
    product_names = [
        item["name"]
        for item in _clean_suggestion_items(suggestions.get("products", []))
        if _match_key(item["name"]) not in {client_key, primary_key}
    ]

    merged_aliases = _clean_list(payload["alternate_names"] + alias_names)
    merged_spokespeople = _clean_list(payload["spokespeople"] + spokesperson_names)
    merged_products = _clean_list(payload["products"] + product_names)

    session_state.analysis_alternate_names = merged_aliases
    session_state.analysis_spokespeople = merged_spokespeople
    session_state.analysis_products = merged_products
    session_state.ui_alternate_names = list(merged_aliases)
    session_state.ui_spokespeople = list(merged_spokespeople)
    session_state.ui_products = list(merged_products)
    session_state.analysis_context_aliases_tags = list(merged_aliases)
    session_state.analysis_context_spokespeople_tags = list(merged_spokespeople)
    session_state.analysis_context_products_tags = list(merged_products)
    session_state.analysis_context_suggestion_payload = suggestions


def init_analysis_context_state(session_state) -> None:
    client_name = str(session_state.get("client_name", "") or "").strip()

    primary_seed = []
    for candidate in [
        session_state.get("analysis_primary_names"),
        session_state.get("ui_primary_names"),
        session_state.get("top_story_entity_names"),
        [client_name] if client_name else [],
    ]:
        cleaned = _clean_list(candidate if isinstance(candidate, list) else [])
        if cleaned:
            primary_seed = cleaned
            break

    alternate_seed = _clean_list(
        session_state.get("analysis_alternate_names")
        or session_state.get("ui_alternate_names")
        or []
    )
    spokes_seed = _clean_list(
        session_state.get("analysis_spokespeople")
        or session_state.get("ui_spokespeople")
        or session_state.get("top_story_spokespeople")
        or []
    )
    products_seed = _clean_list(
        session_state.get("analysis_products")
        or session_state.get("ui_products")
        or session_state.get("top_story_products")
        or []
    )
    guidance_seed = str(
        session_state.get("analysis_guidance")
        or session_state.get("ui_toning_rationale")
        or session_state.get("top_story_guidance")
        or ""
    ).strip()

    session_state.setdefault("analysis_primary_names", primary_seed)
    session_state.setdefault("analysis_alternate_names", alternate_seed)
    session_state.setdefault("analysis_spokespeople", spokes_seed)
    session_state.setdefault("analysis_products", products_seed)
    session_state.setdefault("analysis_guidance", guidance_seed)


def save_analysis_context(
    session_state,
    client_name: str,
    primary_name: str,
    alternate_names: list[str],
    spokespeople: list[str],
    products: list[str],
    guidance: str,
) -> None:
    client_name = str(client_name or "").strip()
    primary_names = _clean_list([primary_name])
    alternate_names = _clean_list(alternate_names)
    spokespeople = _clean_list(spokespeople)
    products = _clean_list(products)
    guidance = str(guidance or "").strip()

    session_state.client_name = client_name
    session_state.analysis_primary_names = primary_names
    session_state.analysis_alternate_names = alternate_names
    session_state.analysis_spokespeople = spokespeople
    session_state.analysis_products = products
    session_state.analysis_guidance = guidance

    # Keep legacy workflow keys in sync so existing flows continue to work.
    session_state.ui_primary_names = list(primary_names)
    session_state.ui_alternate_names = list(alternate_names)
    session_state.ui_spokespeople = list(spokespeople)
    session_state.ui_products = list(products)
    session_state.ui_toning_rationale = guidance

    session_state.top_story_entity_names = list(primary_names + alternate_names)
    session_state.top_story_spokespeople = list(spokespeople)
    session_state.top_story_products = list(products)
    session_state.top_story_guidance = guidance


def get_analysis_context_payload(session_state) -> dict[str, Any]:
    init_analysis_context_state(session_state)
    client_name = str(session_state.get("client_name", "") or "").strip()
    primary_names = _clean_list(session_state.get("analysis_primary_names", []))
    if not primary_names and client_name:
        primary_names = [client_name]

    return {
        "client_name": client_name,
        "primary_name": primary_names[0] if primary_names else "",
        "alternate_names": _clean_list(session_state.get("analysis_alternate_names", [])),
        "spokespeople": _clean_list(session_state.get("analysis_spokespeople", [])),
        "products": _clean_list(session_state.get("analysis_products", [])),
        "guidance": str(session_state.get("analysis_guidance", "") or "").strip(),
    }


def build_analysis_context_text(session_state) -> str:
    payload = get_analysis_context_payload(session_state)
    lines: list[str] = []

    if payload["client_name"]:
        lines.append(f"Client / organization: {payload['client_name']}")
    if payload["primary_name"]:
        lines.append(f"Primary topic or entity of interest: {payload['primary_name']}")
    if payload["alternate_names"]:
        lines.append("Alternate names / aliases: " + "; ".join(payload["alternate_names"]))
    if payload["spokespeople"]:
        lines.append("Key spokespeople: " + "; ".join(payload["spokespeople"]))
    if payload["products"]:
        lines.append("Products / sub-brands / initiatives: " + "; ".join(payload["products"]))
    if payload["guidance"]:
        lines.append("Additional analytical guidance: " + payload["guidance"])

    return "\n".join(lines).strip()


def build_analysis_context_caption(session_state) -> str:
    payload = get_analysis_context_payload(session_state)
    parts: list[str] = []
    if payload["client_name"]:
        parts.append(f"Client: {payload['client_name']}")
    if payload["primary_name"]:
        parts.append(f"Focus: {payload['primary_name']}")
    if payload["alternate_names"]:
        parts.append(f"Aliases: {len(payload['alternate_names'])}")
    if payload["spokespeople"]:
        parts.append(f"Spokespeople: {len(payload['spokespeople'])}")
    if payload["products"]:
        parts.append(f"Products / initiatives: {len(payload['products'])}")
    return " | ".join(parts)
