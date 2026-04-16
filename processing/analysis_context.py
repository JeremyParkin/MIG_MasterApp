from __future__ import annotations

from typing import Any


def _clean_list(values: list[str] | None) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values or []:
        cleaned = str(value or "").strip()
        if not cleaned:
            continue
        key = cleaned.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(cleaned)
    return out


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
    primary_name: str,
    alternate_names: list[str],
    spokespeople: list[str],
    products: list[str],
    guidance: str,
) -> None:
    primary_names = _clean_list([primary_name])
    alternate_names = _clean_list(alternate_names)
    spokespeople = _clean_list(spokespeople)
    products = _clean_list(products)
    guidance = str(guidance or "").strip()

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
