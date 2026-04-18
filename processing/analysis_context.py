from __future__ import annotations

import json
import re
import unicodedata
import hashlib
from typing import Any

from openai import OpenAI
import pandas as pd

from utils.api_meter import add_api_usage, extract_usage_tokens


DEFAULT_ANALYSIS_CONTEXT_MODEL = "gpt-5.4-mini"
AVAILABLE_JUNKY_COVERAGE_FLAGS = [
    "Press Release",
    "Advertorial",
    "Financial Outlet",
    "Market Report Spam",
    "User-Generated",
]
DEFAULT_QUALITATIVE_COVERAGE_FLAGS = [
    "Press Release",
    "Advertorial",
    "Market Report Spam",
    "User-Generated",
]
DEFAULT_DATASET_COVERAGE_FLAGS: list[str] = []
OUTLET_INSIGHT_AGGREGATOR_FLAG = "Aggregator"

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


def split_coverage_flags(value: object) -> list[str]:
    raw = str(value or "").strip()
    if not raw:
        return []
    parts = re.split(r"[;,]\s*|\|\s*", raw)
    return [part.strip() for part in parts if part.strip()]


def get_present_junky_coverage_flags(df_rows: pd.DataFrame | None) -> list[str]:
    if df_rows is None or df_rows.empty or "Coverage Flags" not in df_rows.columns:
        return []

    present: set[str] = set()
    for value in df_rows["Coverage Flags"].tolist():
        for flag in split_coverage_flags(value):
            if flag in AVAILABLE_JUNKY_COVERAGE_FLAGS:
                present.add(flag)
    return [flag for flag in AVAILABLE_JUNKY_COVERAGE_FLAGS if flag in present]


def apply_coverage_flag_policy(
    df_rows: pd.DataFrame,
    excluded_flags: list[str] | None = None,
    keep_row_keys: set[str] | None = None,
) -> pd.DataFrame:
    if df_rows is None or df_rows.empty:
        return pd.DataFrame()

    blocked_flags = {str(flag).strip() for flag in excluded_flags or [] if str(flag).strip()}
    if not blocked_flags or "Coverage Flags" not in df_rows.columns:
        return df_rows.copy().reset_index(drop=True)

    filtered = df_rows.copy()
    keep_row_keys = {str(key).strip() for key in keep_row_keys or set() if str(key).strip()}
    filtered["_coverage_row_key"] = build_coverage_row_key_series(filtered)
    filtered["_flag_blocked"] = filtered["Coverage Flags"].apply(
        lambda value: any(flag in blocked_flags for flag in split_coverage_flags(value))
    )
    filtered = filtered[
        ~(
            filtered["_flag_blocked"]
            & ~filtered["_coverage_row_key"].isin(keep_row_keys)
        )
    ].copy()
    filtered = filtered.drop(columns=["_coverage_row_key", "_flag_blocked"], errors="ignore")
    return filtered.reset_index(drop=True)


def build_coverage_row_key_series(df_rows: pd.DataFrame) -> pd.Series:
    if df_rows is None or df_rows.empty:
        return pd.Series(dtype="object")

    def _row_key(row: pd.Series) -> str:
        raw = "||".join(
            [
                str(row.get("URL", "") or "").strip(),
                str(row.get("Headline", "") or "").strip(),
                str(row.get("Outlet", "") or "").strip(),
                str(row.get("Coverage Flags", "") or "").strip(),
            ]
        )
        return hashlib.sha1(raw.encode("utf-8", errors="ignore")).hexdigest()

    return df_rows.apply(_row_key, axis=1)


def build_coverage_flag_removal_preview(
    df_rows: pd.DataFrame,
    excluded_flags: list[str] | None,
    keep_row_keys: set[str] | None = None,
) -> dict[str, Any]:
    if df_rows is None or df_rows.empty:
        empty = pd.DataFrame()
        return {
            "removed_rows": 0,
            "removed_mentions": 0,
            "counts_df": empty,
            "sample_df": empty,
        }

    working = df_rows.copy()
    if "Coverage Flags" not in working.columns:
        empty = pd.DataFrame()
        return {
            "removed_rows": 0,
            "removed_mentions": 0,
            "counts_df": empty,
            "sample_df": empty,
        }

    blocked_flags = {str(flag).strip() for flag in excluded_flags or [] if str(flag).strip()}
    if not blocked_flags:
        empty = pd.DataFrame()
        return {
            "removed_rows": 0,
            "removed_mentions": 0,
            "counts_df": empty,
            "sample_df": empty,
        }

    working["_row_key"] = build_coverage_row_key_series(working)
    keep_row_keys = {str(key).strip() for key in keep_row_keys or set() if str(key).strip()}
    working["_matched_flags"] = working["Coverage Flags"].apply(
        lambda value: [flag for flag in split_coverage_flags(value) if flag in blocked_flags]
    )
    removed = working[
        working["_matched_flags"].map(bool)
        & ~working["_row_key"].isin(keep_row_keys)
    ].copy()
    if removed.empty:
        empty = pd.DataFrame()
        return {
            "removed_rows": 0,
            "removed_mentions": 0,
            "counts_df": empty,
            "sample_df": empty,
        }

    mentions = pd.to_numeric(removed.get("Mentions", pd.Series(index=removed.index, data=0)), errors="coerce").fillna(0)
    flag_counts: dict[str, dict[str, int]] = {}
    for _, row in removed.iterrows():
        row_mentions = int(pd.to_numeric(pd.Series([row.get("Mentions", 0)]), errors="coerce").fillna(0).iloc[0])
        for flag in row["_matched_flags"]:
            bucket = flag_counts.setdefault(flag, {"Rows": 0, "Mentions": 0})
            bucket["Rows"] += 1
            bucket["Mentions"] += row_mentions

    counts_df = (
        pd.DataFrame(
            [
                {"Coverage Flag": flag, "Rows": values["Rows"], "Mentions": values["Mentions"]}
                for flag, values in flag_counts.items()
            ]
        )
        .sort_values(["Rows", "Mentions", "Coverage Flag"], ascending=[False, False, True])
        .reset_index(drop=True)
    )

    sample_columns = [
        "_row_key",
        "Headline",
        "Outlet",
        "Coverage Flags",
        "URL",
    ]
    sample_existing = [col for col in sample_columns if col in removed.columns]
    sample_df = removed[sample_existing].copy().reset_index(drop=True)
    sample_df = sample_df.rename(columns={"_row_key": "Row Key", "URL": "Link"})
    sample_df["Remove"] = True

    return {
        "removed_rows": int(len(removed)),
        "removed_mentions": int(mentions.sum()),
        "counts_df": counts_df,
        "sample_df": sample_df,
    }


def get_dataset_coverage_flag_exclusions(session_state) -> list[str]:
    return _clean_list(session_state.get("analysis_dataset_excluded_flags", []))


def get_dataset_coverage_keep_keys(session_state) -> set[str]:
    return {
        str(value).strip()
        for value in session_state.get("analysis_dataset_exclusion_keep_keys", []) or []
        if str(value).strip()
    }


def get_qualitative_coverage_keep_keys(session_state) -> set[str]:
    return {
        str(value).strip()
        for value in session_state.get("analysis_qualitative_exclusion_keep_keys", []) or []
        if str(value).strip()
    }


def get_qualitative_coverage_flag_exclusions(session_state) -> list[str]:
    flags = set(get_dataset_coverage_flag_exclusions(session_state))
    flags.update(_clean_list(session_state.get("analysis_qualitative_excluded_flags", [])))
    return sorted(flags)


def get_outlet_insight_coverage_flag_exclusions(session_state) -> list[str]:
    flags = set(get_qualitative_coverage_flag_exclusions(session_state))
    if bool(session_state.get("analysis_exclude_aggregators_from_outlet_insights", True)):
        flags.add(OUTLET_INSIGHT_AGGREGATOR_FLAG)
    return sorted(flags)


def apply_session_coverage_flag_policy(
    df_rows: pd.DataFrame,
    session_state,
    excluded_flags: list[str] | None,
) -> pd.DataFrame:
    keep_row_keys = get_dataset_coverage_keep_keys(session_state) | get_qualitative_coverage_keep_keys(session_state)
    return apply_coverage_flag_policy(
        df_rows,
        excluded_flags=excluded_flags,
        keep_row_keys=keep_row_keys,
    )


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
    present_junky_flags = get_present_junky_coverage_flags(session_state.get("df_traditional"))

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
    session_state.setdefault("analysis_exclude_aggregators_from_outlet_insights", True)
    session_state.setdefault("analysis_qualitative_exclusion_keep_keys", [])
    session_state.setdefault("analysis_dataset_exclusion_keep_keys", [])
    if "analysis_qualitative_excluded_flags" not in session_state:
        default_qualitative_flags = [
            flag for flag in DEFAULT_QUALITATIVE_COVERAGE_FLAGS
            if (not present_junky_flags) or flag in present_junky_flags
        ]
        if session_state.get("analysis_exclude_junky_from_qualitative", None) is True:
            session_state.analysis_qualitative_excluded_flags = default_qualitative_flags
        else:
            session_state.analysis_qualitative_excluded_flags = default_qualitative_flags
    if "analysis_dataset_excluded_flags" not in session_state:
        if session_state.get("analysis_exclude_junky_from_dataset", None) is True:
            session_state.analysis_dataset_excluded_flags = [
                flag for flag in ["Press Release", "Financial Outlet", "Advertorial"]
                if ((not present_junky_flags) or flag in present_junky_flags) and flag in AVAILABLE_JUNKY_COVERAGE_FLAGS
            ]
        else:
            session_state.analysis_dataset_excluded_flags = DEFAULT_DATASET_COVERAGE_FLAGS.copy()


def save_analysis_context(
    session_state,
    client_name: str,
    primary_name: str,
    alternate_names: list[str],
    spokespeople: list[str],
    products: list[str],
    guidance: str,
    qualitative_excluded_flags: list[str],
    dataset_excluded_flags: list[str],
    exclude_aggregators_from_outlet_insights: bool,
    qualitative_exclusion_keep_keys: list[str] | None = None,
    dataset_exclusion_keep_keys: list[str] | None = None,
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
    session_state.analysis_qualitative_excluded_flags = [
        flag for flag in _clean_list(qualitative_excluded_flags) if flag in AVAILABLE_JUNKY_COVERAGE_FLAGS
    ]
    session_state.analysis_dataset_excluded_flags = [
        flag for flag in _clean_list(dataset_excluded_flags) if flag in AVAILABLE_JUNKY_COVERAGE_FLAGS
    ]
    session_state.analysis_exclude_aggregators_from_outlet_insights = bool(exclude_aggregators_from_outlet_insights)
    session_state.analysis_qualitative_exclusion_keep_keys = _clean_list(qualitative_exclusion_keep_keys or [])
    session_state.analysis_dataset_exclusion_keep_keys = _clean_list(dataset_exclusion_keep_keys or [])

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
    present_junky_flags = get_present_junky_coverage_flags(session_state.get("df_traditional"))
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
        "available_junky_flags": present_junky_flags,
        "qualitative_excluded_flags": [
            flag for flag in _clean_list(session_state.get("analysis_qualitative_excluded_flags", DEFAULT_QUALITATIVE_COVERAGE_FLAGS.copy()))
            if flag in present_junky_flags
        ],
        "dataset_excluded_flags": [
            flag for flag in _clean_list(session_state.get("analysis_dataset_excluded_flags", DEFAULT_DATASET_COVERAGE_FLAGS.copy()))
            if flag in present_junky_flags
        ],
        "exclude_aggregators_from_outlet_insights": bool(
            session_state.get("analysis_exclude_aggregators_from_outlet_insights", True)
        ),
        "outlet_insight_excluded_flags": get_outlet_insight_coverage_flag_exclusions(session_state),
        "qualitative_exclusion_keep_keys": _clean_list(session_state.get("analysis_qualitative_exclusion_keep_keys", [])),
        "dataset_exclusion_keep_keys": _clean_list(session_state.get("analysis_dataset_exclusion_keep_keys", [])),
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
    if payload["qualitative_excluded_flags"]:
        lines.append(
            "Coverage flag rule for qualitative workflows: exclude "
            + "; ".join(payload["qualitative_excluded_flags"])
        )
        if payload["qualitative_exclusion_keep_keys"]:
            lines.append(
                f"Qualitative exclusion exceptions kept: {len(payload['qualitative_exclusion_keep_keys'])}"
            )
    if payload["dataset_excluded_flags"]:
        lines.append(
            "Coverage flag rule for dataset-wide exclusion: remove "
            + "; ".join(payload["dataset_excluded_flags"])
        )
        if payload["dataset_exclusion_keep_keys"]:
            lines.append(
                f"Dataset exclusion exceptions kept: {len(payload['dataset_exclusion_keep_keys'])}"
            )
    if payload["exclude_aggregators_from_outlet_insights"]:
        lines.append("Outlet insights rule: exclude Aggregator coverage from outlet charts and narrative")

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
    if payload["qualitative_excluded_flags"]:
        parts.append("Junky flags excluded from qualitative workflows")
    if payload["dataset_excluded_flags"]:
        parts.append("Junky flags removed from dataset")
    if payload["exclude_aggregators_from_outlet_insights"]:
        parts.append("Aggregators excluded from Outlet insights")
    return " | ".join(parts)
