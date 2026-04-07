from __future__ import annotations

import math
import re
from datetime import datetime
from typing import Literal

import pandas as pd


SampleMode = Literal["full", "representative", "custom"]

# DEFAULT_SENTIMENT_SIMILARITY_THRESHOLD = 0.935
# DEFAULT_SENTIMENT_MAX_BATCH_SIZE = 1800
DEFAULT_MAX_FULL_ROWS = 2000


def init_sentiment_config_state(session_state) -> None:
    defaults = {
        "sentiment_config_step": False,
        "sentiment_sample_mode": "representative",
        "sentiment_sample_size": None,
        "sentiment_full_override": False,
        "df_sentiment_rows": pd.DataFrame(),
        "df_sentiment_grouped_rows": pd.DataFrame(),
        "df_sentiment_unique": pd.DataFrame(),
        "sentiment_elapsed_time": 0.0,
        "ui_primary_names": [session_state.get("client_name", "")] if session_state.get("client_name") else [],
        "ui_alternate_names": [],
        "ui_spokespeople": [],
        "ui_products": [],
        "ui_toning_rationale": "",
        "ui_sentiment_type": "3-way",
        "model_choice": "gpt-5.4-nano",
        "toning_config_step": False,
        "last_saved": None,
    }

    for key, value in defaults.items():
        if key not in session_state:
            session_state[key] = value


def calculate_representative_sample_size(
    population_size: int,
    confidence_level: float = 0.95,
    margin_of_error: float = 0.05,
    p: float = 0.5,
) -> int:
    if population_size <= 0:
        return 0

    z = 1.96 if confidence_level == 0.95 else 1.96

    numerator = population_size * (z ** 2) * p * (1 - p)
    denominator = (margin_of_error ** 2) * (population_size - 1) + (z ** 2) * p * (1 - p)

    return max(1, math.ceil(numerator / denominator))


def get_sentiment_source_rows(df_traditional: pd.DataFrame) -> pd.DataFrame:
    if df_traditional is None or df_traditional.empty:
        return pd.DataFrame()

    df = df_traditional.copy()

    if "Type" in df.columns:
        df = df[
            ~df["Type"].isin(
                ["FACEBOOK", "INSTAGRAM", "X", "TWITTER", "LINKEDIN", "TIKTOK", "YOUTUBE", "REDDIT", "BLUESKY"]
            )
        ].copy()

    return df.reset_index(drop=True)


def sample_sentiment_rows(
    df_rows: pd.DataFrame,
    sample_mode: SampleMode,
    custom_sample_size: int | None = None,
    max_full_rows: int = DEFAULT_MAX_FULL_ROWS,
    full_override: bool = False,
    random_state: int = 1,
) -> tuple[pd.DataFrame, int]:
    if df_rows is None or df_rows.empty:
        return pd.DataFrame(), 0

    population_size = len(df_rows)

    if sample_mode == "full":
        if population_size > max_full_rows and not full_override:
            effective_n = max_full_rows
            sampled = df_rows.sample(n=effective_n, random_state=random_state).reset_index(drop=True)
            return sampled, effective_n
        return df_rows.copy().reset_index(drop=True), population_size

    if sample_mode == "representative":
        effective_n = calculate_representative_sample_size(population_size)
        effective_n = min(effective_n, population_size)
        sampled = df_rows.sample(n=effective_n, random_state=random_state).reset_index(drop=True)
        return sampled, effective_n

    if sample_mode == "custom":
        effective_n = int(custom_sample_size or 0)
        effective_n = max(1, min(effective_n, population_size))
        sampled = df_rows.sample(n=effective_n, random_state=random_state).reset_index(drop=True)
        return sampled, effective_n

    return df_rows.copy().reset_index(drop=True), population_size

def build_unique_story_table_from_existing_groups(df_rows: pd.DataFrame) -> pd.DataFrame:
    """
    Build one-row-per-group table from sampled rows using the EXISTING canonical Group ID.
    This does not recluster.
    """
    if df_rows is None or df_rows.empty:
        return pd.DataFrame()

    if "Group ID" not in df_rows.columns:
        raise ValueError("Group ID not found. Basic Cleaning must assign canonical groups before sentiment analysis.")

    working = df_rows.copy()

    agg_dict = {}

    first_pref_cols = [
        "Headline",
        "Date",
        "Outlet",
        "Example Outlet",
        "URL",
        "Example URL",
        "Snippet",
        "Example Snippet",
        "Type",
        "Language",
        "Country",
        "Prov/State",
    ]
    for col in first_pref_cols:
        if col in working.columns:
            agg_dict[col] = "first"

    if "Mentions" in working.columns:
        agg_dict["Mentions"] = "sum"
    if "Impressions" in working.columns:
        agg_dict["Impressions"] = "sum"
    if "Effective Reach" in working.columns:
        agg_dict["Effective Reach"] = "sum"

    unique_rows = (
        working.groupby("Group ID", as_index=False)
        .agg(agg_dict)
        .copy()
    )

    if "Example Outlet" not in unique_rows.columns and "Outlet" in unique_rows.columns:
        unique_rows["Example Outlet"] = unique_rows["Outlet"]

    if "Example URL" not in unique_rows.columns and "URL" in unique_rows.columns:
        unique_rows["Example URL"] = unique_rows["URL"]

    if "Example Snippet" not in unique_rows.columns and "Snippet" in unique_rows.columns:
        unique_rows["Example Snippet"] = unique_rows["Snippet"]

    group_counts = working.groupby("Group ID").size().reset_index(name="Group Count")
    unique_rows = unique_rows.merge(group_counts, on="Group ID", how="left")

    return unique_rows.reset_index(drop=True)

def prepare_sentiment_datasets(
    df_traditional: pd.DataFrame,
    sample_mode: SampleMode,
    custom_sample_size: int | None = None,
    max_full_rows: int = DEFAULT_MAX_FULL_ROWS,
    full_override: bool = False,
    random_state: int = 1,
) -> dict:
    source_rows = get_sentiment_source_rows(df_traditional)
    sampled_rows, effective_sample_size = sample_sentiment_rows(
        source_rows,
        sample_mode=sample_mode,
        custom_sample_size=custom_sample_size,
        max_full_rows=max_full_rows,
        full_override=full_override,
        random_state=random_state,
    )

    if not sampled_rows.empty and "Group ID" not in sampled_rows.columns:
        raise ValueError("Sampled sentiment rows do not contain Group ID. Standard Cleaning must run first.")

    grouped_rows = sampled_rows.copy()
    unique_rows = build_unique_story_table_from_existing_groups(grouped_rows)

    for df_name in [grouped_rows, unique_rows]:
        for col in ["Assigned Sentiment", "AI Sentiment", "AI Sentiment Confidence", "AI Sentiment Rationale"]:
            if col not in df_name.columns:
                df_name[col] = pd.NA

    return {
        "df_sentiment_rows": sampled_rows.reset_index(drop=True),
        "df_sentiment_grouped_rows": grouped_rows.reset_index(drop=True),
        "df_sentiment_unique": unique_rows.reset_index(drop=True),
        "population_size": len(source_rows),
        "sample_size_used": effective_sample_size,
        "unique_story_count": len(unique_rows),
    }

# def prepare_sentiment_datasets(
#     df_traditional: pd.DataFrame,
#     sample_mode: SampleMode,
#     custom_sample_size: int | None = None,
#     max_full_rows: int = DEFAULT_MAX_FULL_ROWS,
#     full_override: bool = False,
#     similarity_threshold: float = DEFAULT_SENTIMENT_SIMILARITY_THRESHOLD,
#     max_batch_size: int = DEFAULT_SENTIMENT_MAX_BATCH_SIZE,
#     random_state: int = 1,
# ) -> dict:
#     source_rows = get_sentiment_source_rows(df_traditional)
#     sampled_rows, effective_sample_size = sample_sentiment_rows(
#         source_rows,
#         sample_mode=sample_mode,
#         custom_sample_size=custom_sample_size,
#         max_full_rows=max_full_rows,
#         full_override=full_override,
#         random_state=random_state,
#     )
#
#     grouped_rows = cluster_by_media_type(
#         sampled_rows.copy(),
#         similarity_threshold=similarity_threshold,
#         max_batch_size=max_batch_size,
#     )
#     unique_rows = build_unique_story_table(grouped_rows)
#
#     for df_name in [grouped_rows, unique_rows]:
#         for col in ["Assigned Sentiment", "AI Sentiment", "AI Sentiment Confidence", "AI Sentiment Rationale"]:
#             if col not in df_name.columns:
#                 df_name[col] = pd.NA
#
#     return {
#         "df_sentiment_rows": sampled_rows.reset_index(drop=True),
#         "df_sentiment_grouped_rows": grouped_rows.reset_index(drop=True),
#         "df_sentiment_unique": unique_rows.reset_index(drop=True),
#         "population_size": len(source_rows),
#         "sample_size_used": effective_sample_size,
#         "unique_story_count": len(unique_rows),
#     }


def _clean_list(lst: list[str]) -> list[str]:
    return [s.strip() for s in (lst or []) if isinstance(s, str) and s.strip()]



def _latin_char_class(ch: str) -> str:
    """
    Return a regex fragment for one character, allowing common Latin diacritic variants.
    """
    variant_map = {
        "a": "aàáâãäåāăąǎȁȃạảấầẩẫậắằẳẵặ",
        "c": "cçćĉċč",
        "d": "dďđ",
        "e": "eèéêëēĕėęěȅȇẹẻẽếềểễệ",
        "g": "gĝğġģ",
        "h": "hĥħ",
        "i": "iìíîïĩīĭįıǐȉȋịỉ",
        "j": "jĵ",
        "k": "kķ",
        "l": "lĺļľŀł",
        "n": "nñńņňŉŋ",
        "o": "oòóôõöøōŏőǒȍȏọỏốồổỗộớờởỡợ",
        "r": "rŕŗř",
        "s": "sśŝşš",
        "t": "tţťŧ",
        "u": "uùúûüũūŭůűųǔȕȗụủứừửữự",
        "w": "wŵ",
        "y": "yýÿŷȳỳỵỷỹ",
        "z": "zźżž",
    }

    lower = ch.lower()
    if lower in variant_map:
        chars = variant_map[lower]
        return f"[{re.escape(chars + chars.upper())}]"

    return re.escape(ch)


def _kw_variant_pattern(kw: str) -> str:
    """
    Build a tolerant regex for a single keyword/phrase:
      - apostrophes optional / curly-safe
      - hyphens/dashes tolerant
      - spaces tolerant
      - optional dots in acronyms
      - Latin diacritics tolerant (Nestle -> Nestlé)
    """
    APOS = r"[\'\u2019\u2032]"
    HYPH = r"[\-\u2010\u2011\u2012\u2013\u2014\u2212]"
    SPACE = r"[ \t\u00A0]+"

    hyphen_chars = "-\u2010\u2011\u2012\u2013\u2014\u2212"
    apos_chars = "'\u2019\u2032"

    out = []
    L = len(kw)
    i = 0

    while i < L:
        ch = kw[i]
        prev_ch = kw[i - 1] if i > 0 else ""
        next_ch = kw[i + 1] if i + 1 < L else ""

        if ch in apos_chars:
            out.append(f"(?:{APOS})?")

        elif ch in hyphen_chars:
            out.append(HYPH)

        elif ch.isspace():
            out.append(SPACE)

        elif ch == ".":
            out.append(r"\.?")

        elif (
            ch.lower() == "s"
            and (i + 1 == L or next_ch.isspace() or next_ch in hyphen_chars)
            and prev_ch not in apos_chars
            and prev_ch.isalnum()
        ):
            out.append(f"(?:{APOS})?{_latin_char_class('s')}")

        elif ch.isalpha():
            out.append(_latin_char_class(ch))

        else:
            out.append(re.escape(ch))

        i += 1

    return "".join(out)


def build_tolerant_regex_str(keywords: list[str]) -> str | None:
    kws = [k for k in (keywords or []) if isinstance(k, str) and k.strip()]
    if not kws:
        return None
    parts = [_kw_variant_pattern(k) for k in kws]
    return r"(?i)(?<!\w)(?:%s)(?!\w)" % "|".join(parts)


def ensure_sentiment_columns(df_grouped: pd.DataFrame, df_unique: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    grouped = df_grouped.copy()
    unique = df_unique.copy()

    for df in [grouped, unique]:
        for col in ["Assigned Sentiment", "AI Sentiment", "AI Sentiment Confidence", "AI Sentiment Rationale"]:
            if col not in df.columns:
                df[col] = pd.NA

    return grouped, unique


def build_sentiment_configuration(
    session_state,
    primary_names: list[str],
    alternate_names: list[str],
    spokespeople: list[str],
    products: list[str],
    toning_rationale: str,
    sentiment_type: str,
    model: str,
) -> None:
    session_state.ui_primary_names = _clean_list(primary_names)
    session_state.ui_alternate_names = _clean_list(alternate_names)
    session_state.ui_spokespeople = _clean_list(spokespeople)
    session_state.ui_products = _clean_list(products)
    session_state.ui_toning_rationale = toning_rationale or ""
    session_state.ui_sentiment_type = sentiment_type
    session_state.sentiment_type = sentiment_type
    session_state.model_choice = model

    named_entity = session_state.ui_primary_names[0]
    aliases = session_state.ui_alternate_names
    spokes = session_state.ui_spokespeople
    prods = session_state.ui_products
    rationale_str = session_state.ui_toning_rationale.strip() if session_state.ui_toning_rationale else None

    display_keywords = list(session_state.ui_primary_names) + aliases + spokes + prods
    seen_cf, deduped_display = set(), []
    for k in display_keywords:
        cf = k.casefold()
        if cf not in seen_cf:
            seen_cf.add(cf)
            deduped_display.append(k.strip())
    session_state.highlight_keyword = deduped_display
    session_state.highlight_regex_str = build_tolerant_regex_str(deduped_display)

    pre_lines = [
        f"PRIMARY ENTITY (brand umbrella): {named_entity}",
        "Assess sentiment toward the *collective entity* defined as:",
        "  • the primary entity,",
        "  • any listed aliases/alternate names,",
        "  • any named spokespeople when acting on behalf of the entity, and",
        "  • any listed products or sub-brands when discussed as part of the entity’s activity/portfolio.",
    ]

    if aliases:
        pre_lines += ["", "ALIASES / ALTERNATE NAMES (treat as the same entity):", ", ".join(aliases)]
    if spokes:
        pre_lines += ["", "SPOKESPEOPLE (attribute on-record statements/actions to the entity unless clearly personal/unrelated):", ", ".join(spokes)]
    if prods:
        pre_lines += ["", "PRODUCTS / SUB-BRANDS (attribute product sentiment to the parent unless clearly unrelated):", ", ".join(prods)]

    pre_lines += [
        "",
        "Judge the *net sentiment* the coverage conveys to a typical reader/viewer about the collective entity.",
        "Ignore sentiment about unrelated third parties unless the story explicitly connects it to the entity.",
    ]
    session_state.pre_prompt = "\n".join(pre_lines).strip()

    context_lines = [
        "Scope Clarifications:",
        f"- Research by {named_entity} on a negative topic is not automatically negative toward the entity.",
        "- Hosting/sponsoring an event about a negative issue is not automatically negative.",
        "- Straight factual coverage is Neutral.",
        "- Passing mentions without a strong stance are generally Neutral.",
        "- Brief/Passing Mentions: If the collective entity appears only briefly in a longer story without explicit praise/criticism or clear attribution of outcomes to the entity, default to NEUTRAL.",
        "",
        "Carry-over / Attribution Rules (collective entity):",
        f"1) When a spokesperson acts explicitly for {named_entity}, attribute their stance to the entity.",
        f"2) When a product or sub-brand is discussed, attribute sentiment to {named_entity} unless clearly unrelated.",
        "3) No sentiment transfer: Do not infer sentiment toward the entity from third parties or adjacent topics.",
        f"4) If the collective entity (primary/aliases/spokespeople-as-entity/products) is not present, label NOT RELEVANT.",
        "",
        "Tie-breakers & Gray Areas:",
        "- Use the audience takeaway as the deciding factor when positive and negative elements coexist.",
        "- Prefer explicit attributions, direct quotes, headlines, and framing to infer stance.",
    ]

    if rationale_str:
        context_lines += [
            "",
            "Analyst Guidance — PRIORITY (overrides defaults in gray areas):",
            f"- {rationale_str}",
            "If this guidance changes the default outcome, follow it and reflect that in your explanation.",
        ]

    session_state.post_prompt = "\n".join(context_lines).strip()

    if sentiment_type == "3-way":
        session_state.sentiment_instruction = f"""
LABEL SET: POSITIVE, NEUTRAL, NEGATIVE, NOT RELEVANT

WHAT TO JUDGE:
- The *collective entity*: {named_entity} + aliases + spokespeople acting on its behalf + products/sub-brands.

CRITERIA:
- POSITIVE: Praise, favorable framing, or beneficial outcomes credited to the collective entity.
- NEUTRAL: Factual/balanced coverage with no clear stance on the collective entity.
- NEGATIVE: Criticism, unfavorable framing, or negative outcomes attributed to the collective entity.
- NOT RELEVANT: The collective entity (as defined) is not present.

OUTPUT:
- Provide the UPPERCASE label, a confidence (0–100), and a 1–2 sentence explanation focused on the collective entity.
- Always output results in English.
""".strip()

        session_state.functions = [{
            "name": "analyze_sentiment",
            "description": "Analyze sentiment toward the collective entity.",
            "parameters": {
                "type": "object",
                "properties": {
                    "named_entity": {"type": "string"},
                    "sentiment": {
                        "type": "string",
                        "enum": ["POSITIVE", "NEUTRAL", "NEGATIVE", "NOT RELEVANT"],
                    },
                    "confidence": {"type": "number", "minimum": 0, "maximum": 100},
                    "explanation": {"type": "string"},
                },
                "required": ["named_entity", "sentiment", "confidence", "explanation"],
            },
        }]
    else:
        session_state.sentiment_instruction = f"""
LABEL SET: VERY POSITIVE, SOMEWHAT POSITIVE, NEUTRAL, SOMEWHAT NEGATIVE, VERY NEGATIVE, NOT RELEVANT

WHAT TO JUDGE:
- The *collective entity*: {named_entity} + aliases + spokespeople acting on its behalf + products/sub-brands.

CRITERIA:
- VERY POSITIVE: Strong praise or substantial positive impact credited to the collective entity.
- SOMEWHAT POSITIVE: Moderate praise or minor positive outcomes.
- NEUTRAL: Factual/balanced coverage with no clear stance on the collective entity.
- SOMEWHAT NEGATIVE: Mild criticism or limited negative impact.
- VERY NEGATIVE: Strong criticism or substantial negative impact attributed to the collective entity.
- NOT RELEVANT: The collective entity (as defined) is not present.

OUTPUT:
- Provide the UPPERCASE label, a confidence (0–100), and a 1–2 sentence explanation focused on the collective entity.
""".strip()

        session_state.functions = [{
            "name": "analyze_sentiment",
            "description": "Analyze sentiment toward the collective entity with intensity levels.",
            "parameters": {
                "type": "object",
                "properties": {
                    "named_entity": {"type": "string"},
                    "sentiment": {
                        "type": "string",
                        "enum": [
                            "VERY POSITIVE", "SOMEWHAT POSITIVE", "NEUTRAL",
                            "SOMEWHAT NEGATIVE", "VERY NEGATIVE", "NOT RELEVANT",
                        ],
                    },
                    "confidence": {"type": "number", "minimum": 0, "maximum": 100},
                    "explanation": {"type": "string"},
                },
                "required": ["named_entity", "sentiment", "confidence", "explanation"],
            },
        }]

    session_state.toning_config_step = True
    session_state.last_saved = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def reset_sentiment_config_state(session_state) -> None:
    session_state.sentiment_config_step = False
    session_state.sentiment_sample_mode = "representative"
    session_state.sentiment_sample_size = None
    session_state.sentiment_full_override = False
    session_state.df_sentiment_rows = pd.DataFrame()
    session_state.df_sentiment_grouped_rows = pd.DataFrame()
    session_state.df_sentiment_unique = pd.DataFrame()
    session_state.sentiment_elapsed_time = 0.0

    session_state.ui_primary_names = [session_state.get("client_name", "")] if session_state.get("client_name") else []
    session_state.ui_alternate_names = []
    session_state.ui_spokespeople = []
    session_state.ui_products = []
    session_state.ui_toning_rationale = ""
    session_state.ui_sentiment_type = "3-way"

    for k in [
        "sentiment_type", "model_choice",
        "pre_prompt", "post_prompt", "sentiment_instruction", "functions",
        "highlight_keyword", "highlight_regex_str"
    ]:
        session_state.pop(k, None)

    session_state.toning_config_step = False
    session_state.last_saved = None