# ai_tagging.py
from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import pandas as pd
from openai import OpenAI

from utils.api_meter import add_api_usage, extract_usage_tokens


DEFAULT_TAGGING_MODEL = "gpt-5.4-nano"
DEFAULT_TAGGING_OBSERVATION_MODEL = "gpt-5.4-mini"
DEFAULT_TAGGING_REVIEW_MODEL = "gpt-5.4-mini"
DEFAULT_TAGGING_MAX_WORKERS = 8
DEFAULT_TAGGING_BATCH_SIZE = 50
DEFAULT_TAGGING_REVIEW_BATCH_SIZE = 50
DEFAULT_TAGGING_REVIEW_CONFIDENCE_THRESHOLD = 90
MAX_RETRIES = 2


def init_ai_tagging_state(session_state) -> None:
    session_state.setdefault("tag_definitions", {})
    session_state.setdefault("tagging_mode", "Single best tag")
    session_state.setdefault("tags_text", "")
    session_state.setdefault("tagging_observation_output", None)
    session_state.setdefault("tagging_review_idx", 0)
    session_state.setdefault("tagging_review_low_conf_threshold", DEFAULT_TAGGING_REVIEW_CONFIDENCE_THRESHOLD)
    session_state.setdefault("tagging_pre_review_n", DEFAULT_TAGGING_REVIEW_BATCH_SIZE)


def clean_text(text: str) -> str:
    return re.sub(r"[\u200B-\u200D\uFEFF\u202A-\u202E]", "", str(text or "")).strip()


def _extract_json_payload(text: str) -> dict[str, Any] | None:
    raw = str(text or "").strip()
    if not raw:
        return None

    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        pass

    m = re.search(r"\{.*\}", raw, flags=re.S)
    if not m:
        return None

    try:
        parsed = json.loads(m.group(0))
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


def _truncate_text(text: str, limit: int = 420) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def build_default_tags_text(client_name: str) -> str:
    client_name = str(client_name or "").strip() or "the brand"
    return f"""Sustainability: {client_name} is discussed in relation to environmental responsibility, green initiatives, or emissions reduction.
Innovation: {client_name} is discussed in relation to new technology, unique approaches, or product breakthroughs.
Other: {client_name} is not discussed in relation to any other tagging topics.
"""


def parse_tag_definitions(tags_text: str) -> dict[str, str]:
    tag_definitions = {}
    for line in str(tags_text or "").strip().splitlines():
        if ":" in line:
            tag, criteria = line.split(":", 1)
            cleaned_tag = clean_text(tag)
            cleaned_criteria = clean_text(criteria)
            if cleaned_tag and cleaned_criteria:
                tag_definitions[cleaned_tag] = cleaned_criteria
    return tag_definitions


def build_function_schemas(tagging_mode: str) -> list[dict[str, Any]]:
    if tagging_mode == "Single best tag":
        return [
            {
                "name": "apply_single_tag",
                "description": "Apply the best-fitting tag to a news story.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "tag": {"type": "string"},
                        "confidence": {"type": "integer"},
                        "explanation": {"type": "string"},
                    },
                    "required": ["tag", "confidence", "explanation"],
                },
            }
        ]

    return [
        {
            "name": "apply_multiple_tags",
            "description": "Apply all relevant tags to a news story.",
            "parameters": {
                "type": "object",
                "properties": {
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "confidence": {"type": "integer"},
                    "explanations": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["tags", "confidence", "explanations"],
            },
        }
    ]


def get_remaining_tagging_rows(df_tagging_unique: pd.DataFrame) -> pd.DataFrame:
    if df_tagging_unique is None or df_tagging_unique.empty:
        return pd.DataFrame()

    working = df_tagging_unique.copy()

    if "Tag_Processed" not in working.columns:
        working["Tag_Processed"] = False

    remaining = working[~working["Tag_Processed"]].copy().reset_index()
    return remaining


def get_effective_tag_series(df_tagging_unique: pd.DataFrame) -> pd.Series:
    if df_tagging_unique is None or df_tagging_unique.empty:
        return pd.Series(dtype="object")

    assigned = (
        df_tagging_unique.get("Assigned Tag", pd.Series(index=df_tagging_unique.index, dtype="object"))
        .fillna("")
        .astype(str)
        .str.strip()
    )
    ai = (
        df_tagging_unique.get("AI Tag", pd.Series(index=df_tagging_unique.index, dtype="object"))
        .fillna("")
        .astype(str)
        .str.strip()
    )
    return assigned.where(assigned != "", ai)


def normalize_tag_list(value: Any) -> list[str]:
    if isinstance(value, list):
        raw_items = value
    else:
        raw_items = str(value or "").split(",")

    cleaned: list[str] = []
    seen: set[str] = set()
    for item in raw_items:
        tag = str(item or "").strip()
        if not tag:
            continue
        norm = tag.casefold()
        if norm in seen:
            continue
        seen.add(norm)
        cleaned.append(tag)
    return cleaned


def build_tagging_prompt(row: pd.Series, tag_definitions: dict[str, str], tagging_mode: str) -> str:
    snippet_column = "Example Snippet" if "Example Snippet" in row.index else "Snippet"
    tag_rules = json.dumps(tag_definitions, indent=2)

    instruction = (
        "Only return ONE tag. Do not return multiple. Even if several might apply, choose the ONE most relevant tag based on the criteria below. Return it as a single string, not as a list. Also return an integer confidence from 0 to 100."
        if tagging_mode == "Single best tag"
        else "Apply all tags that are relevant to the article and also return an integer confidence from 0 to 100."
    )

    return f"""
You are a media analysis AI. Your task is to apply tags to this story based on the definitions provided.

Tag Definitions:
{tag_rules}

Instructions:
{instruction}

Content:
Headline: {row.get("Headline", "")}
Snippet: {row.get(snippet_column, "")}
""".strip()


def call_ai_tagging(
    client: OpenAI,
    row: pd.Series,
    tag_definitions: dict[str, str],
    tagging_mode: str,
    model: str,
) -> tuple[dict[str, Any], int, int]:
    functions = build_function_schemas(tagging_mode)
    prompt = build_tagging_prompt(row, tag_definitions, tagging_mode)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are an expert at tagging news stories based on defined criteria."},
            {"role": "user", "content": prompt},
        ],
        functions=functions,
        function_call={"name": functions[0]["name"]},
    )

    args = json.loads(response.choices[0].message.function_call.arguments)
    in_tok, out_tok = extract_usage_tokens(response)
    return args, in_tok, out_tok


def analyze_story_worker(
    row_tuple,
    tag_definitions: dict[str, str],
    tagging_mode: str,
    model: str,
    api_key: str,
) -> tuple[int, dict[str, Any], str, int, int]:
    idx, row_dict = row_tuple
    row = pd.Series(row_dict)

    client = OpenAI(api_key=api_key)
    last_error = ""

    for _ in range(MAX_RETRIES + 1):
        try:
            args, in_tok, out_tok = call_ai_tagging(
                client=client,
                row=row,
                tag_definitions=tag_definitions,
                tagging_mode=tagging_mode,
                model=model,
            )
            return idx, args, "", in_tok, out_tok
        except Exception as e:
            last_error = str(e)

    return idx, {}, last_error, 0, 0

def apply_tagging_result_to_unique_df(
    df_tagging_unique: pd.DataFrame,
    original_index: int,
    result: dict[str, Any],
    tagging_mode: str,
    tag_definitions: dict[str, str],
) -> pd.DataFrame:
    df = df_tagging_unique.copy()

    df.loc[original_index, "Tag_Processed"] = True

    if "AI Tag" not in df.columns:
        df["AI Tag"] = ""
    if "AI Tag Confidence" not in df.columns:
        df["AI Tag Confidence"] = pd.NA
    if "AI Tag Rationale" not in df.columns:
        df["AI Tag Rationale"] = ""

    if tagging_mode == "Single best tag":
        tag = str(result.get("tag", "") or "").strip()
        confidence = pd.to_numeric(pd.Series([result.get("confidence")]), errors="coerce").iloc[0]
        rationale = str(result.get("explanation", "") or "").strip()

        if "," in tag:
            rationale = "**NOTE: Multiple tags returned in single-tag mode.** " + rationale

        df.loc[original_index, "AI Tag"] = tag
        df.loc[original_index, "AI Tag Confidence"] = int(confidence) if pd.notna(confidence) else pd.NA
        df.loc[original_index, "AI Tag Rationale"] = rationale

        for tag_name in tag_definitions.keys():
            col = f"AI Tag: {tag_name}"
            if col not in df.columns:
                df[col] = 0
            df.loc[original_index, col] = 1 if tag == tag_name else 0

    else:
        tags = result.get("tags", [])
        confidence = pd.to_numeric(pd.Series([result.get("confidence")]), errors="coerce").iloc[0]
        explanations = result.get("explanations", [])

        if not isinstance(tags, list):
            tags = [str(tags)] if tags else []
        if not isinstance(explanations, list):
            explanations = [str(explanations)] if explanations else []

        tags = [str(tag).strip() for tag in tags if str(tag).strip()]
        explanations = [str(exp).strip() for exp in explanations if str(exp).strip()]

        df.loc[original_index, "AI Tag"] = ", ".join(tags)
        df.loc[original_index, "AI Tag Confidence"] = int(confidence) if pd.notna(confidence) else pd.NA
        df.loc[original_index, "AI Tag Rationale"] = " | ".join(explanations)

        for tag_name in tag_definitions.keys():
            col = f"AI Tag: {tag_name}"
            if col not in df.columns:
                df[col] = 0
            df.loc[original_index, col] = 1 if tag_name in tags else 0

    return df


def cascade_tags_to_rows(
    df_tagging_rows: pd.DataFrame,
    df_tagging_unique: pd.DataFrame,
) -> pd.DataFrame:
    if df_tagging_rows is None or df_tagging_rows.empty:
        return df_tagging_rows

    grouped = df_tagging_rows.copy()
    unique = df_tagging_unique.copy()

    tag_cols_to_copy = [
        col
        for col in unique.columns
        if col.startswith("AI Tag")
        or col.startswith("Review AI")
        or col in {"Assigned Tag", "Needs Human Review"}
    ]

    if not tag_cols_to_copy or "Group ID" not in grouped.columns or "Group ID" not in unique.columns:
        return grouped

    merge_cols = ["Group ID"] + tag_cols_to_copy
    mapping = unique[merge_cols].drop_duplicates(subset=["Group ID"]).copy()

    grouped = grouped.drop(columns=tag_cols_to_copy, errors="ignore")
    grouped = grouped.merge(mapping, on="Group ID", how="left")
    return grouped


def build_tag_observation_payload(
    df_tagging_unique: pd.DataFrame,
    include_other: bool,
    per_tag_limit: int = 6,
) -> dict[str, Any]:
    if df_tagging_unique is None or df_tagging_unique.empty:
        return {"distribution": [], "examples_by_tag": {}}

    working = df_tagging_unique.copy()
    for col in ["Headline", "Snippet", "URL", "Outlet", "Type", "AI Tag", "Assigned Tag", "AI Tag Rationale"]:
        if col not in working.columns:
            working[col] = ""
        working[col] = working[col].fillna("").astype(str).str.strip()

    for col in ["Mentions", "Impressions", "Effective Reach", "Group Count"]:
        if col not in working.columns:
            working[col] = 0
        working[col] = pd.to_numeric(working[col], errors="coerce").fillna(0)

    effective_tags = get_effective_tag_series(working)
    tags_expanded = (
        working[["Group ID"]]
        .assign(Tag=effective_tags.fillna("").astype(str).str.split(","))
        .explode("Tag")
    )
    tags_expanded["Tag"] = tags_expanded["Tag"].fillna("").astype(str).str.strip()
    tags_expanded = tags_expanded[tags_expanded["Tag"] != ""].copy()

    if not include_other:
        tags_expanded = tags_expanded[tags_expanded["Tag"].str.lower() != "other"].copy()

    if tags_expanded.empty:
        return {"distribution": [], "examples_by_tag": {}}

    distribution = (
        tags_expanded["Tag"]
        .value_counts()
        .rename_axis("Tag")
        .reset_index(name="Count")
    )
    distribution["Share"] = distribution["Count"] / float(distribution["Count"].sum())

    working["_has_url"] = working["URL"].ne("")
    working["_is_online"] = working["Type"].str.upper().eq("ONLINE")
    if "Date" not in working.columns:
        working["Date"] = ""
    tagged_rows = tags_expanded.merge(working, on="Group ID", how="left")

    examples_by_tag: dict[str, list[dict[str, Any]]] = {}
    for tag_name, tag_group in tagged_rows.groupby("Tag", dropna=False):
        ranked = tag_group.sort_values(
            ["_is_online", "_has_url", "Group Count", "Mentions", "Impressions", "Effective Reach"],
            ascending=[False, False, False, False, False, False],
        )
        examples = []
        for _, row in ranked.drop_duplicates(subset=["Headline"], keep="first").head(per_tag_limit).iterrows():
            headline = str(row.get("Headline", "") or "").strip()
            if not headline:
                continue
            examples.append(
                {
                    "group_id": row.get("Group ID", ""),
                    "headline": headline,
                    "outlet": str(row.get("Outlet", "") or "").strip(),
                    "date": str(row.get("Date", "") or "").strip(),
                    "url": str(row.get("URL", "") or "").strip(),
                    "example_type": str(row.get("Type", "") or "").strip(),
                    "mentions": int(row.get("Mentions", 0) or 0),
                    "impressions": int(row.get("Impressions", 0) or 0),
                    "effective_reach": int(row.get("Effective Reach", 0) or 0),
                    "group_count": int(row.get("Group Count", 0) or 0),
                    "snippet": _truncate_text(row.get("Snippet", ""), 420),
                    "tag_rationale": _truncate_text(row.get("AI Tag Rationale", ""), 220),
                }
            )
        examples_by_tag[str(tag_name)] = examples

    return {
        "distribution": distribution.to_dict(orient="records"),
        "examples_by_tag": examples_by_tag,
    }


def build_tag_observation_prompt(
    client_name: str,
    include_other: bool,
    payload: dict[str, Any],
) -> str:
    return f"""
You are helping a media intelligence analyst write concise, report-ready tagging observations for {client_name or 'the client'}.

Use the finalized tag distribution and representative grouped stories below.
The examples are intentionally selected from the highest-volume and most syndicated coverage in each tag bucket.
Each example may also include an existing AI tag rationale; use it as supporting context, but ground your summary in the coverage details themselves.

Return strict JSON with this shape:
{{
  "overall_observation": "1-2 sentences",
  "tag_sections": [
    {{
      "tag": "LABEL",
      "observation": "1-2 concise sentences"
    }}
  ]
}}

Requirements:
- Keep it factual, neutral, and report-ready.
- Explain what kinds of coverage are driving each tag category.
- Do not overstate tiny buckets.
- {"Exclude Other from the narrative unless it appears in the provided data." if not include_other else "Include Other only if it appears meaningfully in the provided examples."}
- Use only tag labels present in the provided data.

Input data:
{json.dumps(payload, ensure_ascii=True)}
""".strip()


def generate_tag_observations(
    df_tagging_unique: pd.DataFrame,
    client_name: str,
    include_other: bool,
    api_key: str,
    model: str = DEFAULT_TAGGING_OBSERVATION_MODEL,
) -> tuple[dict[str, Any], int, int]:
    payload = build_tag_observation_payload(
        df_tagging_unique=df_tagging_unique,
        include_other=include_other,
    )
    prompt = build_tag_observation_prompt(
        client_name=client_name,
        include_other=include_other,
        payload=payload,
    )

    client = OpenAI(api_key=api_key)
    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": "You write concise, neutral media-intelligence summaries."},
            {"role": "user", "content": prompt},
        ],
        text={"verbosity": "low"},
    )

    add_api_usage(response, model)
    in_tok, out_tok = extract_usage_tokens(response)
    raw = getattr(response, "output_text", "") or ""
    parsed = _extract_json_payload(raw)
    if parsed is None:
        raise ValueError("Model did not return valid JSON for tag observations.")
    parsed["_examples_by_tag"] = payload.get("examples_by_tag", {})
    return parsed, in_tok, out_tok


def ensure_tag_review_columns(
    df_unique: pd.DataFrame,
    df_grouped: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    unique = df_unique.copy()
    grouped = df_grouped.copy()

    review_cols = {
        "Review AI Tag": pd.NA,
        "Review AI Confidence": pd.NA,
        "Review AI Rationale": pd.NA,
        "AI Tag Agreement": pd.NA,
        "Needs Human Review": pd.NA,
        "Assigned Tag": pd.NA,
    }

    for df in [unique, grouped]:
        for col, default_val in review_cols.items():
            if col not in df.columns:
                df[col] = default_val
        if "AI Tag Confidence" not in df.columns:
            df["AI Tag Confidence"] = pd.NA

    return unique, grouped


def compute_tag_review_candidates(df_unique: pd.DataFrame) -> pd.DataFrame:
    if df_unique is None or df_unique.empty:
        return pd.DataFrame()

    working = df_unique.copy()
    assigned = working.get("Assigned Tag", pd.Series(index=working.index, dtype="object")).fillna("").astype(str).str.strip()
    ai_tag = working.get("AI Tag", pd.Series(index=working.index, dtype="object")).fillna("").astype(str).str.strip()
    ai_conf = pd.to_numeric(working.get("AI Tag Confidence", pd.Series(index=working.index, dtype="float")), errors="coerce")

    pool = working[(assigned == "") & (ai_tag != "")].copy()
    if pool.empty:
        return pool

    pool["AI Tag Confidence"] = pd.to_numeric(pool.get("AI Tag Confidence"), errors="coerce")
    for col in ["Mentions", "Impressions", "Effective Reach"]:
        if col not in pool.columns:
            pool[col] = 0
        pool[col] = pd.to_numeric(pool[col], errors="coerce").fillna(0)

    pool["_ai_conf_sort"] = ai_conf.loc[pool.index].fillna(-1)
    pool = (
        pool.sort_values(
            ["_ai_conf_sort", "Mentions", "Impressions", "Effective Reach"],
            ascending=[True, False, False, False],
        )
        .drop(columns=["_ai_conf_sort"], errors="ignore")
        .reset_index()
    )
    return pool


def filter_tag_candidates_for_review_mode(
    candidates_df: pd.DataFrame,
    review_mode: str,
) -> pd.DataFrame:
    if candidates_df.empty:
        return candidates_df

    out = candidates_df.copy()

    if review_mode == "Flagged for human review":
        if "Needs Human Review" in out.columns:
            flagged = out[out["Needs Human Review"].fillna("").astype(str).str.strip() == "Yes"].copy()
            if not flagged.empty:
                return flagged
        return out

    if review_mode == "Disagreements only":
        if "AI Tag Agreement" in out.columns:
            disagreements = out[out["AI Tag Agreement"].fillna("").astype(str).str.strip() == "Disagree"].copy()
            if not disagreements.empty:
                return disagreements
        return out

    if review_mode == "All Tagged Coverage":
        return out

    return out


def compute_all_tagged_candidates(df_unique: pd.DataFrame) -> pd.DataFrame:
    if df_unique is None or df_unique.empty:
        return pd.DataFrame()

    working = df_unique.copy()
    ai_tag = working.get("AI Tag", pd.Series(index=working.index, dtype="object")).fillna("").astype(str).str.strip()
    pool = working[ai_tag != ""].copy()
    if pool.empty:
        return pool

    for col in ["Mentions", "Impressions", "Effective Reach"]:
        if col not in pool.columns:
            pool[col] = 0
        pool[col] = pd.to_numeric(pool[col], errors="coerce").fillna(0)

    pool["AI Tag Confidence"] = pd.to_numeric(
        pool.get("AI Tag Confidence", pd.Series(index=pool.index, dtype="float")),
        errors="coerce",
    ).fillna(-1)
    pool = (
        pool.sort_values(
            ["Mentions", "Impressions", "Effective Reach", "AI Tag Confidence"],
            ascending=[False, False, False, False],
        )
        .reset_index()
    )
    return pool


def write_review_tag_to_group(
    df_unique: pd.DataFrame,
    df_grouped: pd.DataFrame,
    group_id: int,
    review_label: str | None,
    review_confidence: Any,
    review_rationale: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    unique = df_unique.copy()
    grouped = df_grouped.copy()

    for df in [unique, grouped]:
        mask = df["Group ID"] == group_id
        df.loc[mask, "Review AI Tag"] = review_label
        df.loc[mask, "Review AI Confidence"] = review_confidence
        df.loc[mask, "Review AI Rationale"] = review_rationale

    return unique, grouped


def apply_tag_review_flags_to_group(
    df_unique: pd.DataFrame,
    df_grouped: pd.DataFrame,
    group_id: int,
    ai_label: str | None,
    review_label: str | None,
    review_confidence: Any,
    tagging_mode: str = "Single best tag",
    low_conf_threshold: int = DEFAULT_TAGGING_REVIEW_CONFIDENCE_THRESHOLD,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    unique = df_unique.copy()
    grouped = df_grouped.copy()

    if tagging_mode == "Multiple applicable tags":
        ai_tags = {tag.casefold() for tag in normalize_tag_list(ai_label)}
        review_tags = {tag.casefold() for tag in normalize_tag_list(review_label)}
        agreement = bool(ai_tags and review_tags and ai_tags == review_tags)
    else:
        ai_label_clean = str(ai_label or "").strip().casefold()
        review_label_clean = str(review_label or "").strip().casefold()
        agreement = bool(ai_label_clean and review_label_clean and ai_label_clean == review_label_clean)

    review_conf = pd.to_numeric(pd.Series([review_confidence]), errors="coerce").iloc[0]
    needs_human = not (agreement and pd.notna(review_conf) and review_conf >= low_conf_threshold)

    for df in [unique, grouped]:
        mask = df["Group ID"] == group_id
        df.loc[mask, "AI Tag Agreement"] = "Match" if agreement else "Disagree"
        df.loc[mask, "Needs Human Review"] = "Yes" if needs_human else "No"

    return unique, grouped


def set_assigned_tag(
    df_unique: pd.DataFrame,
    df_grouped: pd.DataFrame,
    group_id: int,
    label: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    unique = df_unique.copy()
    grouped = df_grouped.copy()

    for df in [unique, grouped]:
        mask = df["Group ID"] == group_id
        df.loc[mask, "Assigned Tag"] = label

    return unique, grouped


def auto_assign_resolved_tag_matches(
    df_unique: pd.DataFrame,
    df_grouped: pd.DataFrame,
    tagging_mode: str = "Single best tag",
    confidence_threshold: int = DEFAULT_TAGGING_REVIEW_CONFIDENCE_THRESHOLD,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    unique = df_unique.copy()
    grouped = df_grouped.copy()

    required_cols = [
        "Group ID",
        "AI Tag",
        "Review AI Tag",
        "Review AI Confidence",
        "AI Tag Agreement",
        "Assigned Tag",
    ]
    for col in required_cols:
        if col not in unique.columns:
            return unique, grouped, 0

    ai_tag = unique["AI Tag"].fillna("").astype(str).str.strip()
    review_tag = unique["Review AI Tag"].fillna("").astype(str).str.strip()
    assigned = unique["Assigned Tag"].fillna("").astype(str).str.strip()
    review_conf = pd.to_numeric(unique["Review AI Confidence"], errors="coerce")

    if tagging_mode == "Multiple applicable tags":
        exact_match_mask = pd.Series(False, index=unique.index)
        for row_idx in unique.index:
            ai_set = {tag.casefold() for tag in normalize_tag_list(ai_tag.loc[row_idx])}
            review_set = {tag.casefold() for tag in normalize_tag_list(review_tag.loc[row_idx])}
            exact_match_mask.loc[row_idx] = bool(ai_set and review_set and ai_set == review_set)
    else:
        exact_match_mask = ai_tag.str.casefold() == review_tag.str.casefold()

    mask = (
        (assigned == "")
        & (unique["AI Tag Agreement"].fillna("") == "Match")
        & (ai_tag != "")
        & exact_match_mask
        & review_conf.ge(confidence_threshold)
    )

    accepted_count = int(mask.sum())
    if accepted_count == 0:
        return unique, grouped, 0

    for gid in unique.loc[mask, "Group ID"].tolist():
        label = unique.loc[unique["Group ID"] == gid, "AI Tag"].iloc[0]
        unique, grouped = set_assigned_tag(unique, grouped, gid, ", ".join(normalize_tag_list(label)))

    return unique, grouped, accepted_count


def second_opinion_tag_worker(
    row_tuple,
    tag_definitions: dict[str, str],
    tagging_mode: str,
    review_model: str,
    api_key: str,
) -> tuple[int, dict[str, Any], str, int, int]:
    idx, row_dict = row_tuple
    row = pd.Series(row_dict)
    client = OpenAI(api_key=api_key)
    last_error = ""

    for _ in range(MAX_RETRIES + 1):
        try:
            args, in_tok, out_tok = call_ai_tagging(
                client=client,
                row=row,
                tag_definitions=tag_definitions,
                tagging_mode=tagging_mode,
                model=review_model,
            )
            return idx, args, "", in_tok, out_tok
        except Exception as e:
            last_error = str(e)

    return idx, {}, last_error, 0, 0


def run_batch_tag_second_opinion(
    candidates_df: pd.DataFrame,
    df_unique: pd.DataFrame,
    df_grouped: pd.DataFrame,
    tag_definitions: dict[str, str],
    tagging_mode: str,
    api_key: str,
    review_model: str = DEFAULT_TAGGING_REVIEW_MODEL,
    limit: int = DEFAULT_TAGGING_REVIEW_BATCH_SIZE,
    max_workers: int = DEFAULT_TAGGING_MAX_WORKERS,
    low_conf_threshold: int = DEFAULT_TAGGING_REVIEW_CONFIDENCE_THRESHOLD,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], int, int]:
    unique, grouped = ensure_tag_review_columns(df_unique, df_grouped)

    working_candidates = candidates_df.head(limit).copy()
    rows_for_workers = [(idx, row.to_dict()) for idx, row in working_candidates.iterrows()]

    total_in = 0
    total_out = 0
    errors: list[str] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(
                second_opinion_tag_worker,
                row_tuple,
                tag_definitions,
                tagging_mode,
                review_model,
                api_key,
            ): row_tuple[0]
            for row_tuple in rows_for_workers
        }

        for future in as_completed(future_map):
            idx = future_map[future]
            try:
                _, result, error_message, in_tok, out_tok = future.result()
                total_in += int(in_tok or 0)
                total_out += int(out_tok or 0)

                row = working_candidates.loc[idx]
                gid = int(row["Group ID"])

                if error_message:
                    errors.append(f"Story {gid}: {error_message}")
                    continue

                if tagging_mode == "Multiple applicable tags":
                    review_tags = normalize_tag_list(result.get("tags", []))
                    review_label = ", ".join(review_tags)
                    explanations = result.get("explanations", [])
                    if not isinstance(explanations, list):
                        explanations = [str(explanations)] if explanations else []
                    review_rsn = " | ".join(str(exp).strip() for exp in explanations if str(exp).strip())
                else:
                    review_label = str(result.get("tag", "") or "").strip()
                    review_rsn = str(result.get("explanation", "") or "").strip()
                review_conf = pd.to_numeric(pd.Series([result.get("confidence")]), errors="coerce").iloc[0]

                unique, grouped = write_review_tag_to_group(
                    unique,
                    grouped,
                    gid,
                    review_label,
                    int(review_conf) if pd.notna(review_conf) else pd.NA,
                    review_rsn,
                )
                unique, grouped = apply_tag_review_flags_to_group(
                    unique,
                    grouped,
                    gid,
                    ai_label=row.get("AI Tag"),
                    review_label=review_label,
                    review_confidence=review_conf,
                    tagging_mode=tagging_mode,
                    low_conf_threshold=low_conf_threshold,
                )
            except Exception as e:
                errors.append(str(e))

    unique, grouped, _ = auto_assign_resolved_tag_matches(
        unique,
        grouped,
        tagging_mode=tagging_mode,
        confidence_threshold=low_conf_threshold,
    )

    return unique, grouped, errors, total_in, total_out


def reset_ai_tagging_results(
    df_tagging_unique: pd.DataFrame,
    df_tagging_rows: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    unique = df_tagging_unique.copy()
    grouped = df_tagging_rows.copy()

    if "Tag_Processed" in unique.columns:
        unique["Tag_Processed"] = False

    tag_columns_unique = [
        col
        for col in unique.columns
        if col.startswith("AI Tag") or col.startswith("Review AI") or col in {"Needs Human Review", "Assigned Tag", "AI Tag Agreement"}
    ]
    unique.drop(columns=tag_columns_unique, inplace=True, errors="ignore")

    tag_columns_grouped = [
        col
        for col in grouped.columns
        if col.startswith("AI Tag") or col.startswith("Review AI") or col in {"Needs Human Review", "Assigned Tag", "AI Tag Agreement"}
    ]
    grouped.drop(columns=tag_columns_grouped, inplace=True, errors="ignore")

    return unique, grouped
