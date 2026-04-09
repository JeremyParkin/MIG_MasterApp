# ai_tagging.py
from __future__ import annotations

import json
import re
from typing import Any

import pandas as pd
from openai import OpenAI

from utils.api_meter import extract_usage_tokens


DEFAULT_TAGGING_MODEL = "gpt-5.4-nano"
DEFAULT_TAGGING_MAX_WORKERS = 8
DEFAULT_TAGGING_BATCH_SIZE = 25
MAX_RETRIES = 2


def init_ai_tagging_state(session_state) -> None:
    session_state.setdefault("tag_definitions", {})
    session_state.setdefault("tagging_mode", "Single best tag")
    session_state.setdefault("tags_text", "")


def clean_text(text: str) -> str:
    return re.sub(r"[\u200B-\u200D\uFEFF\u202A-\u202E]", "", str(text or "")).strip()


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
                        "explanation": {"type": "string"},
                    },
                    "required": ["tag", "explanation"],
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
                    "explanations": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["tags", "explanations"],
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


def build_tagging_prompt(row: pd.Series, tag_definitions: dict[str, str], tagging_mode: str) -> str:
    snippet_column = "Example Snippet" if "Example Snippet" in row.index else "Snippet"
    tag_rules = json.dumps(tag_definitions, indent=2)

    instruction = (
        "Only return ONE tag. Do not return multiple. Even if several might apply, choose the ONE most relevant tag based on the criteria below. Return it as a single string, not as a list."
        if tagging_mode == "Single best tag"
        else "Apply all tags that are relevant to the article."
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
    if "AI Tag Rationale" not in df.columns:
        df["AI Tag Rationale"] = ""

    if tagging_mode == "Single best tag":
        tag = str(result.get("tag", "") or "").strip()
        rationale = str(result.get("explanation", "") or "").strip()

        if "," in tag:
            rationale = "**NOTE: Multiple tags returned in single-tag mode.** " + rationale

        df.loc[original_index, "AI Tag"] = tag
        df.loc[original_index, "AI Tag Rationale"] = rationale

        for tag_name in tag_definitions.keys():
            col = f"AI Tag: {tag_name}"
            if col not in df.columns:
                df[col] = 0
            df.loc[original_index, col] = 1 if tag == tag_name else 0

    else:
        tags = result.get("tags", [])
        explanations = result.get("explanations", [])

        if not isinstance(tags, list):
            tags = [str(tags)] if tags else []
        if not isinstance(explanations, list):
            explanations = [str(explanations)] if explanations else []

        tags = [str(tag).strip() for tag in tags if str(tag).strip()]
        explanations = [str(exp).strip() for exp in explanations if str(exp).strip()]

        df.loc[original_index, "AI Tag"] = ", ".join(tags)
        df.loc[original_index, "AI Tag Rationale"] = " | ".join(explanations)

        for tag_name in tag_definitions.keys():
            col = f"AI Tag: {tag_name}"
            if col not in df.columns:
                df[col] = 0
            df.loc[original_index, col] = 1 if tag_name in tags else 0

    return df


def cascade_tags_to_grouped_rows(
    df_tagging_grouped_rows: pd.DataFrame,
    df_tagging_unique: pd.DataFrame,
) -> pd.DataFrame:
    if df_tagging_grouped_rows is None or df_tagging_grouped_rows.empty:
        return df_tagging_grouped_rows

    grouped = df_tagging_grouped_rows.copy()
    unique = df_tagging_unique.copy()

    tag_cols_to_copy = [col for col in unique.columns if col.startswith("AI Tag")]

    if not tag_cols_to_copy or "Group ID" not in grouped.columns or "Group ID" not in unique.columns:
        return grouped

    merge_cols = ["Group ID"] + tag_cols_to_copy
    mapping = unique[merge_cols].drop_duplicates(subset=["Group ID"]).copy()

    grouped = grouped.drop(columns=tag_cols_to_copy, errors="ignore")
    grouped = grouped.merge(mapping, on="Group ID", how="left")
    return grouped


def reset_ai_tagging_results(
    df_tagging_unique: pd.DataFrame,
    df_tagging_grouped_rows: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    unique = df_tagging_unique.copy()
    grouped = df_tagging_grouped_rows.copy()

    if "Tag_Processed" in unique.columns:
        unique["Tag_Processed"] = False

    tag_columns_unique = [col for col in unique.columns if col.startswith("AI Tag")]
    unique.drop(columns=tag_columns_unique, inplace=True, errors="ignore")

    tag_columns_grouped = [col for col in grouped.columns if col.startswith("AI Tag")]
    grouped.drop(columns=tag_columns_grouped, inplace=True, errors="ignore")

    return unique, grouped