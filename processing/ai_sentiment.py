# ai_sentiment.py
from __future__ import annotations

import json
import re
from typing import Any

import pandas as pd
from openai import OpenAI

from utils.api_meter import extract_usage_tokens


DEFAULT_SENTIMENT_MODEL = "gpt-5.4-nano"
DEFAULT_SENTIMENT_BATCH_SIZE = 25
DEFAULT_SENTIMENT_MAX_WORKERS = 8
MAX_RETRIES = 2


def _allowed_sentiment_labels(sentiment_type: str) -> list[str]:
    if sentiment_type == "3-way":
        return ["POSITIVE", "NEUTRAL", "NEGATIVE", "NOT RELEVANT"]
    return ["VERY POSITIVE", "SOMEWHAT POSITIVE", "NEUTRAL", "SOMEWHAT NEGATIVE", "VERY NEGATIVE", "NOT RELEVANT"]


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


def _validate_structured_result(result: dict[str, Any], sentiment_type: str) -> tuple[dict[str, Any], str | None]:
    labels = _allowed_sentiment_labels(sentiment_type)

    sentiment = str(result.get("sentiment", "")).strip().upper()
    if sentiment not in labels:
        return {}, f"Invalid sentiment label: {sentiment or 'missing'}"

    conf_val = pd.to_numeric(pd.Series([result.get("confidence")]), errors="coerce").iloc[0]
    if pd.isna(conf_val):
        return {}, "Missing confidence value"
    confidence = int(max(0, min(100, float(conf_val))))

    explanation = str(result.get("explanation", "")).strip()
    if not explanation:
        return {}, "Missing explanation value"

    return {
        "sentiment": sentiment,
        "confidence": confidence,
        "explanation": explanation,
    }, None


def init_ai_sentiment_state(session_state) -> None:
    session_state.setdefault("sentiment_type", "3-way")
    session_state.setdefault("ui_sentiment_type", "3-way")
    session_state.setdefault("model_choice", DEFAULT_SENTIMENT_MODEL)
    session_state.setdefault("pre_prompt", "")
    session_state.setdefault("post_prompt", "")
    session_state.setdefault("sentiment_instruction", "")
    session_state.setdefault("functions", [])


def ensure_ai_sentiment_columns(df_unique: pd.DataFrame, df_grouped: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    unique = df_unique.copy()
    grouped = df_grouped.copy()

    for df in [unique, grouped]:
        for col in ["Assigned Sentiment", "AI Sentiment", "AI Sentiment Confidence", "AI Sentiment Rationale"]:
            if col not in df.columns:
                df[col] = pd.NA

    return unique, grouped


def build_story_prompt(
    headline: str,
    snippet: str,
    pre_prompt: str,
    sentiment_instruction: str,
    post_prompt: str,
) -> str:
    parts = []
    if pre_prompt:
        parts.append(pre_prompt)
    if sentiment_instruction:
        parts.append(sentiment_instruction)
    if post_prompt:
        parts.append(post_prompt)
    parts.append("This is the news story:")
    parts.append(f"HEADLINE: {headline or ''}")
    parts.append(f"BODY: {snippet or ''}")
    return "\n\n".join(parts)


def parse_plain_text_response(txt: str, sentiment_type: str) -> tuple[str | None, int | None, str]:
    cand3 = ["POSITIVE", "NEUTRAL", "NEGATIVE", "NOT RELEVANT"]
    cand5 = ["VERY POSITIVE", "SOMEWHAT POSITIVE", "NEUTRAL", "SOMEWHAT NEGATIVE", "VERY NEGATIVE", "NOT RELEVANT"]
    candidates = cand3 if sentiment_type == "3-way" else cand5

    sentiment = next((c for c in candidates if re.search(rf"\b{re.escape(c)}\b", txt)), None)
    m = re.search(r"confidence[^0-9]{0,10}(\d{1,3})", txt, flags=re.I)
    confidence = max(0, min(100, int(m.group(1)))) if m else None

    return sentiment, confidence, txt


def call_ai_sentiment(
    client: OpenAI,
    story_prompt: str,
    model_id: str,
    functions: list[dict[str, Any]],
    sentiment_type: str,
) -> tuple[dict[str, Any], int, int]:
    if functions:
        try:
            resp = client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": "You are a highly knowledgeable media analysis AI."},
                    {"role": "user", "content": story_prompt},
                ],
                functions=functions,
                function_call={"name": "analyze_sentiment"},
            )
            choice = resp.choices[0]
            in_tok, out_tok = extract_usage_tokens(resp)

            if getattr(choice.message, "function_call", None):
                fc = choice.message.function_call
                if fc and fc.name == "analyze_sentiment":
                    args = json.loads(fc.arguments or "{}")
                    parsed, err = _validate_structured_result(args, sentiment_type)
                    if not err:
                        return parsed, in_tok, out_tok
        except Exception:
            pass

    resp = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": "You are a highly knowledgeable media analysis AI."},
            {
                "role": "user",
                "content": (
                    f"{story_prompt}\n\n"
                    "Return only JSON with keys: sentiment, confidence, explanation. "
                    "Use an allowed sentiment label exactly, confidence must be 0-100."
                ),
            },
        ],
    )
    txt = (resp.choices[0].message.content or "").strip()
    in_tok, out_tok = extract_usage_tokens(resp)

    payload = _extract_json_payload(txt)
    if payload is None:
        raise ValueError("Structured output missing JSON payload.")

    parsed, err = _validate_structured_result(payload, sentiment_type)
    if err:
        raise ValueError(f"Structured output validation failed: {err}")

    return parsed, in_tok, out_tok


def get_remaining_sentiment_rows(
    df_sentiment_unique: pd.DataFrame,
    df_sentiment_grouped_rows: pd.DataFrame,
) -> pd.DataFrame:
    human_labeled_groups = set(
        df_sentiment_grouped_rows.loc[
            df_sentiment_grouped_rows["Assigned Sentiment"].notna(), "Group ID"
        ].unique()
    )
    already_ai_groups = set(
        df_sentiment_unique.loc[
            df_sentiment_unique["AI Sentiment"].notna(), "Group ID"
        ].unique()
    )

    remaining_mask = (
        ~df_sentiment_unique["Group ID"].isin(human_labeled_groups)
    ) & (
        ~df_sentiment_unique["Group ID"].isin(already_ai_groups)
    )

    return df_sentiment_unique.loc[remaining_mask].reset_index(drop=False)


def analyze_sentiment_worker(
    row_tuple,
    pre_prompt: str,
    sentiment_instruction: str,
    post_prompt: str,
    functions: list[dict[str, Any]],
    model_id: str,
    sentiment_type: str,
    api_key: str,
) -> tuple[int, dict[str, Any], str, int, int]:
    idx, row_dict = row_tuple
    row = pd.Series(row_dict)

    client = OpenAI(api_key=api_key)
    last_error = ""

    headline = row.get("Headline", "")
    snippet = row.get("Example Snippet", row.get("Snippet", ""))

    story_prompt = build_story_prompt(
        headline=headline,
        snippet=snippet,
        pre_prompt=pre_prompt,
        sentiment_instruction=sentiment_instruction,
        post_prompt=post_prompt,
    )

    for _ in range(MAX_RETRIES + 1):
        try:
            result, in_tok, out_tok = call_ai_sentiment(
                client=client,
                story_prompt=story_prompt,
                model_id=model_id,
                functions=functions,
                sentiment_type=sentiment_type,
            )
            return idx, result, "", in_tok, out_tok
        except Exception as e:
            last_error = str(e)

    return idx, {}, last_error, 0, 0


def apply_sentiment_result_to_unique_df(
    df_sentiment_unique: pd.DataFrame,
    original_index: int,
    result: dict[str, Any],
) -> pd.DataFrame:
    df = df_sentiment_unique.copy()

    df.loc[original_index, "AI Sentiment"] = result.get("sentiment")
    df.loc[original_index, "AI Sentiment Confidence"] = result.get("confidence")
    df.loc[original_index, "AI Sentiment Rationale"] = result.get("explanation")

    return df


def cascade_sentiment_to_grouped_rows(
    df_sentiment_grouped_rows: pd.DataFrame,
    df_sentiment_unique: pd.DataFrame,
) -> pd.DataFrame:
    grouped = df_sentiment_grouped_rows.copy()
    unique = df_sentiment_unique.copy()

    cols_to_copy = ["Group ID", "AI Sentiment", "AI Sentiment Confidence", "AI Sentiment Rationale"]
    mapping = unique[cols_to_copy].drop_duplicates(subset=["Group ID"]).copy()

    grouped = grouped.drop(columns=["AI Sentiment", "AI Sentiment Confidence", "AI Sentiment Rationale"], errors="ignore")
    grouped = grouped.merge(mapping, on="Group ID", how="left")

    return grouped


def reset_ai_sentiment_results(
    df_sentiment_unique: pd.DataFrame,
    df_sentiment_grouped_rows: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    unique = df_sentiment_unique.copy()
    grouped = df_sentiment_grouped_rows.copy()

    for df in [unique, grouped]:
        for col in ["AI Sentiment", "AI Sentiment Confidence", "AI Sentiment Rationale"]:
            if col in df.columns:
                df[col] = pd.NA

    return unique, grouped

def build_sentiment_distribution(df_unique: pd.DataFrame, sentiment_type: str) -> pd.DataFrame:
    if sentiment_type == "5-way":
        order = [
            "VERY POSITIVE",
            "SOMEWHAT POSITIVE",
            "NEUTRAL",
            "SOMEWHAT NEGATIVE",
            "VERY NEGATIVE",
            "NOT RELEVANT",
        ]
    else:
        order = [
            "POSITIVE",
            "NEUTRAL",
            "NEGATIVE",
            "NOT RELEVANT",
        ]

    assigned = df_unique.get("Assigned Sentiment", pd.Series(index=df_unique.index, dtype="object")).fillna("").astype(str).str.strip()
    ai = df_unique.get("AI Sentiment", pd.Series(index=df_unique.index, dtype="object")).fillna("").astype(str).str.strip()

    final = assigned.where(assigned != "", ai)
    final = final.where(final != "", "UNASSIGNED").str.upper()

    sentiment_counts = final.value_counts().rename_axis("Sentiment").reset_index(name="Count")
    base = pd.DataFrame({"Sentiment": order})
    out = base.merge(sentiment_counts, on="Sentiment", how="left")
    out["Count"] = out["Count"].fillna(0).astype(int)
    total = int(out["Count"].sum())
    out["Share"] = out["Count"] / total if total > 0 else 0
    return out