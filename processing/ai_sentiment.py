# ai_sentiment.py
from __future__ import annotations

import json
import re
from typing import Any

import pandas as pd
from openai import OpenAI

from utils.api_meter import add_api_usage, extract_usage_tokens


DEFAULT_SENTIMENT_MODEL = "gpt-5.4-nano"
DEFAULT_SENTIMENT_OBSERVATION_MODEL = "gpt-5.4-mini"
DEFAULT_SENTIMENT_BATCH_SIZE = 50
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

    assigned = _get_text_series(df_unique, "Assigned Sentiment")
    ai = _get_text_series(df_unique, "AI Sentiment")

    final = assigned.where(assigned != "", ai)
    final = final.where(final != "", "UNASSIGNED").str.upper()

    sentiment_counts = final.value_counts().rename_axis("Sentiment").reset_index(name="Count")
    base = pd.DataFrame({"Sentiment": order})
    out = base.merge(sentiment_counts, on="Sentiment", how="left")
    out["Count"] = out["Count"].fillna(0).astype(int)
    total = int(out["Count"].sum())
    out["Share"] = out["Count"] / total if total > 0 else 0
    return out


def _get_text_series(df: pd.DataFrame, column_name: str) -> pd.Series:
    if column_name not in df.columns:
        return pd.Series(index=df.index, dtype="object")

    values = df[column_name]
    if isinstance(values, pd.DataFrame):
        values = values.iloc[:, 0]

    return values.fillna("").astype(str).str.strip()


def build_final_sentiment_series(df_unique: pd.DataFrame) -> pd.Series:
    assigned = _get_text_series(df_unique, "Assigned Sentiment")
    ai = _get_text_series(df_unique, "AI Sentiment")
    final = assigned.where(assigned != "", ai)
    return final.where(final != "", pd.NA).astype("string").str.upper()


def _truncate_text(text: str, limit: int = 420) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def build_sentiment_observation_payload(
    df_unique: pd.DataFrame,
    df_grouped_rows: pd.DataFrame | None,
    sentiment_type: str,
    include_not_relevant: bool,
    per_sentiment_limit: int = 6,
) -> dict[str, Any]:
    working = df_unique.copy()
    working["Final Sentiment"] = build_final_sentiment_series(working)
    working = working[working["Final Sentiment"].notna()].copy()

    if not include_not_relevant:
        working = working[working["Final Sentiment"] != "NOT RELEVANT"].copy()

    if working.empty:
        return {"distribution": [], "examples_by_sentiment": {}}

    for col in ["Mentions", "Impressions", "Effective Reach", "Group Count"]:
        if col not in working.columns:
            working[col] = 0
        working[col] = pd.to_numeric(working[col], errors="coerce").fillna(0)

    for col in ["Headline", "AI Sentiment Rationale"]:
        if col not in working.columns:
            working[col] = ""
        working[col] = _get_text_series(working, col)

    prime_lookup = pd.DataFrame(columns=["Group ID", "Prime URL", "Prime Outlet", "Prime Type", "Prime Snippet"])
    if isinstance(df_grouped_rows, pd.DataFrame) and not df_grouped_rows.empty and "Group ID" in df_grouped_rows.columns:
        grouped = df_grouped_rows.copy()
        if "Prime Example" in grouped.columns:
            grouped = grouped[grouped["Prime Example"] == 1].copy()
        if grouped.empty:
            grouped = df_grouped_rows.copy()
        for source_col, target_col in [
            ("URL", "Prime URL"),
            ("Example URL", "Prime URL"),
            ("Outlet", "Prime Outlet"),
            ("Example Outlet", "Prime Outlet"),
            ("Type", "Prime Type"),
            ("Example Type", "Prime Type"),
            ("Snippet", "Prime Snippet"),
            ("Example Snippet", "Prime Snippet"),
        ]:
            if source_col in grouped.columns and target_col not in grouped.columns:
                grouped[target_col] = _get_text_series(grouped, source_col)
        keep_cols = [c for c in ["Group ID", "Prime URL", "Prime Outlet", "Prime Type", "Prime Snippet"] if c in grouped.columns]
        prime_lookup = grouped[keep_cols].drop_duplicates(subset=["Group ID"], keep="first").copy()

    working = working.merge(prime_lookup, on="Group ID", how="left")

    working["Display URL"] = _get_text_series(working, "Example URL") if "Example URL" in working.columns else ""
    if "Display URL" not in working.columns:
        working["Display URL"] = ""
    working["Display URL"] = working["Display URL"].where(working["Display URL"] != "", _get_text_series(working, "URL") if "URL" in working.columns else "")
    if "Prime URL" in working.columns:
        working["Display URL"] = working["Display URL"].where(working["Display URL"] != "", _get_text_series(working, "Prime URL"))

    working["Display Outlet"] = _get_text_series(working, "Example Outlet") if "Example Outlet" in working.columns else ""
    working["Display Outlet"] = working["Display Outlet"].where(working["Display Outlet"] != "", _get_text_series(working, "Outlet") if "Outlet" in working.columns else "")
    if "Prime Outlet" in working.columns:
        working["Display Outlet"] = working["Display Outlet"].where(working["Display Outlet"] != "", _get_text_series(working, "Prime Outlet"))

    working["Display Type"] = _get_text_series(working, "Example Type") if "Example Type" in working.columns else ""
    working["Display Type"] = working["Display Type"].where(working["Display Type"] != "", _get_text_series(working, "Type") if "Type" in working.columns else "")
    if "Prime Type" in working.columns:
        working["Display Type"] = working["Display Type"].where(working["Display Type"] != "", _get_text_series(working, "Prime Type"))

    working["Display Snippet"] = _get_text_series(working, "Example Snippet") if "Example Snippet" in working.columns else ""
    working["Display Snippet"] = working["Display Snippet"].where(working["Display Snippet"] != "", _get_text_series(working, "Snippet") if "Snippet" in working.columns else "")
    if "Prime Snippet" in working.columns:
        working["Display Snippet"] = working["Display Snippet"].where(working["Display Snippet"] != "", _get_text_series(working, "Prime Snippet"))

    working["_has_url"] = working["Display URL"].ne("")
    working["_is_online_example"] = working["Display Type"].str.upper().eq("ONLINE")

    distribution_df = build_sentiment_distribution(working.rename(columns={"Final Sentiment": "Assigned Sentiment"}), sentiment_type)
    distribution_records = distribution_df.to_dict(orient="records")

    examples_by_sentiment: dict[str, list[dict[str, Any]]] = {}
    for sentiment, group in working.groupby("Final Sentiment", dropna=False):
        ranked = group.sort_values(
            ["_is_online_example", "_has_url", "Group Count", "Mentions", "Impressions", "Effective Reach"],
            ascending=[False, False, False, False, False, False],
        )
        examples = []
        for _, row in ranked.drop_duplicates(subset=["Headline"], keep="first").head(per_sentiment_limit).iterrows():
            examples.append({
                "group_id": row.get("Group ID", ""),
                "headline": row.get("Headline", ""),
                "outlet": row.get("Display Outlet", ""),
                "url": row.get("Display URL", ""),
                "example_type": row.get("Display Type", ""),
                "mentions": int(row.get("Mentions", 0) or 0),
                "impressions": int(row.get("Impressions", 0) or 0),
                "effective_reach": int(row.get("Effective Reach", 0) or 0),
                "group_count": int(row.get("Group Count", 0) or 0),
                "snippet": _truncate_text(row.get("Display Snippet", ""), 420),
                "sentiment_rationale": _truncate_text(row.get("AI Sentiment Rationale", ""), 220),
            })
        examples_by_sentiment[str(sentiment)] = examples

    return {
        "distribution": distribution_records,
        "examples_by_sentiment": examples_by_sentiment,
    }


def build_sentiment_observation_prompt(
    client_name: str,
    sentiment_type: str,
    include_not_relevant: bool,
    payload: dict[str, Any],
    analysis_context: str = "",
) -> str:
    return f"""
You are helping a media intelligence analyst write concise, report-ready sentiment observations for this analysis focus:
{analysis_context or client_name or 'the client'}.

Use the finalized sentiment distribution and representative grouped stories below.
The examples are intentionally selected from the most syndicated and highest-volume coverage in each sentiment bucket.
Each example may also include an existing AI sentiment rationale; use it as supporting context, but ground your summary in the coverage details themselves.

Return strict JSON with this shape:
{{
  "overall_observation": "1-2 sentences",
  "sentiment_sections": [
    {{
      "sentiment": "LABEL",
      "observation": "1-2 concise sentences"
    }}
  ]
}}

Requirements:
- Keep it factual, neutral, and report-ready.
- Explain what kinds of coverage are driving each sentiment category.
- Do not overstate small buckets.
- If negative or unfavorable coverage is isolated, say so.
- Use only sentiment labels present in the provided data.
- Support both 3-way and 5-way sentiment.
- {"Exclude NOT RELEVANT from the narrative unless it is present in the provided examples." if not include_not_relevant else "Include NOT RELEVANT only if it appears meaningfully in the provided examples."}

Sentiment mode: {sentiment_type}

Input data:
{json.dumps(payload, ensure_ascii=True)}
""".strip()


def generate_sentiment_observations(
    df_unique: pd.DataFrame,
    df_grouped_rows: pd.DataFrame | None,
    client_name: str,
    sentiment_type: str,
    include_not_relevant: bool,
    api_key: str,
    model: str = DEFAULT_SENTIMENT_OBSERVATION_MODEL,
    analysis_context: str = "",
) -> tuple[dict[str, Any], int, int]:
    payload = build_sentiment_observation_payload(
        df_unique=df_unique,
        df_grouped_rows=df_grouped_rows,
        sentiment_type=sentiment_type,
        include_not_relevant=include_not_relevant,
    )
    prompt = build_sentiment_observation_prompt(
        client_name=client_name,
        sentiment_type=sentiment_type,
        include_not_relevant=include_not_relevant,
        payload=payload,
        analysis_context=analysis_context,
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
        raise ValueError("Model did not return valid JSON for sentiment observations.")
    parsed["_examples_by_sentiment"] = payload.get("examples_by_sentiment", {})
    return parsed, in_tok, out_tok
