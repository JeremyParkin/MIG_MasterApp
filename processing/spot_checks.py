# spot_checks.py
from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pandas as pd
from deep_translator import GoogleTranslator
from openai import OpenAI

from utils.api_meter import extract_usage_tokens


# ====================
# Priority defaults
# ====================
W_GROUP = 0.40
W_NEG = 0.35
W_IMP = 0.15
W_LOWCF = 0.10
DEFAULT_CONF_THRESH = 75

DEFAULT_SECOND_OPINION_MODEL = "gpt-5.4-mini"
MAX_RETRIES = 2
DEFAULT_REVIEW_CONFIDENCE_THRESHOLD = 90


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
        "named_entity": result.get("named_entity"),
        "sentiment": sentiment,
        "confidence": confidence,
        "explanation": explanation,
    }, None


# ====================
# State init
# ====================
def init_spot_check_state(session_state) -> None:
    defaults = {
        "initial_ai_label": {},
        "spot_checked_groups": set(),
        "accepted_initial": set(),
        "spot_ai_loading": False,
        "spot_ai_refresh_requested": False,
        "spot_idx": 0,
        "spot_lock_gid": None,
        "spot_ai_model_override": None,
        "spotcheck_low_conf_threshold": DEFAULT_REVIEW_CONFIDENCE_THRESHOLD,
    }
    for key, value in defaults.items():
        if key not in session_state:
            session_state[key] = value


# ====================
# Text / highlighting
# ====================
def escape_markdown(text: str) -> str:
    text = str(text or "")
    markdown_special_chars = r"\`*_{}[]()#+-.!$"
    url_pattern = r"https?:\/\/[^\s]+"

    def esc(part: str) -> str:
        return re.sub(r"([{}])".format(re.escape(markdown_special_chars)), r"\\\1", part)

    parts = re.split(r"(" + url_pattern + r")", text)
    return "".join(part if re.match(url_pattern, part) else esc(part) for part in parts)


def _simple_highlight(text: str, keywords: list[str], bg: str = "goldenrod", fg: str = "black") -> str:
    text = str(text or "")
    if not keywords:
        return text

    escaped = []
    for k in keywords:
        k_esc = re.escape(k)
        escaped.append(rf"\b{k_esc}\b" if " " not in k else k_esc)

    pattern = r"(?:%s)" % "|".join(escaped)

    def repl(m):
        return f"<span style='background-color:{bg};color:{fg};'>{m.group(0)}</span>"

    return re.sub(pattern, repl, text, flags=re.IGNORECASE)


def highlight_with_tolerant_regex(
    text: str,
    tolerant_pat_str: str | None,
    fallback_keywords: list[str],
    bg: str = "goldenrod",
    fg: str = "black",
) -> str:
    s = str(text or "")
    if tolerant_pat_str:
        try:
            rx = re.compile(tolerant_pat_str)
            return rx.sub(
                lambda m: f"<span style='background-color:{bg};color:{fg};'>{m.group(0)}</span>",
                s,
            )
        except re.error:
            pass

    return _simple_highlight(s, fallback_keywords, bg=bg, fg=fg)


# ====================
# Translation helpers
# ====================
def split_text(text: str, limit: int = 700, sentence_limit: int = 350) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", str(text or ""))
    chunks: list[str] = []
    current = ""

    for s in sentences:
        s = s or ""
        while len(s) > sentence_limit:
            part, s = s[:sentence_limit], s[sentence_limit:]
            current = (current + " " + part).strip() if current else part
            if len(current) >= limit:
                chunks.append(current)
                current = ""

        if len(current) + len(s) <= limit:
            current += (" " if current else "") + s
        else:
            if current:
                chunks.append(current)
            current = s

    if current:
        chunks.append(current)

    return chunks


def translate_concurrently(chunks: list[str]) -> list[str]:
    translator = GoogleTranslator(source="auto", target="en")
    results: list[str | None] = [None] * len(chunks)

    with ThreadPoolExecutor(max_workers=20) as ex:
        futures = [(i, ex.submit(translator.translate, c)) for i, c in enumerate(chunks)]
        for i, fut in futures:
            try:
                results[i] = fut.result()
            except Exception as e:
                results[i] = f"Error: {e}"

    return [str(x or "") for x in results]


def translate_text(text: str) -> str:
    return " ".join(translate_concurrently(split_text(text)))


def apply_translation_to_group(
    df_unique: pd.DataFrame,
    df_grouped: pd.DataFrame,
    group_id: int,
    translated_headline: str | None,
    translated_body: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    unique = df_unique.copy()
    grouped = df_grouped.copy()

    for df in [unique, grouped]:
        for col in ["Translated Headline", "Translated Body"]:
            if col not in df.columns:
                df[col] = pd.NA

    for df in [unique, grouped]:
        mask = df["Group ID"] == group_id
        df.loc[mask, ["Translated Headline", "Translated Body"]] = [translated_headline, translated_body]

    return unique, grouped


# ====================
# Candidate scoring
# ====================
def get_group_count(df_unique: pd.DataFrame, df_grouped: pd.DataFrame, gid: int) -> int:
    try:
        val = df_unique.loc[df_unique["Group ID"] == gid, "Group Count"]
        if len(val) and pd.notna(val.iloc[0]):
            return int(val.iloc[0])
    except Exception:
        pass

    try:
        return int((df_grouped["Group ID"] == gid).sum())
    except Exception:
        return 1


def compute_candidates(
    df_unique: pd.DataFrame,
    df_grouped: pd.DataFrame,
    sentiment_type: str,
    conf_thresh: int = DEFAULT_CONF_THRESH,
) -> pd.DataFrame:
    us = df_unique.copy()
    dt = df_grouped.copy()

    assigned_us = set(us.loc[us["Assigned Sentiment"].notna(), "Group ID"].unique()) if "Assigned Sentiment" in us.columns else set()
    assigned_dt = set(dt.loc[dt["Assigned Sentiment"].notna(), "Group ID"].unique()) if "Assigned Sentiment" in dt.columns else set()
    assigned_any = assigned_us | assigned_dt

    pool = us[(~us["Group ID"].isin(assigned_any)) & (us["AI Sentiment"].notna())].copy()
    if pool.empty:
        return pool

    pool["AI_UPPER"] = pool["AI Sentiment"].astype(str).str.upper().str.strip()
    pool["AI_CONF"] = pd.to_numeric(pool["AI Sentiment Confidence"], errors="coerce").fillna(100)
    pool["GROUP_CT"] = pd.to_numeric(pool.get("Group Count", 1), errors="coerce").fillna(1)

    if sentiment_type == "3-way":
        pool["NEG_SCORE"] = pool["AI_UPPER"].map({"NEGATIVE": 1.0}).fillna(0.0)
    else:
        pool["NEG_SCORE"] = pool["AI_UPPER"].map(
            {"VERY NEGATIVE": 1.0, "SOMEWHAT NEGATIVE": 0.7}
        ).fillna(0.0)

    conf_thresh = max(1, int(conf_thresh))
    pool["LOWCONF"] = ((conf_thresh - pool["AI_CONF"]) / conf_thresh).clip(lower=0, upper=1)

    max_gc = pool["GROUP_CT"].max()
    pool["GC_NORM"] = (pool["GROUP_CT"] / max_gc) if max_gc > 0 else 0.0

    imp_col = next((c for c in ["Impressions", "impressions", "IMPRESSIONS"] if c in pool.columns), None)
    if imp_col:
        pool["_IMP"] = pd.to_numeric(pool[imp_col], errors="coerce").fillna(0)
        max_imp = pool["_IMP"].max()
        pool["IMP_NORM"] = (pool["_IMP"] / max_imp) if max_imp > 0 else 0.0
    else:
        pool["IMP_NORM"] = 0.0

    pool["SCORE"] = (
        W_GROUP * pool["GC_NORM"] +
        W_NEG * pool["NEG_SCORE"] +
        W_IMP * pool["IMP_NORM"] +
        W_LOWCF * pool["LOWCONF"]
    )

    pool = pool.sort_values(["SCORE", "GROUP_CT", "IMP_NORM"], ascending=[False, False, False]).reset_index(drop=True)
    return pool


# ====================
# Sentiment second opinion
# ====================
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
    parts.append(f"HEADLINE: {headline}")
    parts.append(f"BODY: {snippet}")
    return "\n\n".join(parts)


def call_ai_sentiment(
    story_prompt: str,
    model_to_use: str,
    functions: list[dict[str, Any]],
    sentiment_type: str,
    api_key: str,
) -> tuple[dict[str, Any] | None, int, int, str]:
    client = OpenAI(api_key=api_key)

    try:
        if functions:
            resp = client.chat.completions.create(
                model=model_to_use,
                messages=[
                    {"role": "system", "content": "You are a highly knowledgeable media analysis AI."},
                    {"role": "user", "content": story_prompt},
                ],
                functions=functions,
                function_call={"name": "analyze_sentiment"},
            )
            in_tok, out_tok = extract_usage_tokens(resp)
            choice = resp.choices[0]

            if getattr(choice.message, "function_call", None):
                fc = choice.message.function_call
                if fc and fc.name == "analyze_sentiment":
                    args = json.loads(fc.arguments or "{}")
                    parsed, err = _validate_structured_result(args, sentiment_type)
                    if not err:
                        return parsed, in_tok, out_tok, ""
                    fallback_note = f"Function output invalid: {err}"
                else:
                    fallback_note = "Function output missing expected tool call."
            else:
                fallback_note = "Function output missing tool payload."
    except Exception as e:
        fallback_note = f"Function-calling fallback used due to: {e}"

    try:
        resp = client.chat.completions.create(
            model=model_to_use,
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
        in_tok, out_tok = extract_usage_tokens(resp)
        txt = (resp.choices[0].message.content or "").strip()

        payload = _extract_json_payload(txt)
        if payload is None:
            return None, in_tok, out_tok, "Structured output missing JSON payload."

        parsed, err = _validate_structured_result(payload, sentiment_type)
        if err:
            return None, in_tok, out_tok, f"Structured output validation failed: {err}"

        return parsed, in_tok, out_tok, fallback_note
    except Exception as e:
        return None, 0, 0, f"AI sentiment failed: {e}"


# ====================
# Write-backs / stats
# ====================
def write_second_opinion_to_group(
    df_unique: pd.DataFrame,
    df_grouped: pd.DataFrame,
    group_id: int,
    label: str | None,
    confidence: Any,
    rationale: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    unique = df_unique.copy()
    grouped = df_grouped.copy()

    for df in [unique, grouped]:
        mask = df["Group ID"] == group_id
        df.loc[mask, ["AI Sentiment", "AI Sentiment Confidence", "AI Sentiment Rationale"]] = [
            label,
            confidence,
            rationale,
        ]

    return unique, grouped


def set_assigned_sentiment(
    df_unique: pd.DataFrame,
    df_grouped: pd.DataFrame,
    group_id: int,
    label: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    unique = df_unique.copy()
    grouped = df_grouped.copy()

    for df in [unique, grouped]:
        mask = df["Group ID"] == group_id
        df.loc[mask, "Assigned Sentiment"] = label

    return unique, grouped


def update_acceptance_tracking(
    session_state,
    group_id: int,
    final_label: str,
) -> None:
    session_state.spot_checked_groups.add(group_id)

    init_map = session_state.initial_ai_label
    if (
        group_id in init_map
        and final_label.strip().upper() == str(init_map[group_id]).strip().upper()
    ):
        session_state.accepted_initial.add(group_id)


from concurrent.futures import ThreadPoolExecutor, as_completed


def ensure_review_columns(
    df_unique: pd.DataFrame,
    df_grouped: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    unique = df_unique.copy()
    grouped = df_grouped.copy()

    review_cols = {
        "Review AI Sentiment": pd.NA,
        "Review AI Confidence": pd.NA,
        "Review AI Rationale": pd.NA,
        "AI Agreement": pd.NA,
        "Needs Human Review": pd.NA,
    }

    for df in [unique, grouped]:
        for col, default_val in review_cols.items():
            if col not in df.columns:
                df[col] = default_val

    return unique, grouped


def write_review_opinion_to_group(
    df_unique: pd.DataFrame,
    df_grouped: pd.DataFrame,
    group_id: int,
    review_label: str | None,
    review_confidence,
    review_rationale: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    unique = df_unique.copy()
    grouped = df_grouped.copy()

    for df in [unique, grouped]:
        mask = df["Group ID"] == group_id
        df.loc[mask, "Review AI Sentiment"] = review_label
        df.loc[mask, "Review AI Confidence"] = review_confidence
        df.loc[mask, "Review AI Rationale"] = review_rationale

    return unique, grouped

def apply_review_flags_to_group(
    df_unique: pd.DataFrame,
    df_grouped: pd.DataFrame,
    group_id: int,
    ai_label: str | None,
    ai_confidence,
    review_label: str | None,
    review_confidence,
    low_conf_threshold: int = 60,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    unique = df_unique.copy()
    grouped = df_grouped.copy()

    ai_label_clean = str(ai_label or "").strip().upper()
    review_label_clean = str(review_label or "").strip().upper()

    agreement = bool(ai_label_clean and review_label_clean and ai_label_clean == review_label_clean)

    review_conf = pd.to_numeric(pd.Series([review_confidence]), errors="coerce").iloc[0]

    # New rule:
    # match + high review confidence => no human review needed
    if agreement and pd.notna(review_conf) and review_conf >= low_conf_threshold:
        needs_human = False
    else:
        needs_human = True

    for df in [unique, grouped]:
        mask = df["Group ID"] == group_id
        df.loc[mask, "AI Agreement"] = "Match" if agreement else "Disagree"
        df.loc[mask, "Needs Human Review"] = "Yes" if needs_human else "No"

    return unique, grouped

# def apply_review_flags_to_group(
#     df_unique: pd.DataFrame,
#     df_grouped: pd.DataFrame,
#     group_id: int,
#     ai_label: str | None,
#     ai_confidence,
#     review_label: str | None,
#     review_confidence,
#     low_conf_threshold: int = 60,
# ) -> tuple[pd.DataFrame, pd.DataFrame]:
#     unique = df_unique.copy()
#     grouped = df_grouped.copy()
#
#     ai_label_clean = str(ai_label or "").strip().upper()
#     review_label_clean = str(review_label or "").strip().upper()
#
#     agreement = bool(ai_label_clean and review_label_clean and ai_label_clean == review_label_clean)
#
#     ai_conf = pd.to_numeric(pd.Series([ai_confidence]), errors="coerce").iloc[0]
#     review_conf = pd.to_numeric(pd.Series([review_confidence]), errors="coerce").iloc[0]
#
#     low_conf = False
#     if pd.notna(ai_conf) and ai_conf < low_conf_threshold:
#         low_conf = True
#     if pd.notna(review_conf) and review_conf < low_conf_threshold:
#         low_conf = True
#
#     needs_human = (not agreement) or low_conf
#
#     for df in [unique, grouped]:
#         mask = df["Group ID"] == group_id
#         df.loc[mask, "AI Agreement"] = "Match" if agreement else "Disagree"
#         df.loc[mask, "Needs Human Review"] = "Yes" if needs_human else "No"
#
#     return unique, grouped

def auto_assign_resolved_match_to_group(
    df_unique: pd.DataFrame,
    df_grouped: pd.DataFrame,
    group_id: int,
    ai_label: str | None,
    review_label: str | None,
    review_confidence,
    confidence_threshold: int = 60,
) -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    unique = df_unique.copy()
    grouped = df_grouped.copy()

    ai_label_clean = str(ai_label or "").strip().upper()
    review_label_clean = str(review_label or "").strip().upper()
    review_conf = pd.to_numeric(pd.Series([review_confidence]), errors="coerce").iloc[0]

    should_auto_assign = (
        ai_label_clean != ""
        and ai_label_clean == review_label_clean
        and pd.notna(review_conf)
        and review_conf >= confidence_threshold
    )

    if not should_auto_assign:
        return unique, grouped, False

    for df in [unique, grouped]:
        mask = df["Group ID"] == group_id
        df.loc[mask, "Assigned Sentiment"] = ai_label_clean

    return unique, grouped, True


def second_opinion_worker(
    row_tuple,
    pre_prompt: str,
    sentiment_instruction: str,
    post_prompt: str,
    functions: list[dict[str, Any]],
    review_model: str,
    sentiment_type: str,
    api_key: str,
) -> tuple[int, dict[str, Any], str, int, int]:
    idx, row_dict = row_tuple
    row = pd.Series(row_dict)

    headline = row.get("Headline", "") or ""
    snippet = row.get("Example Snippet", row.get("Snippet", "")) or ""

    story_prompt = build_story_prompt(
        headline=headline,
        snippet=snippet,
        pre_prompt=pre_prompt,
        sentiment_instruction=sentiment_instruction,
        post_prompt=post_prompt,
    )

    result, in_tok, out_tok, note = call_ai_sentiment(
        story_prompt=story_prompt,
        model_to_use=review_model,
        functions=functions,
        sentiment_type=sentiment_type,
        api_key=api_key,
    )

    if result is None:
        return idx, {}, note or "Second opinion failed.", 0, 0

    return idx, result, note or "", in_tok, out_tok


def run_batch_second_opinion(
    candidates_df: pd.DataFrame,
    df_unique: pd.DataFrame,
    df_grouped: pd.DataFrame,
    pre_prompt: str,
    sentiment_instruction: str,
    post_prompt: str,
    functions: list[dict[str, Any]],
    sentiment_type: str,
    api_key: str,
    review_model: str = DEFAULT_SECOND_OPINION_MODEL,
    limit: int = 50,
    max_workers: int = 8,
    low_conf_threshold: int = 60,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], int, int]:
    unique, grouped = ensure_review_columns(df_unique, df_grouped)

    working_candidates = candidates_df.head(limit).copy()
    rows_for_workers = [(idx, row.to_dict()) for idx, row in working_candidates.iterrows()]

    total_in = 0
    total_out = 0
    errors: list[str] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(
                second_opinion_worker,
                row_tuple,
                pre_prompt,
                sentiment_instruction,
                post_prompt,
                functions,
                review_model,
                sentiment_type,
                api_key,
            ): row_tuple[0]
            for row_tuple in rows_for_workers
        }

        for future in as_completed(future_map):
            idx = future_map[future]

            try:
                _, result, note, in_tok, out_tok = future.result()
                total_in += int(in_tok or 0)
                total_out += int(out_tok or 0)

                row = working_candidates.loc[idx]
                gid = int(row["Group ID"])

                review_label = result.get("sentiment")
                review_conf = result.get("confidence")
                review_rsn = result.get("explanation")

                unique, grouped = write_review_opinion_to_group(
                    unique,
                    grouped,
                    gid,
                    review_label,
                    review_conf,
                    review_rsn,
                )

                unique, grouped = apply_review_flags_to_group(
                    unique,
                    grouped,
                    gid,
                    ai_label=row.get("AI Sentiment"),
                    ai_confidence=row.get("AI Sentiment Confidence"),
                    review_label=review_label,
                    review_confidence=review_conf,
                    low_conf_threshold=low_conf_threshold,
                )

                unique, grouped, _ = auto_assign_resolved_match_to_group(
                    unique,
                    grouped,
                    gid,
                    ai_label=row.get("AI Sentiment"),
                    review_label=review_label,
                    review_confidence=review_conf,
                    confidence_threshold=low_conf_threshold,
                )

                if note:
                    errors.append(f"Group {gid}: {note}")

            except Exception as e:
                errors.append(f"Candidate {idx}: {e}")

    return unique, grouped, errors, total_in, total_out
