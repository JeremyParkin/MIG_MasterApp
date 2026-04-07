from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Tuple

import pandas as pd
from openai import OpenAI
from utils.api_meter import add_api_usage
from utils.api_meter import extract_usage_tokens


TYPE_DICT = {
    "RADIO": "broadcast transcript",
    "TV": "broadcast transcript",
    "PODCAST": "broadcast transcript",
    "ONLINE": "online article",
    "PRINT": "print article",
}

DEFAULT_MODEL = "gpt-5.4-mini"
SHORT_SNIPPET_THRESHOLD = 150
DEFAULT_MAX_WORKERS = 8
MAX_RETRIES = 2


def init_top_story_summary_state(session_state) -> None:
    for key, default in {
        "top_story_entity_names": [],
        "top_story_spokespeople": [],
        "top_story_products": [],
        "top_story_guidance": "",
        "top_story_entity_names_seeded": False,
    }.items():
        if key not in session_state:
            session_state[key] = default


def normalize_summary_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize fields needed for this page."""
    df = df.copy()

    defaults = {
        "Group ID": pd.NA,
        "Headline": "",
        "Example Outlet": "",
        "Example URL": "",
        "Example Type": "",
        "Example Snippet": "",
        "Mentions": 0,
        "Impressions": 0,
        "Chart Callout": "",
        "Top Story Summary": "",
        "Entity Sentiment Label": "",
        "Entity Sentiment Rationale": "",
        "Entity Sentiment": "",
    }

    for col, default in defaults.items():
        if col not in df.columns:
            df[col] = default

    text_cols = [
        "Headline",
        "Example Outlet",
        "Example URL",
        "Example Type",
        "Example Snippet",
        "Chart Callout",
        "Top Story Summary",
        "Entity Sentiment Label",
        "Entity Sentiment Rationale",
        "Entity Sentiment",
    ]
    for col in text_cols:
        df[col] = df[col].fillna("").astype(str)

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    if "Mentions" in df.columns:
        df["Mentions"] = pd.to_numeric(df["Mentions"], errors="coerce").fillna(0).astype(int)

    if "Impressions" in df.columns:
        df["Impressions"] = pd.to_numeric(df["Impressions"], errors="coerce").fillna(0).astype(int)

    return df


def escape_markdown(text: str) -> str:
    markdown_special_chars = r"\`*_{}[]()#+-.!$"
    pattern = r"([" + re.escape(markdown_special_chars) + r"])"
    return re.sub(pattern, r"\\\1", str(text))


def seed_entity_names(session_state, client_name: str) -> None:
    if session_state.get("top_story_entity_names_seeded", False):
        return

    current_entity_names = session_state.get("top_story_entity_names", [])
    current_entity_names = [str(x).strip() for x in current_entity_names if str(x).strip()]

    client_name = str(client_name or "").strip()
    if client_name and not any(x.lower() == client_name.lower() for x in current_entity_names):
        current_entity_names = [client_name] + current_entity_names

    session_state.top_story_entity_names = current_entity_names
    session_state.top_story_entity_names_seeded = True


def build_entity_context(
    primary_name: str,
    alternate_names: List[str],
    spokespeople: List[str],
    products: List[str],
    additional_guidance: str,
) -> str:
    lines = [f"Primary entity: {primary_name.strip()}"]

    if alternate_names:
        lines.append("Alternate names / aliases: " + "; ".join(alternate_names))
    if spokespeople:
        lines.append("Key spokespeople: " + "; ".join(spokespeople))
    if products:
        lines.append("Products / sub-brands / initiatives: " + "; ".join(products))
    if additional_guidance.strip():
        lines.append("Additional user guidance: " + additional_guidance.strip())

    lines.append(
        "Treat references to alternate names, spokespeople, products, sub-brands, and initiatives as relevant to the primary entity only when the coverage clearly maps them back to that entity."
    )

    return "\n".join(lines)


def build_master_prompt(
    row: pd.Series,
    entity_context: str,
) -> str:
    example_type = row.get("Example Type", "")
    snippet = row.get("Example Snippet", "")
    headline = row.get("Headline", "")
    story_type = TYPE_DICT.get(example_type, "news story")
    outlet = row.get("Example Outlet", "")

    if example_type in ["RADIO", "TV", "PODCAST"]:
        source_guidance = (
            "The source is a broadcast transcript. Broadcast transcripts may contain unrelated advertisements, tosses, or other segments that should be ignored."
        )
    else:
        source_guidance = f"The source is a {story_type}."

    return f"""
You are a media intelligence analyst producing structured outputs for an executive report.

ENTITY CONTEXT
{entity_context}

OUTPUTS REQUIRED
Return all of the following:
1. chart_callout
2. top_story_summary
3. entity_sentiment_label
4. entity_sentiment_rationale

GLOBAL RULES
- Stay neutral and factual.
- Do not invent facts, implications, motives, or significance.
- Base everything only on the story content provided.
- The brand/entity may appear directly or indirectly through aliases, spokespeople, products, sub-brands, or initiatives.
- If the entity is secondary to the main story, make that clear rather than overstating its importance.
- Avoid vague shorthand like "faces challenges" or "under scrutiny" unless you specify what that means in this story.
- Use present tense where natural.
- No markdown, bullets, or labels inside field values.
- Return all outputs in English

FIELD-SPECIFIC RULES

chart_callout:
- One sentence.
- Usually about 12-20 words.
- Suitable for a trend-chart annotation.
- Must clearly say what the story is about and how the entity appears in it.
- Do not sound promotional or analytical.
- Do not mention the outlet unless it materially adds value.

top_story_summary:
- One sentence.
- Usually about 30-50 words.
- Executive-style summary of the story and the entity's role in it.
- Slightly fuller than the chart callout, but still concise.

entity_sentiment_label:
- Must be exactly one of: Positive, Neutral, Negative.

entity_sentiment_rationale:
- One short sentence (typically 8–18 words).
- Explain the sentiment toward the primary entity specifically, not the broader topic.
- Consider references to aliases, spokespeople, products, and sub-brands where relevant.
- Explain WHY the sentiment label applies based on specific coverage dynamics.
- Do NOT repeat or restate the sentiment label.
- Do NOT define the sentiment category.
- Focus on what the coverage says or does regarding the entity.
- Use concrete signals.

STORY INPUT
Headline: {headline}
Outlet: {outlet}
Type: {example_type or "Unknown"}
{source_guidance}

Body:
{snippet}
""".strip()


def build_prompt_preview(entity_context: str) -> str:
    preview_row = pd.Series({
        "Headline": "Example headline placeholder",
        "Example Outlet": "Example outlet",
        "Example Type": "ONLINE",
        "Example Snippet": "Example story text placeholder showing how the AI will interpret coverage.",
    })

    return build_master_prompt(
        row=preview_row,
        entity_context=entity_context,
    )


def get_structured_schema() -> Dict[str, Any]:
    return {
        "name": "top_story_outputs",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "chart_callout": {"type": "string"},
                "top_story_summary": {"type": "string"},
                "entity_sentiment_label": {
                    "type": "string",
                    "enum": ["Positive", "Neutral", "Negative"],
                },
                "entity_sentiment_rationale": {"type": "string"},
            },
            "required": [
                "chart_callout",
                "top_story_summary",
                "entity_sentiment_label",
                "entity_sentiment_rationale",
            ],
        },
    }


def extract_response_text(response) -> str:
    """Robustly extract output text from a Responses API response."""
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


def generate_structured_story_outputs(
    client: OpenAI,
    prompt: str,
    model: str = DEFAULT_MODEL,
) -> Dict[str, str]:
    schema = get_structured_schema()

    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "system",
                "content": "You are a highly skilled media intelligence analyst who produces concise, structured, report-ready outputs.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        text={
            "verbosity": "low",
            "format": {
                "type": "json_schema",
                "name": schema["name"],
                "strict": schema["strict"],
                "schema": schema["schema"],
            },
        },
    )

    add_api_usage(response, model)
    in_tok, out_tok = extract_usage_tokens(response)

    raw_text = extract_response_text(response).strip()
    if not raw_text:
        raise ValueError("No structured output text was returned.")

    parsed = json.loads(raw_text)

    sentiment_label = parsed.get("entity_sentiment_label", "").strip()
    sentiment_rationale = parsed.get("entity_sentiment_rationale", "").strip()

    parsed["entity_sentiment_label"] = sentiment_label
    parsed["entity_sentiment_rationale"] = sentiment_rationale
    parsed["entity_sentiment"] = f"{sentiment_label}: {sentiment_rationale}".strip(": ").strip()

    return parsed, in_tok, out_tok


def generate_outputs_for_row(
    row_tuple: Tuple[int, Dict[str, Any]],
    entity_context: str,
    api_key: str,
) -> Tuple[int, Dict[str, str], str]:
    i, row_dict = row_tuple
    row = pd.Series(row_dict)
    snippet = str(row.get("Example Snippet", "") or "")

    in_tok = 0
    out_tok = 0

    if len(snippet) < SHORT_SNIPPET_THRESHOLD:
        return i, {
            "Chart Callout": "Snippet too short to generate callout",
            "Top Story Summary": "Snippet too short to generate summary",
            "Entity Sentiment Label": "Neutral",
            "Entity Sentiment Rationale": "Snippet is too short to assess sentiment reliably.",
            "Entity Sentiment": "Neutral: Snippet is too short to assess sentiment reliably.",
        }, ""

    prompt = build_master_prompt(row, entity_context)

    last_error = ""
    client = OpenAI(api_key=api_key)
    for _ in range(MAX_RETRIES + 1):
        try:
            parsed, in_tok, out_tok = generate_structured_story_outputs(
                client=client,
                prompt=prompt,
                model=DEFAULT_MODEL,
            )
            return i, {
                "Chart Callout": parsed.get("chart_callout", "").strip(),
                "Top Story Summary": parsed.get("top_story_summary", "").strip(),
                "Entity Sentiment Label": parsed.get("entity_sentiment_label", "").strip(),
                "Entity Sentiment Rationale": parsed.get("entity_sentiment_rationale", "").strip(),
                "Entity Sentiment": parsed.get("entity_sentiment", "").strip(),
            }, "", in_tok, out_tok
        except Exception as e:
            last_error = str(e)

    return i, {}, last_error, in_tok, out_tok


def generate_outputs_for_dataframe(
    df: pd.DataFrame,
    entity_context: str,
    api_key: str,
    max_workers: int = DEFAULT_MAX_WORKERS,
) -> tuple[pd.DataFrame, list[str]]:
    working_df = normalize_summary_df(df.copy())
    rows_for_workers = [(i, row.to_dict()) for i, row in working_df.iterrows()]
    errors: list[str] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(
                generate_outputs_for_row,
                row_tuple,
                entity_context,
                api_key,
            ): row_tuple[0]
            for row_tuple in rows_for_workers
        }

        for future in as_completed(future_map):
            i = future_map[future]
            try:
                row_index, outputs, error_message = future.result()

                if error_message:
                    errors.append(f"Story {row_index + 1}: {error_message}")
                else:
                    for col, value in outputs.items():
                        working_df.at[row_index, col] = value

            except Exception as e:
                errors.append(f"Story {i + 1}: {e}")

    return working_df, errors


def build_markdown_output(
    df: pd.DataFrame,
    show_callout: bool,
    show_top_story_summary: bool,
    show_sentiment: bool,
    show_mentions: bool,
    show_impressions: bool,
) -> str:
    markdown_content = ""

    for _, row in df.iterrows():
        head = escape_markdown(row.get("Headline", ""))
        outlet = escape_markdown(row.get("Example Outlet", ""))
        link = escape_markdown(row.get("Example URL", ""))
        date_val = row.get("Date")

        if pd.notna(date_val):
            date_text = pd.to_datetime(date_val).strftime("%B %d, %Y")
        else:
            date_text = ""

        markdown_content += f"__[{head}]({link})__  \n"
        markdown_content += f"_{outlet}_"
        if date_text:
            markdown_content += f" – {date_text}"
        markdown_content += "  \n"

        if show_top_story_summary and "Top Story Summary" in df.columns:
            value = row.get("Top Story Summary", "")
            if pd.notna(value) and str(value).strip():
                markdown_content += f"{value}  \n\n"

        if show_callout and "Chart Callout" in df.columns:
            value = row.get("Chart Callout", "")
            if pd.notna(value) and str(value).strip():
                markdown_content += f":material/chat_bubble: **Trend Chart Callout:** <br>{value}  \n\n"

        if show_sentiment and "Entity Sentiment" in df.columns:
            value = row.get("Entity Sentiment", "")
            if pd.notna(value) and str(value).strip():
                markdown_content += f":material/pie_chart: **Sentiment Opinion:** <br>_{value}_  \n\n"

        if show_mentions:
            mentions = row.get("Mentions", 0)
            markdown_content += f"**Mentions**: {mentions} &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"

        if show_impressions:
            impressions = row.get("Impressions", 0)
            markdown_content += f"**Impressions**: {int(impressions):,}"

        if show_mentions or show_impressions:
            markdown_content += "<br>"

        markdown_content += "<br>"

    return markdown_content