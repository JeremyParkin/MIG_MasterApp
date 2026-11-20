"""Article evidence and request handling shared by Insights generators."""

import json

import pandas as pd
from openai import BadRequestError

PRIMARY_STORY_LIMIT = 20
PRIMARY_ARTICLE_TEXT = "Primary Article Text"


def primary_article_text(story, *, prefer_example: bool = False) -> str:
    """Use a dedicated body when available, then the fullest existing snippet."""
    def text(field):
        value = story.get(field)
        return value if isinstance(value, str) and value.strip() else ""

    example_url = text("Example URL").strip()
    source_url = text("URL").strip()
    if prefer_example and example_url and source_url and example_url != source_url:
        return text("Example Snippet")

    return (
        text(PRIMARY_ARTICLE_TEXT) or text("Full Text") or text("Article Text")
        or max((text("Snippet"), text("Example Snippet")), key=len)
    )


def request_insights_response(client, **request):
    """Keep all evidence in the request, and explain context-limit rejections."""
    request["truncation"] = "disabled"
    try:
        return client.responses.create(**request)
    except BadRequestError as exc:
        message = str(exc).lower()
        if exc.code != "context_length_exceeded" and not any(
            phrase in message for phrase in ("context window", "context length", "maximum input tokens")
        ):
            raise
        size = len(json.dumps(request, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
        raise ValueError(
            f"The Insights request exceeds the context capacity of {request['model']} "
            f"({size:,} serialized bytes). Use a smaller coverage scope or fewer labels per Insights request "
            "where available, then try again. Primary article text was kept in full."
        ) from exc


ARTICLE_REFERENCE_GUIDANCE = (
    "Use each primary example's article body from articles_by_id, keyed by its story_id. "
    "A story can appear in multiple evidence lists. Use each list's order and attribution, "
    "and treat linked pickup rows as coverage of the same story."
)


def story_identity(group_id, headline) -> str:
    if pd.notna(group_id) and str(group_id).strip():
        return str(group_id).strip()
    return f"ungrouped:{headline}"


def ranked_story_keys(rows: pd.DataFrame) -> pd.Series:
    groups = rows.get("Group ID", pd.Series("", index=rows.index))
    return pd.Series(
        [story_identity(group, headline) for group, headline in zip(groups, rows["Headline"])],
        index=rows.index, dtype="object",
    )


def distinct_ranked_stories(rows: pd.DataFrame) -> pd.DataFrame:
    """Keep the highest-ranked row of each group, falling back to the headline for legacy rows."""
    return rows.loc[~ranked_story_keys(rows).duplicated()]


def share_primary_articles(payload, *primary_fields, group_field="group_id", headline_field="headline", body_field="snippet"):
    """Return a wire payload without changing the examples retained by the app."""
    articles = {}

    def reference(examples):
        refs = []
        for example in examples:
            entry = example.copy()
            story_id = story_identity(entry.get(group_field), entry.get(headline_field, ""))
            body = entry.pop(body_field, "")
            if not articles.get(story_id):
                articles[story_id] = body
            entry["story_id"] = story_id
            refs.append(entry)
        return refs

    result = dict(payload)
    for field in primary_fields:
        examples = payload.get(field, [])
        result[field] = (
            {label: reference(rows) for label, rows in examples.items()}
            if isinstance(examples, dict) else reference(examples)
        )
    result["articles_by_id"] = articles
    return result
