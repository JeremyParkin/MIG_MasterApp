from __future__ import annotations

import html
from typing import Iterable, Mapping


def build_linked_example_blocks_html(
    items: Iterable[Mapping[str, object]],
    *,
    title_key: str = "headline",
    url_key: str = "url",
    outlet_key: str = "outlet",
    type_key: str = "example_type",
    mentions_key: str = "mentions",
    impressions_key: str = "impressions",
    effective_reach_key: str = "effective_reach",
    show_outlet: bool = True,
    show_media_type: bool = True,
    show_mentions: bool = True,
    show_impressions: bool = True,
    show_effective_reach: bool = True,
) -> str:
    blocks: list[str] = []

    for item in items:
        headline = str(item.get(title_key, "") or "").strip()
        if not headline:
            continue

        url = str(item.get(url_key, "") or "").strip()
        outlet = str(item.get(outlet_key, "") or "").strip()
        media_type = str(item.get(type_key, "") or "").strip()
        mentions = int(item.get(mentions_key, 0) or 0)
        impressions = int(item.get(impressions_key, 0) or 0)
        effective_reach = int(item.get(effective_reach_key, 0) or 0)

        meta_parts: list[str] = []
        if show_outlet and outlet:
            meta_parts.append(outlet)
        if show_media_type and media_type:
            meta_parts.append(media_type)

        metric_parts: list[str] = []
        if show_mentions:
            metric_parts.append(f"Mentions: {mentions:,}")
        if show_impressions:
            metric_parts.append(f"Impressions: {impressions:,}")
        if show_effective_reach:
            metric_parts.append(f"Effective Reach: {effective_reach:,}")

        meta_line = " | ".join(meta_parts + metric_parts)
        headline_html = (
            f'<a href="{html.escape(url, quote=True)}" target="_blank">{html.escape(headline)}</a>'
            if url
            else html.escape(headline)
        )
        metrics_html = (
            f'<div style="font-size:0.84rem; opacity:0.72; letter-spacing:0.01em; margin-top:0.12rem;">{html.escape(meta_line)}</div>'
            if meta_line
            else ""
        )
        blocks.append(
            '<div style="margin:0 0 0.7rem 0;">'
            f'<div style="line-height:1.35;">{headline_html}</div>'
            f"{metrics_html}"
            "</div>"
        )

    return "".join(blocks)
