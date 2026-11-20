import json
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import pandas as pd
import pytest
from openai import BadRequestError

from processing import ai_sentiment, ai_tagging, author_insights, outlet_insights, rapid_insights, regions, top_stories, top_story_summaries
from processing.rapid_resolution import RAPID_TAG_REVIEW_MODE_APPLICABLE


def stories(count, *, paragraphs=55, tag_count=1):
    tags = {f"Theme {index}": "Fictional report topic" for index in range(tag_count)}
    return pd.DataFrame([
        {
            "Group ID": n,
            "Group Count": 1,
            "Prime Example": 1,
            "Headline": f"Transit dispatch {n}",
            "Author": "Alex Example",
            "Outlet": "Metro Bulletin",
            "URL": f"https://example.test/{n}",
            "Type": "ONLINE",
            "Country": "Canada",
            "Date": pd.Timestamp("2026-11-01"),
            "Mentions": 1,
            "Impressions": 1_000 - n,
            "Effective Reach": 800 - n,
            "Snippet": (
                f"Dispatch {n} opens with a note on the new route.\n\n"
                + "".join(f"Section {i} details café service near the station and passenger feedback.\n" for i in range(paragraphs))
                + f"End of the original article {n}."
            ),
            "Example Snippet": f"Dispatch {n} opens with a note on the new route.",
            "AI Sentiment": "NEUTRAL",
            "AI Sentiment Confidence": 95,
            "Review AI Confidence": None,
            "AI Tag": ", ".join(tags),
            "Jev Sentiment": "NEUTRAL",
            "Jev 5-Way Sentiment": "NEUTRAL",
            "Jev Best Tag": next(iter(tags)),
            "Jev Tags": ", ".join(tags),
        }
        for n in range(count)
    ]), tags


BASE = {"client_name": "Example Transit", "api_key": "mock"}


def author_request(frame, tags):
    metrics, groups = author_insights.build_author_metrics(frame)
    headline_df = author_insights.build_author_headline_table(groups, "Alex Example", limit=60, include_article_text=True)
    return author_insights.generate_author_summary(
        author_name="Alex Example", author_row=metrics.iloc[0], headline_df=headline_df, **BASE,
    )


def outlet_request(frame, tags):
    metrics, groups = outlet_insights.build_outlet_metrics(frame)
    headline_df = outlet_insights.build_outlet_headline_table(groups, "Metro Bulletin", limit=60, include_article_text=True)
    return outlet_insights.generate_outlet_summary(
        outlet_name="Metro Bulletin", outlet_row=metrics.iloc[0], headline_df=headline_df,
        top_authors_df=pd.DataFrame(), **BASE,
    )


def region_request(frame, tags):
    source = regions.build_regions_source_df(frame)
    return regions.generate_regions_insight_output(
        metric_label="Mentions", filtered_df=source, level_label="Countries", level_key="Country",
        ranking_df=regions.build_region_rankings(source, "Country", metric_label="Mentions"),
        analysis_context="", **BASE,
    )


def sentiment_request(frame, tags):
    return ai_sentiment.generate_sentiment_observations(
        frame, frame, sentiment_type="3-way", include_not_relevant=False, **BASE,
    )


def tagging_request(frame, tags):
    return ai_tagging.generate_tag_observations(frame, include_other=False, **BASE)


def rapid_sentiment_request(frame, tags):
    return rapid_insights.generate_rapid_sentiment_observations(
        frame, scale="3-way", include_not_relevant=False, **BASE,
    )


def rapid_tag_request(frame, tags):
    return rapid_insights.generate_rapid_tag_observations(
        frame, mode=RAPID_TAG_REVIEW_MODE_APPLICABLE, include_other=False, tag_definitions=tags, **BASE,
    )


def top_story_request(frame, tags):
    candidates = top_stories.build_grouped_story_candidates(frame)
    selected = top_stories.save_selected_rows(candidates.assign(**{"Top Story": True}), candidates, pd.DataFrame())
    return top_story_summaries.generate_top_story_observation(
        selected, entity_name=BASE["client_name"], api_key=BASE["api_key"],
    )


PATHS = [
    (author_insights, author_request, "Representative stories:", "primary_examples", "supporting_evidence"),
    (outlet_insights, outlet_request, "Representative stories:", "primary_examples", "supporting_evidence"),
    (regions, region_request, "Region profile context:", "top_stories", "supporting_story_evidence"),
    (ai_sentiment, sentiment_request, "Input data:", "examples_by_sentiment", "aligned_evidence_by_sentiment"),
    (ai_tagging, tagging_request, "Input data:", "examples_by_tag", "aligned_evidence_by_tag"),
    (rapid_insights, rapid_sentiment_request, "Input data:", "examples_by_sentiment", "aligned_evidence_by_sentiment"),
    (rapid_insights, rapid_tag_request, "Input data:", "examples_by_tag", "aligned_evidence_by_tag"),
    (top_story_summaries, top_story_request, "Input data:", "stories", None),
]


def capture(module, generate, frame, tags, *, error=None):
    reply = SimpleNamespace(
        output_text='{"overall_observation":"Mocked","tail_observation":"Mocked","blurb":"Mocked"}',
        usage=None,
    )
    with patch.object(module, "OpenAI") as client:
        create = client.return_value.responses.create
        if error is None:
            create.return_value = reply
            generate(frame, tags)
        else:
            create.side_effect = error
            with pytest.raises(ValueError, match="exceeds the context capacity.*serialized bytes") as raised:
                generate(frame, tags)
            assert raised.value.__cause__ is error
            assert "Primary article text was kept in full." in str(raised.value)
            assert create.call_count == 1
        return create.call_args.kwargs


def payload_from(request, marker):
    prompt = request["input"][-1]["content"]
    return json.JSONDecoder().raw_decode(prompt.split(marker, 1)[1].lstrip())[0]


@pytest.mark.parametrize(("module", "generate", "marker", "primary_key", "supporting_key"), PATHS)
@pytest.mark.parametrize("count", [7, 63])
def test_requests_send_full_primary_articles_and_separate_support(
    module, generate, marker, primary_key, supporting_key, count,
):
    frame, tags = stories(count)
    # Cover all three sources, including the full Agility body in Snippet.
    frame.loc[0, "Full Text"] = "Full Text export\n\n" + frame.loc[0, "Snippet"]
    frame.loc[1, "Article Text"] = "Article Text export\n\n" + frame.loc[1, "Snippet"]
    expected = dict(zip(frame["Headline"], frame["Snippet"]))
    expected[frame.loc[0, "Headline"]] = frame.loc[0, "Full Text"]
    expected[frame.loc[1, "Headline"]] = frame.loc[1, "Article Text"]

    kwargs = capture(module, generate, frame, tags)
    payload = payload_from(kwargs, marker)
    primary = payload[primary_key]
    supporting = payload[supporting_key] if supporting_key else []
    if isinstance(primary, dict):
        primary, = primary.values()
        supporting, = supporting.values()
    assert kwargs["truncation"] == "disabled"
    assert len(primary) == min(20, count)
    if supporting_key:
        assert len(supporting) == min(40, max(count - 20, 0))

    headline_key, text_key = ("Headline", "Snippet") if module is regions else ("headline", "snippet")
    assert {item[headline_key] for item in primary}.isdisjoint(item[headline_key] for item in supporting)
    assert len({item["story_id"] for item in primary}) == len(primary)
    assert all(text_key not in item for item in primary)
    assert {item[headline_key]: payload["articles_by_id"][item["story_id"]] for item in primary} == {
        f"Transit dispatch {n}": expected[f"Transit dispatch {n}"] for n in range(min(20, count))
    }
    if module in (author_insights, outlet_insights, regions):
        cap = 240 if module is regions else 200
        assert all(len(item[text_key]) <= cap for item in supporting)
    else:
        assert all(text_key not in item for item in supporting)

    if module is regions:
        profile_request = kwargs["input"][-1]["content"]
        assert all("Example Snippet" not in item for item in primary)
        for item in primary:
            encoded = json.dumps(payload["articles_by_id"][item["story_id"]], ensure_ascii=False)
            assert profile_request.count(encoded) == 1


def request_size(request):
    return len(json.dumps(request, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def test_many_tags_and_long_articles_measure_request_size_and_surface_provider_limit():
    many, tags = stories(63, tag_count=30)
    long, _ = stories(24, paragraphs=1_000)
    established = capture(ai_tagging, tagging_request, many, tags)
    rapid = capture(rapid_insights, rapid_tag_request, many, tags)
    long_request = capture(ai_sentiment, sentiment_request, long, {})
    sizes = [request_size(req) for req in (established, rapid, long_request)]
    assert max(sizes[:2]) < 1_000_000
    assert sizes[2] > 1_000_000
    for request in (established, rapid):
        payload = payload_from(request, "Input data:")
        primary = payload["examples_by_tag"]
        assert len(payload["articles_by_id"]) == 20
        assert len(primary) == 30
        assert all(len(articles) == 20 for articles in primary.values())
    print(f"Serialized bytes (30-tag established, 30-tag Rapid, long Sentiment): {sizes}")

    error = BadRequestError(
        "The request exceeds the model context window.",
        response=httpx.Response(400, request=httpx.Request("POST", "https://example.test/responses")),
        body={"code": "context_length_exceeded"},
    )
    rejected = capture(rapid_insights, rapid_tag_request, many, tags, error=error)
    assert rejected == rapid


def test_syndicated_group_keeps_pickups_and_distinct_same_headline_group():
    frame, tags = stories(24)
    frame.loc[1, "Headline"] = frame.loc[0, "Headline"]
    pickup = frame.iloc[[0]].assign(Headline="Syndicated transit dispatch", **{"Prime Example": 0})
    pickup["URL"] = "https://example.test/pickup"
    frame = pd.concat([frame, pickup], ignore_index=True)
    for module, generate in ((author_insights, author_request), (outlet_insights, outlet_request)):
        request = capture(module, generate, frame, tags)
        payload = payload_from(request, "Representative stories:")
        primary = payload["primary_examples"]
        assert len(primary) == len({row["story_id"] for row in primary}) == 20
        first, second = [next(row for row in primary if row["story_id"] == key) for key in ("0", "1")]
        assert first["headline"] == second["headline"]
        assert first["syndicated_pickups"] == 1
        assert first["effective_reach"] == int(frame.loc[frame["Group ID"].eq(0), "Effective Reach"].sum())
        assert payload["articles_by_id"]["0"] == frame.loc[0, "Snippet"]


@pytest.mark.parametrize(("module", "generate"), [(ai_tagging, tagging_request), (rapid_insights, rapid_tag_request)])
def test_tag_lists_reuse_body_and_preserve_tag_order_and_metadata(module, generate):
    frame, tags = stories(25, tag_count=2)
    frame.loc[0, "AI Tag Rationale"] = "The same story covers both fictional themes."
    request = capture(module, generate, frame, tags)
    payload = payload_from(request, "Input data:")
    first, second = payload["examples_by_tag"].values()
    assert [row["story_id"] for row in first] == [row["story_id"] for row in second]
    assert len(first) == len(payload["articles_by_id"]) == 20
    assert first[0]["outlet"] == second[0]["outlet"] == "Metro Bulletin"
    assert first[0]["effective_reach"] == second[0]["effective_reach"] == 800
    assert payload["articles_by_id"][first[0]["story_id"]] == frame.loc[0, "Snippet"]
    assert request["input"][-1]["content"].count("End of the original article 0.") == 1
    if module is ai_tagging:
        assert first[0]["tag_rationale"] == frame.loc[0, "AI Tag Rationale"]


def test_regional_profile_links_ranked_pickups_to_shared_group_body():
    frame, tags = stories(25)
    pickup = frame.iloc[[0]].assign(
        Headline="Syndicated transit dispatch", Outlet="Another Bulletin",
        Mentions=3, Impressions=900, **{"Prime Example": 0, "Effective Reach": 650},
    )
    pickup["URL"] = "https://example.test/pickup"
    frame = pd.concat([frame, pickup], ignore_index=True)
    request = capture(regions, region_request, frame, tags)
    profile = payload_from(request, "Region profile context:")
    rows = payload_from(request, "Representative rows:")
    primary = rows["primary_examples"]
    assert len({row["story_id"] for row in primary}) == len(primary) == 20
    picked = next(row for row in primary if row["story_id"] == "0")
    grouped = next(row for row in profile["top_stories"] if row["story_id"] == "0")
    extra = next(row for row in rows["supporting_evidence"] if row["Group ID"] == "0")
    assert {picked["Outlet"], extra["Outlet"]} == {"Another Bulletin", "Metro Bulletin"}
    assert picked["Source Group ID"] == extra["Source Group ID"] == "0"
    assert grouped["Mentions"] == 4 and grouped["Effective Reach"] == 1450
    assert "Snippet" not in grouped and "Snippet" not in picked and "Snippet" not in extra
    assert request["input"][-1]["content"].count("End of the original article 0.") == 1


def test_region_links_consolidated_source_group_ids():
    frame, tags = stories(25)
    pickup = frame.iloc[[0]].assign(Outlet="Another Bulletin", **{"Group ID": 99, "Mentions": 5})
    pickup["URL"] = "https://example.test/pickup"
    request = capture(regions, region_request, pd.concat([frame, pickup], ignore_index=True), tags)
    profile = payload_from(request, "Region profile context:")
    rows = payload_from(request, "Representative rows:")
    story = next(entry for entry in profile["top_stories"] if entry["Headline"] == frame.loc[0, "Headline"])
    assert set(top_stories.parse_source_group_ids(story["Source Group IDs"])) == {"0", "99"}
    matching = [
        row for row in rows["primary_examples"] + rows["supporting_evidence"]
        if row["Group ID"] == story["story_id"]
    ]
    assert {row["Source Group ID"] for row in matching} == {"0", "99"}
    assert {row["Outlet"] for row in matching} == {"Metro Bulletin", "Another Bulletin"}
    assert request["input"][-1]["content"].count("End of the original article 0.") == 1


def test_region_pickup_groups_cannot_expand_full_article_limit():
    frame, tags = stories(41, paragraphs=20)
    # The grouped ranking and row ranking can see different leaders.
    # Exercise the request boundary explicitly with disjoint evidence lists.
    profile = {
        "region": "Canada",
        "top_stories": [
            {"Group ID": str(n), "Headline": row["Headline"], "Snippet": row["Snippet"]}
            for n, row in frame.iloc[:20].iterrows()
        ],
    }
    pickup_rows = {
        "primary_examples": [
            {"Group ID": str(n), "Headline": frame.loc[n, "Headline"],
             "Snippet": frame.loc[n, "Snippet"], "Outlet": frame.loc[n, "Outlet"]}
            for n in [0, *range(20, 39)]
        ],
        "supporting_evidence": [],
    }

    def generate(_frame, _tags):
        regions.generate_region_profile_observation(
            level="Country", metric_label="Mentions", analysis_context="",
            region_profile=profile, level_payload={}, representative_rows=pickup_rows, **BASE,
        )

    request = capture(regions, generate, frame, tags)
    sent_profile = payload_from(request, "Region profile context:")
    sent_rows = payload_from(request, "Representative rows:")["primary_examples"]
    assert len(sent_profile["articles_by_id"]) == 20
    assert sent_rows[0]["article_ref"] == "0"
    assert len(sent_rows) == 20
    assert all("article_ref" not in row and "Snippet" not in row for row in sent_rows[1:])
    assert sent_rows[-1]["story_id"] == "38"
    assert "End of the original article 38." not in request["input"][-1]["content"]
