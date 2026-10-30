from __future__ import annotations

import io
import json
from types import SimpleNamespace

import httpx
import pandas as pd
import pytest
from openai import OpenAI

from processing import (
    ai_sentiment,
    ai_tagging,
    analysis_context,
    author_insights,
    outlet_insights,
    rapid_insights,
    rapid_second_opinion,
    regions,
    spot_checks,
    top_story_summaries,
)
from processing.ai_sentiment import init_ai_sentiment_state
from processing.sentiment_config import init_sentiment_config_state
from utils import api_meter
from utils.session_snapshot import build_session_snapshot_bytes, load_session_state_from_file


class State(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


@pytest.fixture
def meter_state(monkeypatch):
    state = State()
    monkeypatch.setattr(api_meter, "st", SimpleNamespace(session_state=state))
    return state


@pytest.fixture
def sdk_stub():
    """Exercise real SDK serialization and parsing without network or inference."""
    clients = []

    def make_client(payload, endpoint):
        requests = []

        def handle(request):
            assert request.url.path == f"/v1/{endpoint}"
            body = json.loads(request.content)
            requests.append(body)
            response = {"id": "test-response", "model": body["model"], "created": 0}
            if endpoint == "chat/completions":
                response.update(
                    object="chat.completion",
                    choices=[{
                        "index": 0,
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [{
                                "id": "test-call",
                                "type": "function",
                                "function": {
                                    "name": body["tool_choice"]["function"]["name"],
                                    "arguments": json.dumps(payload),
                                },
                            }],
                        },
                    }],
                    usage={"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18},
                )
            else:
                response.update(
                    object="response",
                    created_at=0,
                    status="completed",
                    output=[{
                        "id": "test-message",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{
                            "type": "output_text",
                            "text": payload if isinstance(payload, str) else json.dumps(payload),
                            "annotations": [],
                        }],
                    }],
                    usage={
                        "input_tokens": 10,
                        "output_tokens": 8,
                        "total_tokens": 18,
                        "input_tokens_details": {"cached_tokens": 0},
                        "output_tokens_details": {"reasoning_tokens": 3},
                    },
                )
            return httpx.Response(200, json=response)

        client = OpenAI(
            api_key="test-only",
            base_url="https://openai.test/v1",
            http_client=httpx.Client(transport=httpx.MockTransport(handle)),
        )
        clients.append(client)
        return client, requests

    yield make_client
    for client in clients:
        client.close()


def test_fresh_sentiment_initializers_agree_on_new_default():
    for initialize in (init_ai_sentiment_state, init_sentiment_config_state):
        state = State()
        initialize(state)
        assert state.model_choice == "gpt-6-luna"


@pytest.mark.parametrize("model", ["gpt-5.6-luna", "gpt-6-luna"])
def test_snapshot_load_and_initialization_preserve_model_results_and_overrides(model):
    completed = pd.DataFrame({
        "Group ID": [1],
        "AI Sentiment": ["NEUTRAL"],
        "Assigned Sentiment": ["POSITIVE"],
        "Assigned Sentiment Source": ["HUMAN"],
    })
    narrative = {"overall_observation": "Previously completed narrative."}
    state = State(
        model_choice=model,
        tagging_model=model,
        df_sentiment_unique=completed,
        top_story_observation_output=narrative,
    )
    snapshot, skipped = build_session_snapshot_bytes(state)
    assert not skipped
    restored = State()
    load_session_state_from_file(restored, io.BytesIO(snapshot))
    init_ai_sentiment_state(restored)
    init_sentiment_config_state(restored)
    assert restored.model_choice == model
    assert restored.tagging_model == model
    assert restored.top_story_observation_output == narrative
    pd.testing.assert_frame_equal(restored.df_sentiment_unique, completed)


def test_new_pricing_preserves_historical_costs_and_meter_totals(meter_state):
    assert api_meter.get_model_prices("gpt-6-luna") == {"in": 0.10, "out": 0.50}
    api_meter.apply_usage_to_session(1_000_000, 1_000_000, "gpt-5.6-luna")
    historical = meter_state.api_meter["by_model"]["gpt-5.6-luna"].copy()
    api_meter.apply_usage_to_session(1_000_000, 1_000_000, "gpt-6-luna")
    assert historical["cost_usd"] == pytest.approx(1.40)
    assert meter_state.api_meter["by_model"]["gpt-5.6-luna"] == historical
    assert meter_state.api_meter["by_model"]["gpt-6-luna"]["cost_usd"] == pytest.approx(0.60)
    assert api_meter.get_api_cost_usd() == pytest.approx(2.00)
    api_meter.apply_usage_to_session(1_000_000, 1_000_000, "gpt-6.1-terra")
    assert api_meter.get_model_prices("gpt-6.1-terra") == {"in": 1.00, "out": 5.00}
    assert meter_state.api_meter["by_model"]["gpt-5.6-luna"] == historical
    assert meter_state.api_meter["by_model"]["gpt-6-luna"]["cost_usd"] == pytest.approx(0.60)
    assert meter_state.api_meter["by_model"]["gpt-6.1-terra"]["cost_usd"] == pytest.approx(6.00)
    assert api_meter.get_api_cost_usd() == pytest.approx(8.00)


@pytest.mark.parametrize("workflow", ["sentiment", "rapid"])
def test_review_chat_requests_keep_function_calling_and_none_effort(workflow, sdk_stub, monkeypatch):
    payload = {"named_entity": "Client", "sentiment": "NEUTRAL", "confidence": 88, "explanation": "Factual mention."}
    if workflow == "rapid":
        payload = {"sentiment_outcome": "0", "sentiment_confidence": 88, "sentiment_rationale": "Factual mention."}
    client, requests = sdk_stub(payload, "chat/completions")
    module = spot_checks if workflow == "sentiment" else rapid_second_opinion
    monkeypatch.setattr(module, "OpenAI", lambda **kwargs: client)
    if workflow == "sentiment":
        result, in_tokens, out_tokens, note = spot_checks.call_ai_sentiment(
            "Story prompt", spot_checks.DEFAULT_SECOND_OPINION_MODEL,
            [{"name": "analyze_sentiment", "parameters": {"type": "object", "properties": {}}}],
            "3-way", "test-only",
        )
        assert note == ""
    else:
        result, in_tokens, out_tokens, _ = rapid_second_opinion.call_rapid_review_luna("Story prompt", "test-only")
    assert result == payload
    assert (in_tokens, out_tokens) == (10, 8)
    assert len(requests) == 1
    assert requests[0]["model"] == "gpt-6-luna"
    assert requests[0]["reasoning_effort"] == "none"
    assert requests[0]["messages"][1]["content"] == "Story prompt"


@pytest.mark.parametrize("workflow,effort", [("top_story", "low"), ("analysis_context", "medium")])
@pytest.mark.parametrize("model_override", [None, "gpt-6-luna"])
def test_responses_requests_keep_structured_output_and_reasoning(
    workflow, effort, model_override, sdk_stub, meter_state, monkeypatch,
):
    if workflow == "top_story":
        payload = {
            "chart_callout": "Client announces dates.",
            "top_story_summary": "Client announces program dates.",
            "entity_sentiment_label": "Neutral",
            "entity_sentiment_rationale": "Factual reporting of program dates.",
        }
    else:
        payload = {"assessment": "No additions needed.", "aliases": [], "spokespeople": [], "products": []}
    client, requests = sdk_stub(payload, "responses")
    model_kwargs = {} if model_override is None else {"model": model_override}
    if workflow == "top_story":
        result, in_tokens, out_tokens = top_story_summaries.generate_structured_story_outputs(
            client, "Story prompt", **model_kwargs,
        )
        assert result["entity_sentiment"] == "Neutral: Factual reporting of program dates."
    else:
        monkeypatch.setattr(analysis_context, "OpenAI", lambda **kwargs: client)
        result, in_tokens, out_tokens = analysis_context.generate_analysis_context_suggestions(
            "Client", "Client", [], [], [], "", "test-only", **model_kwargs,
        )
        assert requests[0]["tools"] == [{"type": "web_search"}]
    assert all(result[key] == value for key, value in payload.items())
    assert (in_tokens, out_tokens) == (10, 8)
    assert len(requests) == 1
    assert requests[0]["model"] == (model_override or "gpt-6.1-terra")
    assert requests[0]["reasoning"] == {"effort": effort}
    assert requests[0]["text"]["format"]["type"] == "json_schema"
    assert requests[0]["text"]["format"]["strict"] is True
    assert meter_state.api_meter["out_tokens"] == 8
    expected_cost = 0.000005 if model_override else 0.000050
    assert api_meter.get_api_cost_usd() == pytest.approx(expected_cost)


@pytest.fixture
def narrative_frame():
    return pd.DataFrame([{
        "Group ID": 1,
        "Headline": "Client announces program dates",
        "Snippet": "Client announces dates for its Programs initiative.",
        "Outlet": "Example News",
        "Author": "Example Author",
        "Type": "ONLINE",
        "URL": "https://example.com/story",
        "Date": "2026-10-01",
        "Prime Example": 1,
        "Group Count": 1,
        "Mentions": 1,
        "Impressions": 1000,
        "Effective Reach": 500,
        "Assigned Sentiment": "NEUTRAL",
        "AI Sentiment": "NEUTRAL",
        "AI Sentiment Confidence": 88,
        "Review AI Confidence": None,
        "Assigned Tag": "Programs",
        "Jev Sentiment": "NEUTRAL",
        "Jev 5-Way Sentiment": "NEUTRAL",
        "Jev Sentiment Score": 0,
        "Jev Mixture Label": "VERY CONSISTENT",
        "Jev Best Tag": "Programs",
        "Jev Tags": "Programs",
    }])


@pytest.mark.parametrize("workflow", [
    "sentiment", "tagging", "rapid_sentiment", "rapid_tagging", "top_observation",
    "region_overview", "region_profile", "author", "outlet",
])
def test_narrative_workflows_use_terra_and_preserve_response_contracts(
    workflow, narrative_frame, sdk_stub, meter_state, monkeypatch,
):
    observation = "Coverage reports program dates."
    payload = {"overall_observation": observation}
    if workflow in {"sentiment", "rapid_sentiment"}:
        payload["sentiment_sections"] = [{"sentiment": "NEUTRAL", "observation": observation}]
    elif workflow in {"tagging", "rapid_tagging"}:
        payload["tag_sections"] = [{"tag": "Programs", "observation": observation}]
    elif workflow == "region_overview":
        payload["tail_observation"] = ""
    elif workflow == "region_profile":
        payload = {"blurb": observation}
    elif workflow in {"author", "outlet"}:
        payload = observation

    client, requests = sdk_stub(payload, "responses")
    module = {
        "sentiment": ai_sentiment, "tagging": ai_tagging,
        "rapid_sentiment": rapid_insights, "rapid_tagging": rapid_insights,
        "top_observation": top_story_summaries,
        "region_overview": regions, "region_profile": regions,
        "author": author_insights, "outlet": outlet_insights,
    }[workflow]
    monkeypatch.setattr(module, "OpenAI", lambda **kwargs: client)
    frame = narrative_frame
    original = frame.copy(deep=True)
    regional_args = {
        "client_name": "Client", "analysis_context": "", "level": "Country",
        "metric_label": "Mentions", "api_key": "test-only",
    }
    if workflow == "sentiment":
        result = ai_sentiment.generate_sentiment_observations(frame, frame, "Client", "3-way", False, "test-only")
    elif workflow == "tagging":
        result = ai_tagging.generate_tag_observations(frame, "Client", True, "test-only")
    elif workflow == "rapid_sentiment":
        result = rapid_insights.generate_rapid_sentiment_observations(
            frame, client_name="Client", scale="3-way", include_not_relevant=False, api_key="test-only",
        )
    elif workflow == "rapid_tagging":
        result = rapid_insights.generate_rapid_tag_observations(
            frame, client_name="Client", mode=rapid_insights.RAPID_TAG_REVIEW_MODE_BEST,
            include_other=True, tag_definitions={"Programs": "Program coverage"}, api_key="test-only",
        )
    elif workflow == "top_observation":
        result = top_story_summaries.generate_top_story_observation(frame, "Client", "test-only")
    elif workflow == "region_overview":
        ranking = pd.DataFrame([{
            "Region": "Canada", "Mentions": 1, "Impressions": 1000,
            "Effective Reach": 500, "Outlet Count": 1, "Example Outlets": "Example News",
        }])
        result = regions.generate_region_level_overview(ranking_df=ranking, filtered_df=frame, **regional_args)
    elif workflow == "region_profile":
        result = regions.generate_region_profile_observation(
            region_profile={"region": "Canada"}, level_payload={},
            representative_rows={"primary_examples": [], "supporting_evidence": []}, **regional_args,
        )
    elif workflow == "author":
        metrics, stories = author_insights.build_author_metrics(frame)
        headlines = author_insights.build_author_headline_table(stories, "Example Author")
        result = author_insights.generate_author_summary("Example Author", "Client", metrics.iloc[0], headlines, "test-only")
    else:
        result = outlet_insights.generate_outlet_summary("Example News", "Client", frame.iloc[0], frame, frame, "test-only")

    output, in_tokens, out_tokens = result[:3]
    if workflow == "region_profile":
        assert output == payload["blurb"]
    elif isinstance(payload, str):
        assert output == payload
    else:
        assert all(output[key] == value for key, value in payload.items())
    if workflow.startswith("rapid_"):
        assert len(result[3]) == 64  # Preserve the observation fingerprint contract.
    assert (in_tokens, out_tokens) == (10, 8)
    assert len(requests) == 1
    assert requests[0]["model"] == "gpt-6.1-terra"
    assert requests[0]["reasoning"] == {"effort": "low"}
    assert requests[0]["text"]["verbosity"] == "low"
    if workflow not in {"author", "outlet"}:
        assert requests[0]["text"]["format"]["type"] == "json_schema"
        assert requests[0]["text"]["format"]["strict"] is True
    assert meter_state.api_meter["out_tokens"] == 8
    assert api_meter.get_api_cost_usd() == pytest.approx(0.000050)
    pd.testing.assert_frame_equal(frame, original)
