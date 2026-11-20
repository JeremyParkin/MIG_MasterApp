import io
import json
import zipfile
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest
from deep_translator.exceptions import TooManyRequests
from streamlit.testing.v1 import AppTest

from processing.analysis_context import build_coverage_row_key_series, build_sentiment_analysis_context_text
from processing.download_exports import (
    build_clean_workbook_bytes,
    build_report_copy_docx_bytes,
    build_scoped_traditional_export_bundle,
)
from processing.jev_sentiment import build_jev_combined_sentiment_payload, get_remaining_jev_sentiment_rows
from processing.rapid_insights import rapid_observation_current_fingerprint
from processing.rapid_resolution import (
    apply_rapid_sentiment_human_selection,
    apply_rapid_tag_human_selection,
    resolve_final_rapid_sentiment_row,
    resolve_final_rapid_tag_row,
    resolve_rapid_tag_row,
)
from processing.rapid_sampling import scoped_rapid_source_rows
from processing.rapid_second_opinion import build_rapid_review_candidates, build_rapid_review_prompt, run_rapid_review_batch
from processing.spot_checks import escape_markdown
from ui.app_shell import build_pages
from utils.session_snapshot import build_session_snapshot_bytes, load_session_state_from_file


class SessionState(dict):
    __setattr__ = dict.__setitem__


def test_deployment_switch_blocks_rapid_navigation_and_exports_after_restore(monkeypatch):
    source = pd.DataFrame({"Group ID": [1], "Headline": ["Story"]})
    state = {
        "df_traditional": source,
        "df_jev_sentiment_grouped_rows": source,
        "df_jev_sentiment_unique": source.assign(
            **{"Jev Sentiment": "POSITIVE", "Jev 5-Way Sentiment": "SOMEWHAT POSITIVE"}
        ),
        "rapid_sentiment_observation_output": {"overall_observation": "Rapid-only narrative"},
        "sentiment_observation_output": {"overall_observation": "Established narrative"},
    }
    snapshot, _ = build_session_snapshot_bytes(state)
    restored = SessionState()
    load_session_state_from_file(restored, io.BytesIO(snapshot))
    monkeypatch.setenv("MIG_ENABLE_RAPID_LABELING", "0")

    with patch("ui.app_shell.st.Page", side_effect=lambda path, **kwargs: kwargs["title"]):
        pages = build_pages()
        assert "Sentiment" in pages and "Tagging" in pages
        assert "Rapid Labeling" not in pages
        monkeypatch.delenv("MIG_ENABLE_RAPID_LABELING")
        assert "Rapid Labeling" in build_pages()

    workbook = pd.ExcelFile(io.BytesIO(build_clean_workbook_bytes(restored)))
    assert "RAPID LABELING SAMPLE" in workbook.sheet_names
    monkeypatch.setenv("MIG_ENABLE_RAPID_LABELING", "0")
    workbook = pd.ExcelFile(io.BytesIO(build_clean_workbook_bytes(restored)))
    assert "RAPID LABELING SAMPLE" not in workbook.sheet_names
    assert "Final Rapid Sentiment 3-Way" not in workbook.parse("CLEAN TRAD").columns
    assert not workbook.parse("EXPORT METADATA")["Field"].dropna().str.startswith("Rapid Labeling").any()
    assert "Rapid-only narrative" not in _report_xml(restored)
    assert "df_jev_sentiment_unique" in restored


def _report_xml(state):
    with zipfile.ZipFile(io.BytesIO(build_report_copy_docx_bytes(state))) as document:
        return document.read("word/document.xml").decode()


def test_export_requires_current_rapid_observation_fingerprint():
    unique = pd.DataFrame({
        "Group ID": [1],
        "Headline": ["Story"],
        "Jev Sentiment": ["POSITIVE"],
        "Jev 5-Way Sentiment": ["SOMEWHAT POSITIVE"],
    })
    state = {
        "df_traditional": unique[["Group ID", "Headline"]],
        "df_jev_sentiment_unique": unique,
        "rapid_sentiment_observation_output": {"overall_observation": "Current Rapid narrative"},
        "sentiment_observation_output": {"overall_observation": "Established narrative"},
    }
    state["rapid_sentiment_observation_fingerprint"] = rapid_observation_current_fingerprint(
        family="sentiment",
        unique_df=unique,
        analysis_context=build_sentiment_analysis_context_text(state),
    )
    assert "Current Rapid narrative" in _report_xml(state)
    unique.loc[0, "Jev Sentiment"] = "NEUTRAL"
    assert "Current Rapid narrative" not in _report_xml(state)


def test_saved_single_best_mode_keeps_independent_applicable_evidence():
    row = {
        "Jev Best Tag": "Access",
        "Jev Tags": "Access",
        "Jev Tag [Access]": "YES",
        "Jev Tag [Affordability]": "YES",
        "Rapid Review Status": "Completed",
        "Rapid Review Best Tag": "Access",
        "Rapid Review Tags": "Access",
        "Rapid Review Raw Response": '{"applicable_tags": ["Access", "Affordability"]}',
    }
    result = resolve_rapid_tag_row(row, tag_definitions={"Access": "Availability", "Affordability": "Costs"})
    assert result["Effective Rapid Best Tag"] == "Access"
    assert result["Effective Rapid Tags"] == "Access; Affordability"
    assert result["Rapid Tag Resolution Status"] == "Cross-model agreement"


def test_failed_first_pass_rows_are_eligible_for_explicit_retry():
    rows = pd.DataFrame({"Group ID": [1, 2], "Jev Error": ["Timed out", pd.NA]})
    assert get_remaining_jev_sentiment_rows(rows)["Group ID"].tolist() == [2]
    assert get_remaining_jev_sentiment_rows(rows, retry_errors=True)["Group ID"].tolist() == [1]


def test_changed_context_blocks_rapid_until_explicit_reset():
    app = AppTest.from_file("pages/Jev_Sentiment_Experimental.py")
    app.session_state["standard_step"] = True
    app.session_state["analysis_context_saved_at"] = "saved"
    app.session_state["analysis_primary_names"] = ["New entity"]
    app.session_state["df_traditional"] = pd.DataFrame({"Group ID": [1], "Headline": ["Story"]})
    app.session_state["jev_sentiment_config_step"] = True
    app.session_state["jev_sentiment_section"] = "Run Jev Analysis"
    app.session_state["jev_analysis_context_fingerprint"] = "previous context"
    app.session_state["df_jev_sentiment_unique"] = pd.DataFrame({
        "Group ID": [1],
        "Jev Sentiment": ["POSITIVE"],
        "Rapid Review Status": ["Completed"],
        "Rapid Human Sentiment": ["NEUTRAL"],
    })
    app.session_state["rapid_sentiment_observation_output"] = {"overall_observation": "Old narrative"}
    app.run()
    assert not app.exception
    assert not any(button.label == "Run Next Rapid Batch" for button in app.button)
    reset = next(button for button in app.button if button.label == "Reset Rapid Results for current context")
    reset.click().run()
    assert not app.exception
    row = app.session_state["df_jev_sentiment_unique"].iloc[0]
    assert row["Group ID"] == 1
    assert row[["Jev Sentiment", "Rapid Review Status", "Rapid Human Sentiment"]].isna().all()
    assert not app.session_state["rapid_sentiment_observation_output"]


def test_all_applicable_tag_review_saves_each_story_and_reopens_its_exact_tags(monkeypatch):
    monkeypatch.setenv("MIG_ENABLE_RAPID_LABELING", "1")
    stories = pd.DataFrame({
        "Group ID": [10, 20],
        "Headline": ["Financing program", "Policy announcement"],
        "Group Count": [2, 1],
        "Mentions": [2, 1],
        "Impressions": [100, 50],
        "Effective Reach": [80, 40],
    })
    app = AppTest.from_file("pages/Jev_Sentiment_Experimental.py")
    state = {
        "standard_step": True,
        "analysis_context_saved_at": "saved",
        "analysis_primary_names": ["Example"],
        "df_traditional": stories,
        "df_jev_sentiment_grouped_rows": stories,
        "df_jev_sentiment_unique": stories.assign(**{
            "Jev Sentiment": "NEUTRAL",
            "Jev 5-Way Sentiment": "NEUTRAL",
            "Jev Best Tag": ["Financing", "Policy"],
            "Jev Tags": ["Financing; Inclusion", "Inclusion; Policy"],
        }),
        "jev_sentiment_config_step": True,
        "jev_sentiment_section": "Spot Checks",
        "jev_tag_definitions": {
            "Financing": "Lending programs",
            "Inclusion": "Access for underserved groups",
            "Policy": "Public policy",
        },
        "jev_tagging_mode": "Multiple applicable tags",
        "rapid_spot_review_domain": "Tagging",
        "rapid_spot_review_view": "All Rapid sample coverage",
        "client_name": "Example",
        "export_name": "Rapid review regression",
    }
    for key, value in state.items():
        app.session_state[key] = value

    def selected_tags(group_id):
        return {
            checkbox.label for checkbox in app.checkbox
            if checkbox.value and checkbox.key.startswith(f"rapid_tag_multi_{group_id}_")
        }

    app.run()
    assert not app.exception
    assert selected_tags(10) == {"Financing", "Inclusion"}
    app.checkbox(key="rapid_tag_multi_10_Inclusion").uncheck().run()
    app.checkbox(key="rapid_tag_multi_10_Policy").check().run()
    app.button(key="rapid_assign_tags_10").click().run()
    assert not app.exception
    assert selected_tags(20) == {"Inclusion", "Policy"}
    app.checkbox(key="rapid_tag_multi_20_Policy").uncheck().run()
    app.checkbox(key="rapid_tag_multi_20_Financing").check().run()
    app.button(key="rapid_assign_tags_20").click().run()
    assert not app.exception

    app.button(key="rapid_prev_Tagging").click().run()
    assert selected_tags(10) == {"Financing", "Policy"}
    app.button(key="rapid_next_Tagging").click().run()
    assert selected_tags(20) == {"Financing", "Inclusion"}

    download = AppTest.from_file("pages/11-Download.py")
    for key in state:
        download.session_state[key] = app.session_state[key]
    download.run()
    download.button(key="build_clean_workbook").click().run()
    assert not download.exception
    with pd.ExcelFile(io.BytesIO(download.session_state["clean_excel_bytes"])) as workbook:
        for sheet in ("CLEAN TRAD", "RAPID LABELING SAMPLE"):
            tags = workbook.parse(sheet).set_index("Group ID")["Final Rapid Tags"]
            assert tags.to_dict() == {10: "Financing; Policy", 20: "Financing; Inclusion"}


QUEUE_TAGS = {"Programs": "Institutional programs", "Policy": "Public policy", "Access": "Access to services"}
RECOMMENDED = "Recommended / flagged for review"


def run_scripted_second_opinion(monkeypatch, *, jev=None, luna=None, omit=None, fail=False):
    monkeypatch.setattr("processing.rapid_second_opinion.time.sleep", lambda _: None)
    row = {
        "Group ID": 10,
        "Headline": "Example institution updates its programs",
        "Snippet": "The Example institution announced new programs and a policy update.",
        "Group Count": 1,
        "Mentions": 1,
        "Impressions": 100,
        "Effective Reach": 80,
        "Jev Sentiment": "POSITIVE",
        "Jev 5-Way Sentiment": "SOMEWHAT POSITIVE",
        "Jev Sentiment Score": 3.0,
        "Jev Score Relevant Probability": 0.95,
        "Jev Mixture Label": "VERY CONSISTENT",
        "Jev Best Tag": "Programs",
        "Jev Tags": "Programs; Policy",
        "Jev Tag [Programs]": "YES",
        "Jev Tag [Policy]": "YES",
        "Jev Tag [Access]": "NO",
        **(jev or {}),
    }
    reply = {
        "sentiment_outcome": "5",
        "sentiment_confidence": 0.91,
        "sentiment_rationale": "The article is favorable toward the institution.",
        "best_tag": "Policy",
        "applicable_tags": ["Policy", "Programs"],
        "tag_confidence": 0.9,
        "tag_rationale": "The story covers both the programs and the policy.",
        **(luna or {}),
    }
    if omit:
        reply.pop(omit)

    def scripted_luna(*args, **kwargs):
        if fail:
            raise RuntimeError("Scripted second-opinion failure")
        return reply, 17, 9, json.dumps(reply)

    frame = pd.DataFrame([row])
    candidates = build_rapid_review_candidates(frame, tag_definitions=QUEUE_TAGS)
    assert candidates["Group ID"].tolist() == [10]
    updated, summary = run_rapid_review_batch(
        frame, candidates, {}, "scripted", tag_definitions=QUEUE_TAGS, call_fn=scripted_luna,
    )
    assert summary["done"] == 1
    return updated, summary


def open_queue(monkeypatch, frame, domain, *, scale="3-way", mode="All applicable", overrides=None, timeout=3):
    monkeypatch.setenv("MIG_ENABLE_RAPID_LABELING", "1")
    app = AppTest.from_file("pages/Jev_Sentiment_Experimental.py", default_timeout=timeout)
    state = {
        "standard_step": True,
        "analysis_context_saved_at": "saved",
        "analysis_primary_names": ["Example"],
        "df_traditional": frame,
        "df_jev_sentiment_grouped_rows": frame,
        "df_jev_sentiment_unique": frame,
        "jev_sentiment_config_step": True,
        "jev_sentiment_section": "Spot Checks",
        "jev_tag_definitions": QUEUE_TAGS,
        "jev_tagging_mode": "Multiple applicable tags",
        "rapid_spot_review_domain": domain,
        "rapid_spot_review_view": RECOMMENDED,
        "rapid_human_sentiment_scale": scale,
        "rapid_human_tag_review_mode": mode,
        **(overrides or {}),
    }
    for key, value in state.items():
        app.session_state[key] = value
    return app.run()


def queue_size(app):
    assert not app.exception
    return int(next(metric.value for metric in app.metric if metric.label == "In review queue"))


@pytest.mark.parametrize(
    ("domain", "legacy_setting", "labeled_view", "jev"),
    [
        ("Sentiment", {"scale": "5-way"}, "All Rapid-labeled sentiment", {
            "Jev Sentiment Score": 4.0,
            "Rapid Review Priority Tier": "Tier 1 - Contradictions / malformed outputs",
            "Rapid Review Priority Reason": "raw 5-way conflicts with score-derived 5-way",
        }),
        ("Tagging", {"mode": "Single best fit"}, "All Rapid-tagged coverage", {
            "Jev Best Tag Selected Probability": 0.4,
            "Rapid Review Priority Tier": "Tier 3 - Uncertainty",
            "Rapid Review Priority Reason": "low best-tag probability",
        }),
    ],
)
def test_rapid_queue_clears_inactive_flags_only_in_scoped_views(monkeypatch, domain, legacy_setting, labeled_view, jev):
    if domain == "Sentiment":
        # Jev's score and Luna's categorical intensity differ from Jev's five-way label.
        luna = {"sentiment_outcome": "7", "best_tag": "Programs"}
    else:
        luna = {"sentiment_outcome": "3"}
    frame, summary = run_scripted_second_opinion(monkeypatch, jev=jev, luna=luna)
    assert summary["successful"] == 1
    app = open_queue(monkeypatch, frame, domain)
    assert queue_size(app) == 0
    for view in ("Cross-model disagreements", "Internal conflicts"):
        app.selectbox(key="rapid_spot_review_view").select(view).run()
        assert queue_size(app) == 0
    for view in (labeled_view, "All Rapid sample coverage"):
        app.selectbox(key="rapid_spot_review_view").select(view).run()
        assert queue_size(app) == 1
        assert app.button(key="rapid_spot_translate")
    assert queue_size(open_queue(monkeypatch, frame, domain, **legacy_setting)) == 1
    pd.testing.assert_frame_equal(app.session_state["df_jev_sentiment_unique"].loc[:, frame.columns], frame)


@pytest.mark.parametrize(
    ("domain", "jev", "luna", "omit", "fail"),
    [
        ("Sentiment", {}, {"sentiment_outcome": "NOT_RELEVANT"}, None, False),
        ("Sentiment", {"Jev Score Relevant Probability": 0.1}, {}, None, False),
        ("Sentiment", {"Jev Score Relevant Probability": 0.5, "Jev Sentiment Score": 4.0}, {}, None, False),
        ("Sentiment", {"Jev Sentiment": "NEUTRAL", "Jev 5-Way Sentiment": "NEUTRAL",
                       "Jev Sentiment Score": 0.0, "Jev Score Relevant Probability": 0.5}, {}, None, True),
        ("Sentiment", {"Jev Score Relevant Probability": 0.5}, {}, "sentiment_outcome", False),
        ("Tagging", {"Jev Best Tag Selected Probability": 0.4},
         {"best_tag": "Programs", "applicable_tags": ["Programs"]}, None, False),
        ("Tagging", {"Jev Best Tag": "Access"}, {}, None, False),
        ("Tagging", {"Jev Best Tag": "Other", "Jev Tags": "Other", "Jev Tag [Programs]": "NO",
                     "Jev Tag [Policy]": "NO", "Jev Best Tag Selected Probability": 0.4},
         {"best_tag": "Other"}, "applicable_tags", False),
    ],
)
def test_rapid_queue_keeps_active_or_unconfirmed_flags(monkeypatch, domain, jev, luna, omit, fail):
    frame, summary = run_scripted_second_opinion(monkeypatch, jev=jev, luna=luna, omit=omit, fail=fail)
    if fail or omit == "sentiment_outcome":
        assert summary["successful"] == 0
        assert frame.loc[0, "Rapid Review Status"] == "Error"
    elif omit == "applicable_tags":
        assert "applicable_tags" not in json.loads(frame.loc[0, "Rapid Review Raw Response"])
        assert frame.loc[0, "Rapid Review Applicable Tags"] == "Other"
    assert queue_size(open_queue(monkeypatch, frame, domain)) == 1


def test_rapid_queue_accepts_numeric_zero_from_both_passes(monkeypatch):
    frame, summary = run_scripted_second_opinion(
        monkeypatch,
        jev={"Jev Sentiment": "NEUTRAL", "Jev 5-Way Sentiment": "NEUTRAL", "Jev Sentiment Score": 0.0},
        luna={"sentiment_outcome": 0},
    )
    assert summary["successful"] == 1
    assert frame.loc[0, "Rapid Review Sentiment Score"] == 0.0
    assert queue_size(open_queue(monkeypatch, frame, "Sentiment")) == 0


def test_rapid_queue_borderline_example_and_saved_analyst_decisions(monkeypatch):
    frame, summary = run_scripted_second_opinion(
        monkeypatch,
        jev={
            "Snippet": "The Example institution published an update. Families criticized the institution's handling of access.",
            "Jev Sentiment": "NEUTRAL",
            "Jev 5-Way Sentiment": "NEUTRAL",
            "Jev Sentiment Score": 0.0,
            "Jev Probability NEGATIVE": 0.43,
            "Jev Probability NEUTRAL": 0.47,
        },
        luna={
            "sentiment_outcome": "-3",
            "sentiment_rationale": "Families explicitly criticized the institution's handling of access.",
            "applicable_tags": ["Policy"],
        },
    )
    assert summary["successful"] == 1
    assert frame.loc[0, "Jev Sentiment Score"] == 0.0
    assert frame.loc[0, "Rapid Review 3-Way Sentiment"] == "NEGATIVE"
    assert resolve_final_rapid_sentiment_row(frame.iloc[0])["Final Rapid Sentiment 3-Way"] == "NEUTRAL"
    assert queue_size(open_queue(monkeypatch, frame, "Sentiment")) == 1

    frame = apply_rapid_sentiment_human_selection(frame, 10, label="POSITIVE", scale="3-way")
    frame = apply_rapid_tag_human_selection(frame, 10, assignment=["Access"], tagging_mode="All applicable",
                                           tag_definitions=QUEUE_TAGS)
    assert resolve_final_rapid_sentiment_row(frame.iloc[0])["Final Rapid Sentiment 3-Way"] == "POSITIVE"
    assert resolve_final_rapid_tag_row(frame.iloc[0], tag_definitions=QUEUE_TAGS)["Final Rapid Tags"] == "Access"
    for domain in ("Sentiment", "Tagging"):
        app = open_queue(monkeypatch, frame, domain)
        assert queue_size(app) == 0
        app.selectbox(key="rapid_spot_review_view").select("All Rapid sample coverage").run()
        assert queue_size(app) == 1
        pd.testing.assert_frame_equal(app.session_state["df_jev_sentiment_unique"].loc[:, frame.columns], frame)


@pytest.mark.parametrize("prior_is_valid", [False, True])
def test_rapid_translation_reopens_saved_story_and_can_retry(monkeypatch, prior_is_valid):
    original_headline = "Une annonce à la radio"
    original_body = "La station diffuse une entrevue."
    saved_error = f"Error: {TooManyRequests()}"
    previous = ["Earlier headline", "Earlier body"] if prior_is_valid else [saved_error, saved_error]
    columns = ["Translated Headline", "Translated Body"]
    frame = pd.DataFrame([{
        "Group ID": 2969,
        "Headline": original_headline,
        "Snippet": original_body,
        "Group Count": 1,
        "Mentions": 1,
        "Impressions": 100,
        "Effective Reach": 80,
        "Jev Sentiment": "NEUTRAL",
        "Jev 5-Way Sentiment": "NEUTRAL",
        **dict(zip(columns, previous, strict=True)),
    }])
    snapshot, skipped = build_session_snapshot_bytes({"df_jev_sentiment_unique": frame})
    assert not skipped
    restored = SessionState()
    load_session_state_from_file(restored, io.BytesIO(snapshot))
    app = open_queue(monkeypatch, restored["df_jev_sentiment_unique"], "Sentiment")
    app.selectbox(key="rapid_spot_review_view").select("All Rapid sample coverage").run()

    def rendered_text():
        assert not app.exception
        return " ".join(element.value for element in app.markdown)

    opened = rendered_text()
    assert saved_error not in opened
    assert (previous[0] if prior_is_valid else original_headline) in opened
    assert escape_markdown(previous[1] if prior_is_valid else original_body) in opened

    provider = "processing.spot_checks.GoogleTranslator"
    with patch(provider) as translator:
        translator.return_value.translate.side_effect = ["New headline", TooManyRequests()]
        app.button(key="rapid_spot_translate").click().run()
        assert translator.return_value.translate.call_count == 2

    failed = rendered_text()
    assert original_headline in failed and escape_markdown(original_body) in failed
    assert [error.value for error in app.error] == ["Translation failed. Select Translate to retry."]
    for key in ("df_jev_sentiment_unique", "df_jev_sentiment_grouped_rows"):
        assert app.session_state[key].iloc[0][columns].tolist() == previous

    with patch(provider) as translator:
        translator.return_value.translate.side_effect = ["A radio announcement", "The station broadcasts an interview."]
        app.button(key="rapid_spot_translate").click().run()
        assert translator.return_value.translate.call_count == 2
    assert not app.error
    assert "A radio announcement" in rendered_text()
    assert escape_markdown("The station broadcasts an interview.") in rendered_text()
    for key in ("df_jev_sentiment_unique", "df_jev_sentiment_grouped_rows"):
        assert app.session_state[key].iloc[0][columns].tolist() == [
            "A radio announcement", "The station broadcasts an interview.",
        ]


@pytest.mark.parametrize(
    ("group_id", "first", "five", "assignment", "expected"),
    [
        (1720, "NEUTRAL", "NEUTRAL", "POSITIVE", ("POSITIVE", "NEUTRAL")),
        (371, "NEGATIVE", "SOMEWHAT NEGATIVE", "NEGATIVE", ("NEGATIVE", "SOMEWHAT NEGATIVE")),
        (2969, "NOT RELEVANT", "NEUTRAL", "NOT RELEVANT", ("NOT RELEVANT", "NOT RELEVANT")),
    ],
)
def test_rapid_review_displays_current_final_values_after_restore(monkeypatch, group_id, first, five, assignment, expected):
    frame = pd.DataFrame({
        "Group ID": [group_id], "Headline": ["Example headline"],
        "Group Count": [1], "Mentions": [1], "Impressions": [100], "Effective Reach": [80],
        "Jev Sentiment": [first], "Jev 5-Way Sentiment": [five],
        "Jev Sentiment Score": [0.0],
        "Final Rapid Sentiment 3-Way": [float("nan")],
        "Final Rapid Sentiment 5-Way": ["Stale final"],
        "Effective Rapid Sentiment 3-Way": ["Stale machine"],
    })
    frame = apply_rapid_sentiment_human_selection(frame, group_id, scale="3-way", label=assignment)
    app = open_queue(monkeypatch, frame, "Sentiment", overrides={"rapid_spot_review_view": "All Rapid sample coverage"})
    assert not app.exception
    assert [caption.value for caption in app.caption if caption.value.startswith("Final ")] == [
        f"Final 3-way: {expected[0]}", f"Final 5-way: {expected[1]}",
    ]


def test_rapid_review_invalidates_workbook_and_rebuilds_clean_audit(monkeypatch):
    frame, summary = run_scripted_second_opinion(
        monkeypatch,
        jev={"Jev Sentiment": "NEUTRAL", "Jev 5-Way Sentiment": "NEUTRAL"},
        luna={"sentiment_outcome": "3"},
    )
    assert summary["successful"] == 1
    articles = pd.DataFrame({
        "Group ID": [10, 10], "Headline": ["Original headline", "Another copy"],
        "Translated Headline": [f"Error: {TooManyRequests()}", "Valid translation"],
        "Translated Body": [f"Partial translation Error: {TooManyRequests()}", None],
    })
    state = {
        "standard_step": True,
        "export_name": "Rapid audit",
        "client_name": "Example",
        "include_labeling_audit_columns": True,
        "df_traditional": articles,
        "df_jev_sentiment_grouped_rows": articles,
        "df_jev_sentiment_unique": frame,
        "jev_tag_definitions": QUEUE_TAGS,
    }
    download = AppTest.from_file("pages/11-Download.py")
    for key, value in state.items():
        download.session_state[key] = value
    download.run()
    download.button(key="build_clean_workbook").click().run()
    assert not download.error and not download.exception
    before = download.session_state["clean_excel_bytes"]

    review = open_queue(monkeypatch, frame, "Sentiment", overrides={
        **state, "rapid_spot_review_view": "All Rapid sample coverage",
        "clean_excel_bytes": before,
        "clean_excel_built_at": download.session_state["clean_excel_built_at"],
    })
    review.button(key="rapid_assign_sentiment_10_3-way_NEGATIVE").click().run()
    assert not review.exception
    assert "clean_excel_bytes" not in review.session_state
    assert "clean_excel_built_at" not in review.session_state
    del download.session_state["clean_excel_bytes"]
    del download.session_state["clean_excel_built_at"]
    for key in ("df_jev_sentiment_unique", "_rapid_clean_workbook_needs_rebuild"):
        download.session_state[key] = review.session_state[key]
    download.run()
    assert not download.get("download_button")
    assert any("Build the cleaned data workbook again" in item.value for item in download.warning)
    download.button(key="build_clean_workbook").click().run()
    assert not download.error and not download.exception
    assert any(button.label == "Download cleaned data workbook" for button in download.get("download_button"))
    assert "_rapid_clean_workbook_needs_rebuild" not in download.session_state
    after = download.session_state["clean_excel_bytes"]

    for data, final in ((before, "NEUTRAL"), (after, "NEGATIVE")):
        with pd.ExcelFile(io.BytesIO(data)) as workbook:
            for sheet in ("RAPID LABELING SAMPLE", "CLEAN TRAD"):
                rows = workbook.parse(sheet).set_index("Headline")
                assert rows["Final Rapid Sentiment 3-Way"].tolist() == [final, final]
                assert rows["Final Rapid Sentiment 5-Way"].tolist() == ["NEUTRAL", "NEUTRAL"]
                assert rows["Score-band agreement (3-way)"].tolist() == ["Match", "Match"]
                assert rows["Score-band agreement (5-way)"].tolist() == ["Match", "Match"]
                assert "Rapid Sentiment 3-Way Agreement" not in rows.columns
                assert "Rapid Sentiment 5-Way Agreement" not in rows.columns
                assert rows["Rapid Sentiment"].tolist() == ["NEUTRAL", "NEUTRAL"]
                assert not any(name.endswith(("_x", "_y")) for name in rows.columns)
                assert pd.isna(rows.loc["Original headline", "Translated Headline"])
                assert pd.isna(rows.loc["Original headline", "Translated Body"])
                assert rows.loc["Another copy", "Translated Headline"] == "Valid translation"
    assert articles.loc[0, "Translated Headline"].startswith("Error:")

    review.session_state["clean_excel_bytes"] = after
    review.session_state["jev_sentiment_section"] = "Run Jev Analysis"
    review.run()
    next(b for b in review.button if b.label == "Reset Rapid Results").click().run()
    assert not review.exception
    assert "clean_excel_bytes" not in review.session_state
    del download.session_state["clean_excel_bytes"]
    download.session_state["_rapid_clean_workbook_needs_rebuild"] = review.session_state["_rapid_clean_workbook_needs_rebuild"]
    download.session_state["df_jev_sentiment_unique"] = review.session_state["df_jev_sentiment_unique"]
    download.run()
    assert not download.get("download_button")
    assert any("Build the cleaned data workbook again" in item.value for item in download.warning)
    download.button(key="build_clean_workbook").click().run()
    assert not download.error and not download.exception
    assert not any("Build the cleaned data workbook again" in item.value for item in download.warning)
    with pd.ExcelFile(io.BytesIO(download.session_state["clean_excel_bytes"])) as rebuilt:
        assert "RAPID LABELING SAMPLE" not in rebuilt.sheet_names
        assert "Final Rapid Sentiment 3-Way" not in rebuilt.parse("CLEAN TRAD")


def test_rapid_batch_controls_run_500_stories_and_retain_individual_errors(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "mock")
    monkeypatch.setenv("OPENAI_API_KEY", "mock")
    monkeypatch.setattr("processing.jev_sentiment.time", SimpleNamespace(sleep=lambda _: None))
    monkeypatch.setattr("processing.rapid_second_opinion.time", SimpleNamespace(sleep=lambda _: None))
    stories = pd.DataFrame({
        "Group ID": range(1, 504),
        "Headline": [f"Mock story {number}" for number in range(1, 504)],
        "Group Count": 1, "Mentions": 1, "Impressions": 100, "Effective Reach": 80,
    })
    jev_error_id = 497
    luna_error_id = 498

    def mock_jev(_url, *, json, **_kwargs):
        headline = json["state"]["story"]["headline"]
        if headline == f"Mock story {jev_error_id}":
            raise RuntimeError("Mocked Jev failure")
        neutral = {"type": "choice", "choice": "NEUTRAL", "probabilities": {"NEUTRAL": 1}}
        response = {
            "answers": {
                "sentiment": neutral,
                "sentiment_5_way": neutral,
                "sentiment_score": {"type": "score", "score": 4, "legend": {}, "probabilities": {"4": 1}},
                "sentiment_relevant": {"type": "noul", "noul": 1},
                "sentiment_mixture": {"type": "score", "score": 0, "legend": {}, "probabilities": {"0": 1}},
            },
            "mock_headline": headline,
        }
        return SimpleNamespace(status_code=200, raise_for_status=lambda: None, json=lambda: response)

    def mock_luna(*, messages, **_kwargs):
        headline = messages[-1]["content"].split("Headline: ")[-1].splitlines()[0]
        if headline == f"Mock story {luna_error_id}":
            raise RuntimeError("Mocked Luna failure")
        args = {
            "sentiment_outcome": "0",
            "sentiment_confidence": 1,
            "sentiment_rationale": headline,
        }
        tool = SimpleNamespace(function=SimpleNamespace(arguments=json.dumps(args)))
        message = SimpleNamespace(tool_calls=[tool])
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=None)

    app = open_queue(monkeypatch, stories, "Sentiment", timeout=30, overrides={
        "jev_sentiment_section": "Run Jev Analysis",
        "jev_tag_definitions": {},
        "ai_checkpoint_rapid_first_completed": 250,
        "ai_checkpoint_rapid_second_completed": 250,
    })

    def assert_on_demand_checkpoint_only():
        assert not any("checkpoint" in warning.value.lower() for warning in app.warning)
        assert not app.get("download_button")
        assert any("Save & Load" in caption.value for caption in app.caption)

    batch = app.number_input(key="jev_sentiment_batch_size")
    assert (batch.value, batch.max) == (50, 500)
    assert_on_demand_checkpoint_only()
    app.session_state["clean_excel_bytes"] = b"pre-Jev workbook"
    with patch("processing.jev_sentiment.requests.post", side_effect=mock_jev):
        batch.set_value(500).run()
        next(button for button in app.button if button.label == "Run Next Rapid Batch").click().run()
        assert not app.exception
        summary = app.session_state["__last_jev_sentiment_batch_summary__"]
        assert (summary["done"], summary["successful"], len(summary["errors"])) == (500, 499, 1)
        assert "clean_excel_bytes" not in app.session_state
        assert app.session_state["_rapid_clean_workbook_needs_rebuild"]
        assert app.number_input(key="jev_sentiment_batch_size").max == 3
        results = app.session_state["df_jev_sentiment_unique"].set_index("Group ID")
        assert results.loc[jev_error_id, "Jev Error"] == "Mocked Jev failure"
        assert pd.isna(results.loc[501, "Jev Sentiment"])
        next(button for button in app.button if button.label == "Run Next Rapid Batch").click().run()
    assert not app.exception
    results = app.session_state["df_jev_sentiment_unique"].set_index("Group ID")
    completed = results["Jev Sentiment"].eq("NEUTRAL")
    assert completed.sum() == 502
    assert all(json.loads(row["Jev Raw Response"])["mock_headline"] == row["Headline"]
               for _, row in results.loc[completed].iterrows())
    assert_on_demand_checkpoint_only()

    app.session_state["jev_sentiment_section"] = "AI Second Opinion"
    app.run()
    batch = app.number_input(key="rapid_review_batch_size_input")
    assert (batch.value, batch.max) == (50, 500)
    assert_on_demand_checkpoint_only()
    app.session_state["clean_excel_bytes"] = b"pre-Luna workbook"
    with patch("processing.rapid_second_opinion.OpenAI") as client:
        client.return_value.chat.completions.create.side_effect = mock_luna
        batch.set_value(500).run()
        next(button for button in app.button if button.label == "Run AI Second Opinion").click().run()
    assert not app.exception
    summary = app.session_state["__last_rapid_review_batch_summary__"]
    assert (summary["done"], summary["successful"], len(summary["errors"])) == (500, 499, 1)
    assert "clean_excel_bytes" not in app.session_state
    assert app.session_state["_rapid_clean_workbook_needs_rebuild"]
    results = app.session_state["df_jev_sentiment_unique"].set_index("Group ID")
    assert results.loc[luna_error_id, "Rapid Review Status"] == "Error"
    assert results.loc[luna_error_id, "Rapid Review Error"] == "Mocked Luna failure"
    assert results.loc[jev_error_id, "Jev Error"] == "Mocked Jev failure"
    assert pd.isna(results.loc[jev_error_id, "Rapid Review Status"])
    completed = results["Rapid Review Status"].eq("Completed")
    assert (results.loc[completed, "Rapid Review Sentiment Rationale"] == results.loc[completed, "Headline"]).all()
    assert app.number_input(key="rapid_review_batch_size_input").max == 3
    assert_on_demand_checkpoint_only()

    save = AppTest.from_file("pages/12-Save_Load.py")
    save.session_state["upload_step"] = True
    save.session_state["standard_step"] = True
    save.session_state["client_name"] = "Batch fixture"
    save.session_state["df_jev_sentiment_unique"] = app.session_state["df_jev_sentiment_unique"]
    save.session_state["_session_file_build_requested"] = True
    with patch("utils.session_download.register_session_download", return_value="/media/checkpoint.bin"):
        save.run()
    assert not save.exception and not save.error
    assert any("Download session file" in item.value for item in save.markdown)
    restored = SessionState()
    load_session_state_from_file(restored, io.BytesIO(save.session_state["_prepared_session_file"]["data"]))
    pd.testing.assert_frame_equal(restored["df_jev_sentiment_unique"].set_index("Group ID"), results)


@pytest.fixture
def scope_stories():
    return pd.DataFrame({
        "Group ID": [100, 100, 200, 300, 400],
        "Headline": ["Radio prime", "Online copy", "TV story", "Radio story", "Online prime"],
        "Snippet": ["Radio body", "Online copy body", "TV body", "Another radio body", "Online body"],
        "URL": [f"https://example.test/{index}" for index in range(5)],
        "Type": ["RADIO", "ONLINE", "TV", "RADIO", "ONLINE"],
        "Prime Example": [1, 0, 1, 1, 1],
        "Group Count": [2, 2, 1, 1, 1],
        "Mentions": 1, "Impressions": 100, "Effective Reach": 80,
    })


@pytest.mark.parametrize("media_types", [[], ["ONLINE"]], ids=["unrestricted", "online"])
@pytest.mark.parametrize("mode", [
    "Use full eligible dataset", "Use representative sample", "Set custom sample size",
    "Reuse prepared Sentiment sample", "Reuse tagging sample",
])
def test_rapid_preparation_applies_scope_and_selects_eligible_representatives(monkeypatch, scope_stories, media_types, mode):
    previous = scope_stories.assign(**{"Example Snippet": "Stale reused body"})
    state = {
        "jev_sentiment_config_step": False,
        "jev_sentiment_section": "Prepare Sample",
        "analysis_dataset_media_types": media_types,
        "df_sentiment_rows": previous,
        "df_tagging_rows": previous,
    }
    app = open_queue(monkeypatch, scope_stories, "Sentiment", overrides=state)
    next(radio for radio in app.radio if radio.label == "Rapid Labeling dataset mode").set_value(mode).run()
    next(button for button in app.button if button.label == "Prepare Rapid Labeling Sample").click().run()
    assert not app.exception
    scoped, _, _ = build_scoped_traditional_export_bundle({
        "df_traditional": scope_stories, "analysis_dataset_media_types": media_types,
    })
    expected_types = set(scoped["Type"])
    for key in ("df_jev_sentiment_rows", "df_jev_sentiment_grouped_rows", "df_jev_sentiment_unique"):
        prepared = app.session_state[key]
        assert set(prepared["Type"]) == expected_types
        assert set(prepared["Group ID"]) == set(scoped["Group ID"])
    unique = app.session_state["df_jev_sentiment_unique"]
    representative = unique.loc[unique["Group ID"].eq(100)].iloc[0]
    headline, body = ("Online copy", "Online copy body") if media_types else ("Radio prime", "Radio body")
    assert representative["Headline"] == headline
    assert "Example Snippet" not in unique
    jev_story = build_jev_combined_sentiment_payload(representative, {})["state"]["story"]
    assert (jev_story["headline"], jev_story["body"]) == (headline, body)
    luna_prompt = build_rapid_review_prompt(representative, {})
    assert f"Headline: {headline}\n" in luna_prompt and luna_prompt.endswith(f"Text: {body}")
    labeled = unique.assign(**{
        "Jev Sentiment": "NEUTRAL", "Jev 5-Way Sentiment": "NEUTRAL",
        "Jev Sentiment Score": 0.0, "Jev Mixture Label": "VERY CONSISTENT",
    })
    assert set(build_rapid_review_candidates(labeled)["Group ID"]) == set(scoped["Group ID"])
    app.session_state["df_jev_sentiment_unique"] = labeled
    app.session_state["jev_sentiment_section"] = "Spot Checks"
    app.session_state["rapid_spot_review_view"] = "All Rapid sample coverage"
    assert queue_size(app.run()) == scoped["Group ID"].nunique()


@pytest.mark.parametrize("legacy", [False, True], ids=["recorded-scope", "legacy"])
def test_rapid_scope_change_blocks_both_passes_and_requires_confirmed_replacement(monkeypatch, scope_stories, legacy):
    app = open_queue(monkeypatch, scope_stories, "Sentiment", overrides={
        "jev_sentiment_config_step": False, "jev_sentiment_section": "Prepare Sample",
    })
    next(button for button in app.button if button.label == "Prepare Rapid Labeling Sample").click().run()
    saved = app.session_state["df_jev_sentiment_unique"].assign(**{
        "Jev Sentiment": "NEUTRAL", "Rapid Human Sentiment": "POSITIVE",
    })
    app.session_state["df_jev_sentiment_unique"] = saved
    if legacy:
        del app.session_state["rapid_sample_scope_signature"]
        app.session_state["jev_analysis_context_fingerprint"] = ""
    app.session_state["analysis_dataset_media_types"] = ["ONLINE"]
    for section, run_label in (
        ("Run Jev Analysis", "Run Next Rapid Batch"), ("AI Second Opinion", "Run AI Second Opinion"),
    ):
        app.session_state["jev_sentiment_section"] = section
        app.run()
        assert not app.exception
        assert any("saved Rapid sample does not match" in item.value for item in app.warning)
        assert not any(button.label == run_label for button in app.button)
        pd.testing.assert_frame_equal(app.session_state["df_jev_sentiment_unique"], saved)
    app.button(key="jev_sentiment_nav_prepare").click().run()
    prepare = next(button for button in app.button if button.label == "Prepare Rapid Labeling Sample")
    assert prepare.disabled
    pd.testing.assert_frame_equal(app.session_state["df_jev_sentiment_unique"], saved)
    app.checkbox(key="rapid_confirm_scope_replacement").check().run()
    next(button for button in app.button if button.label == "Prepare Rapid Labeling Sample").click().run()
    assert not app.exception
    prepared = app.session_state["df_jev_sentiment_unique"]
    assert prepared["Type"].tolist() == ["ONLINE", "ONLINE"]
    assert prepared["Rapid Human Sentiment"].isna().all()


def test_rapid_scope_matches_clean_trad_dates_flags_and_keep_exceptions(scope_stories):
    stories = scope_stories.assign(**{
        "Date": pd.to_datetime(["2026-04-04", "2026-04-05", "2026-04-05", "2026-04-06", "2026-04-10"]),
        "Coverage Flags": ["", "Advertorial", "", "", ""],
    })
    keys = build_coverage_row_key_series(stories)
    state = SessionState(
        df_traditional=stories,
        analysis_dataset_media_types=["ONLINE"],
        analysis_dataset_end_date="2026-04-06",
        analysis_dataset_excluded_flags=["Advertorial"],
        analysis_dataset_exclusion_keep_keys=[keys.iloc[2]],
    )
    clean, _, _ = build_scoped_traditional_export_bundle(state)
    rapid = scoped_rapid_source_rows(state)
    assert rapid["Group ID"].tolist() == [200]
    pd.testing.assert_frame_equal(rapid, clean)


def test_rapid_workbook_scope_omits_legacy_results_and_invalidates_cached_downloads(monkeypatch, scope_stories):
    monkeypatch.setenv("MIG_ENABLE_RAPID_LABELING", "1")
    saved = scope_stories.loc[scope_stories["Prime Example"].eq(1)].assign(**{
        "Jev Sentiment": "NEGATIVE", "Jev 5-Way Sentiment": "SOMEWHAT NEGATIVE",
        "Rapid Human Sentiment Review State": "Assigned", "Rapid Human Sentiment": "POSITIVE",
        "Rapid Human Sentiment Scale": "3-way",
    })
    state = {
        "standard_step": True, "client_name": "Example", "export_name": "Scope regression",
        "include_labeling_audit_columns": True,
        "df_traditional": scope_stories.assign(**{
            "Jev Sentiment": "Stale input", "Score-band agreement (3-way)": "Stale audit",
        }),
        "df_jev_sentiment_grouped_rows": scope_stories,
        "df_jev_sentiment_unique": saved,
    }
    download = AppTest.from_file("pages/11-Download.py")
    for key, value in state.items():
        download.session_state[key] = value
    download.run()
    download.button(key="build_clean_workbook").click().run()
    assert not download.exception and not download.error
    with pd.ExcelFile(io.BytesIO(download.session_state["clean_excel_bytes"])) as workbook:
        assert "RAPID LABELING SAMPLE" in workbook.sheet_names

    download.session_state["analysis_dataset_media_types"] = ["ONLINE"]
    download.run()
    assert not download.exception
    assert any("workbook will omit its Rapid results" in msg.value for msg in download.warning)
    assert any("restore a compatible scope or prepare a new sample" in msg.value for msg in download.warning)
    assert not download.get("download_button")
    assert "clean_excel_bytes" not in download.session_state
    download.button(key="build_clean_workbook").click().run()
    assert not download.exception and not download.error
    assert any(b.label == "Download cleaned data workbook" for b in download.get("download_button"))
    pd.testing.assert_frame_equal(download.session_state["df_jev_sentiment_unique"], saved)
    with pd.ExcelFile(io.BytesIO(download.session_state["clean_excel_bytes"])) as workbook:
        assert "RAPID LABELING SAMPLE" not in workbook.sheet_names
        clean = workbook.parse("CLEAN TRAD")
        assert clean["Type"].tolist() == ["ONLINE", "ONLINE"]
        assert all("Rapid" not in col and not col.startswith(("Jev ", "Score-band")) for col in clean)
        meta = workbook.parse("EXPORT METADATA").set_index("Field")["Value"]
        assert "Rapid Labeling Run" not in meta and "Rapid Labeling Processed Groups" not in meta
        assert meta["Rapid Labeling Export"].startswith("Omitted:")

    review = open_queue(monkeypatch, saved, "Sentiment", overrides={
        **state,
        "analysis_dataset_media_types": ["ONLINE"],
        "jev_sentiment_section": "Prepare Sample",
        "clean_excel_bytes": download.session_state["clean_excel_bytes"],
        "clean_excel_built_at": download.session_state["clean_excel_built_at"],
        "_clean_excel_scope_key": download.session_state["_clean_excel_scope_key"],
    })
    review.checkbox(key="rapid_confirm_scope_replacement").check().run()
    next(b for b in review.button if b.label == "Prepare Rapid Labeling Sample").click().run()
    assert not review.exception
    assert "clean_excel_bytes" not in review.session_state
    assert "_clean_excel_scope_key" not in review.session_state
    assert review.session_state["_rapid_clean_workbook_needs_rebuild"]
    fresh = review.session_state["df_jev_sentiment_unique"]
    assert fresh["Type"].tolist() == ["ONLINE", "ONLINE"]
    assert fresh["Rapid Human Sentiment"].isna().all()
    fresh = fresh.assign(**{"Jev Sentiment": "NEGATIVE", "Jev 5-Way Sentiment": "SOMEWHAT NEGATIVE"})
    review.session_state["df_jev_sentiment_unique"] = fresh
    with pd.ExcelFile(io.BytesIO(build_clean_workbook_bytes(review.session_state.filtered_state))) as workbook:
        assert workbook.parse("RAPID LABELING SAMPLE")["Final Rapid Sentiment 3-Way"].tolist() == ["NEGATIVE", "NEGATIVE"]
        assert workbook.parse("CLEAN TRAD")["Final Rapid Sentiment 3-Way"].tolist() == ["NEGATIVE", "NEGATIVE"]


def test_download_word_guidance_changes_from_regenerate_to_rebuild(monkeypatch, scope_stories):
    monkeypatch.setenv("MIG_ENABLE_RAPID_LABELING", "1")
    frame = scope_stories.loc[scope_stories["Prime Example"].eq(1)].reset_index(drop=True).assign(**{
        "Jev Sentiment": "POSITIVE", "Jev 5-Way Sentiment": "SOMEWHAT POSITIVE",
    })
    state = {
        "standard_step": True, "client_name": "Example", "export_name": "Word freshness",
        "df_traditional": scope_stories, "df_jev_sentiment_unique": frame,
        "rapid_sentiment_observation_output": {"overall_observation": "Original Rapid observations"},
        "sentiment_observation_output": {"overall_observation": "Established observations"},
    }

    def fingerprint(rows):
        return rapid_observation_current_fingerprint(
            family="sentiment", unique_df=rows, analysis_context=build_sentiment_analysis_context_text(state),
        )

    state["rapid_sentiment_observation_fingerprint"] = fingerprint(frame)
    app = AppTest.from_file("pages/11-Download.py")
    for key, value in state.items():
        app.session_state[key] = value
    app.run()
    build = "build_report_copy_document"
    app.button(key=build).click().run()
    assert not app.error and not app.exception
    original = app.session_state["report_copy_docx_bytes"]

    revised = apply_rapid_sentiment_human_selection(frame, 100, scale="3-way", label="NEGATIVE")
    app.session_state["df_jev_sentiment_unique"] = revised
    app.run()
    assert app.button(key=build).disabled
    assert any("Regenerate Rapid sentiment observations" in warning.value for warning in app.warning)
    assert not any(b.label == "Download report copy document" for b in app.get("download_button"))
    assert app.session_state["report_copy_docx_bytes"] == original

    app.session_state["rapid_sentiment_observation_output"] = {"overall_observation": "Regenerated Rapid observations"}
    app.session_state["rapid_sentiment_observation_fingerprint"] = fingerprint(revised)
    app.run()
    assert not app.button(key=build).disabled
    assert not any("Regenerate Rapid sentiment observations" in warning.value for warning in app.warning)
    assert any("Build the report copy document again" in warning.value for warning in app.warning)
    app.button(key=build).click().run()
    assert not app.warning and not app.error and not app.exception
    assert any(b.label == "Download report copy document" for b in app.get("download_button"))
    with zipfile.ZipFile(io.BytesIO(app.session_state["report_copy_docx_bytes"])) as document:
        xml = document.read("word/document.xml").decode()
        assert "Regenerated Rapid observations" in xml and "Established observations" in xml

    app.session_state["analysis_dataset_media_types"] = ["ONLINE"]
    app.run()
    assert app.button(key=build).disabled
    assert any("Before rebuilding the Word report" in warning.value and "restore a compatible scope" in warning.value
               for warning in app.warning)
    assert not any(b.label == "Download report copy document" for b in app.get("download_button"))
    pd.testing.assert_frame_equal(app.session_state["df_jev_sentiment_unique"], revised)
    del app.session_state["analysis_dataset_media_types"]
    app.run()
    assert not app.button(key=build).disabled
    assert not any("restore a compatible scope" in warning.value for warning in app.warning)
    assert any(b.label == "Download report copy document" for b in app.get("download_button"))
