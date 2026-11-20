import io
import json
import zipfile
from unittest.mock import patch

import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from processing.ai_sentiment import ensure_ai_sentiment_columns
from processing.ai_tagging import ensure_tag_review_columns


@pytest.fixture
def coverage(monkeypatch):
    monkeypatch.setenv("MIG_ENABLE_RAPID_LABELING", "1")
    monkeypatch.setattr(st, "secrets", {"key": "mock"})
    return pd.DataFrame({
        "Group ID": [11, 12],
        "Prime Example": [1, 1],
        "Group Count": [1, 1],
        "Headline": ["New bus service", "Radio interview"],
        "Snippet": ["New routes are open.", "The agency was interviewed."],
        "Type": ["ONLINE", "RADIO"],
        "Source": ["QA", "QA"],
        "URL": ["https://example.test/bus", "https://example.test/radio"],
        "Date": pd.to_datetime(["2026-01-01", "2026-01-02"]),
        "Mentions": [1, 1],
        "Impressions": [100, 50],
        "Effective Reach": [100, 50],
    })


def open_page(path, state):
    app = AppTest.from_file(path, default_timeout=20)
    for key, value in state.items():
        if not key.startswith(("sentiment_nav_", "tagging_nav_")):
            app.session_state[key] = value
    app.run()
    assert not app.exception and not app.error
    return app


def base_state(coverage):
    return {
        "standard_step": True, "upload_step": True,
        "client_name": "Example Transit", "export_name": "QA",
        "analysis_context_saved_at": "saved",
        "analysis_primary_names": ["Example Transit"],
        "df_traditional": coverage,
    }


def buttons(app):
    assert not app.exception and not app.error
    return {button.label for button in app.get("download_button") if not button.proto.disabled}


def build(app, widget):
    app.button(key=widget).click().run()
    assert not app.exception and not app.error


def docx_text(app):
    with zipfile.ZipFile(io.BytesIO(app.session_state["report_copy_docx_bytes"])) as document:
        return document.read("word/document.xml").decode()


@pytest.mark.parametrize("family", ["sentiment", "tagging"])
def test_established_review_marks_workbook_and_requires_word_regeneration(coverage, family):
    state = base_state(coverage)
    is_sentiment = family == "sentiment"
    page = "pages/Sentiment.py" if is_sentiment else "pages/8-Tagging.py"
    grouped = coverage.copy()
    if is_sentiment:
        grouped["AI Sentiment"] = "NEUTRAL"
        grouped["AI Sentiment Confidence"] = 90
        state["sentiment_type"] = "3-way"
        nav = "sentiment_nav_distribution"
        observe = "generate_sentiment_observations"
        export_col, old, corrected = "Final Sentiment", "NEUTRAL", "NEGATIVE"
    else:
        grouped["AI Tag"] = "Operations"
        grouped["Tag_Processed"] = True
        state["tagging_mode"] = "Multiple applicable tags"
        state["tag_definitions"] = {"Access": "Access to services", "Operations": "Service operations"}
        nav = "tagging_nav_insights"
        observe = "generate_tag_observations"
        export_col, old, corrected = "Final Tag", "Operations", "Access, Operations"
    ensure_columns = ensure_ai_sentiment_columns if is_sentiment else ensure_tag_review_columns
    unique, grouped = ensure_columns(grouped, grouped)
    state.update({
        f"{family}_config_step": True, f"{family}_sample_mode": "full",
        f"df_{family}_rows": grouped, f"df_{family}_grouped_rows": grouped,
        f"df_{family}_unique": unique,
    })
    generator_module = f"processing.ai_{family}"
    insights = open_page(page, state)
    insights.button(key=nav).click().run()
    with patch(f"{generator_module}.{observe}", return_value=({"overall_observation": "Original observations"}, 0, 0)):
        build(insights, observe)

    download = open_page("pages/11-Download.py", insights.session_state.filtered_state)
    build(download, "build_clean_workbook")
    build(download, "build_report_copy_document")
    build(download, "build_notebooklm_bundle")
    old_xlsx = download.session_state["clean_excel_bytes"]
    old_docx = download.session_state["report_copy_docx_bytes"]
    old_bundle = download.session_state["notebooklm_zip_bytes"]
    assert "Original observations" in docx_text(download)
    assert not download.warning

    review = open_page(page, download.session_state.filtered_state)
    if is_sentiment:
        review.button(key="sentiment_nav_review").click().run()
        review.selectbox(key="spotcheck_review_mode").set_value("All Toned Coverage").run()
        next(button for button in review.button if button.label == "NEGATIVE").click().run()
    else:
        review.button(key="tagging_nav_spot_checks").click().run()
        review.selectbox(key="tagging_review_mode").set_value("All Tagged Coverage").run()
        next(c for c in review.checkbox if str(c.key).startswith("tag_review_multi_") and c.label == "Access").check().run()
        next(button for button in review.button if button.label == "Save & next").click().run()

    download = open_page("pages/11-Download.py", review.session_state.filtered_state)
    assert buttons(download) == {
        "Download earlier cleaned data workbook", "Download earlier report copy document", "Download NotebookLM bundle",
    }
    assert download.session_state["clean_excel_bytes"] == old_xlsx
    assert download.session_state["report_copy_docx_bytes"] == old_docx
    assert download.session_state["notebooklm_zip_bytes"] == old_bundle
    assert any("Regenerate" in warning.value and "observations" in warning.value for warning in download.warning)
    assert download.button(key="build_report_copy_document").disabled
    build(download, "build_clean_workbook")
    assert download.button(key="build_report_copy_document").disabled
    assert "Download cleaned data workbook" in buttons(download)
    assert any("Regenerate" in warning.value for warning in download.warning)
    assert not any("cleaned data workbook" in warning.value for warning in download.warning)
    earlier_rows = pd.read_excel(io.BytesIO(old_xlsx), sheet_name="CLEAN TRAD")
    rebuilt_rows = pd.read_excel(io.BytesIO(download.session_state["clean_excel_bytes"]), sheet_name="CLEAN TRAD")
    assert set(earlier_rows[export_col]) == {old}
    assert set(rebuilt_rows[export_col]) == {old, corrected}

    insights = open_page(page, download.session_state.filtered_state)
    insights.button(key=nav).click().run()
    with patch(f"{generator_module}.{observe}", return_value=({"overall_observation": "Regenerated observations"}, 0, 0)):
        build(insights, observe)
    download = open_page("pages/11-Download.py", insights.session_state.filtered_state)
    assert not download.button(key="build_report_copy_document").disabled
    assert "Download earlier report copy document" in buttons(download)
    assert any("Rebuild it before using it" in warning.value for warning in download.warning)
    build(download, "build_report_copy_document")
    assert not download.warning
    assert "Download report copy document" in buttons(download)
    assert "Regenerated observations" in docx_text(download)
    assert "Original observations" not in docx_text(download)


def notebook_stories(data):
    with zipfile.ZipFile(io.BytesIO(data)) as bundle:
        return {row["Headline"] for filename in bundle.namelist() for row in json.loads(bundle.read(filename))}


def test_notebook_scope_change_keeps_previous_archive_and_rebuilds(coverage):
    app = open_page("pages/11-Download.py", base_state(coverage))
    build(app, "build_notebooklm_bundle")
    old = app.session_state["notebooklm_zip_bytes"]
    assert notebook_stories(old) == {"New bus service", "Radio interview"}
    assert buttons(app) == {"Download NotebookLM bundle"}

    app.session_state["analysis_dataset_media_types"] = ["ONLINE"]
    app.run()
    assert app.session_state["notebooklm_zip_bytes"] == old
    assert buttons(app) == {"Download earlier NotebookLM bundle"}
    assert any("Rebuild it before using it" in warning.value for warning in app.warning)
    assert all("Current" not in caption.value for caption in app.caption)
    build(app, "build_notebooklm_bundle")
    assert not app.warning
    assert buttons(app) == {"Download NotebookLM bundle"}
    assert notebook_stories(app.session_state["notebooklm_zip_bytes"]) == {"New bus service"}


def test_notebook_ignores_edits_to_rows_outside_its_sample(coverage):
    from processing.notebooklm_exports import compile_notebooklm_bundle

    def small_bundle(rows, client, *, write_archive):
        return compile_notebooklm_bundle(rows, client, max_files=1, max_rows_per_file=1, write_archive=write_archive)

    with patch("processing.notebooklm_exports.compile_notebooklm_bundle", side_effect=small_bundle):
        app = open_page("pages/11-Download.py", base_state(coverage))
        build(app, "build_notebooklm_bundle")
        old = app.session_state["notebooklm_zip_bytes"]
        included, = notebook_stories(old)
        updated = coverage.copy()
        updated.loc[updated["Headline"].ne(included), "Headline"] = "Unselected edit"
        app.session_state["df_traditional"] = updated
        app.run()
        assert buttons(app) == {"Download NotebookLM bundle"}
        assert not app.warning
        assert app.session_state["notebooklm_zip_bytes"] == old


def test_regions_warning_explains_omission_until_regeneration(coverage):
    from processing.download_exports import _build_regions_signature_for_export, _regions_export_source

    state = base_state(coverage.assign(Country=["Canada", "United States"]))
    state.update(regions_prepared=True, regions_analysis_levels=["Countries"])
    signature = _build_regions_signature_for_export(state, _regions_export_source(state))
    state["regions_generated_output"] = {
        "Countries": {"signature": signature, "content": {"overall_observation": "Original region narrative"}},
    }
    app = open_page("pages/11-Download.py", state)
    build(app, "build_clean_workbook")
    build(app, "build_report_copy_document")
    assert "Original region narrative" in docx_text(app)
    assert not app.warning

    app.session_state["regions_include_countries"] = ["Canada"]
    app.run()
    assert any("Rebuilding alone will omit" in warning.value for warning in app.warning)
    assert "Download earlier cleaned data workbook" in buttons(app)
    assert "Download earlier report copy document" in buttons(app)
    build(app, "build_clean_workbook")
    workbook = io.BytesIO(app.session_state["clean_excel_bytes"])
    assert pd.read_excel(workbook, sheet_name="Regions")["Overall Observation"].isna().all()
    assert any("Rebuilding alone will omit" in warning.value for warning in app.warning)

    state = app.session_state.filtered_state
    signature = _build_regions_signature_for_export(state, _regions_export_source(state))
    app.session_state["regions_generated_output"] = {
        "Countries": {"signature": signature, "content": {"overall_observation": "Regenerated region narrative"}},
    }
    app.run()
    assert not any("Rebuilding alone will omit" in warning.value for warning in app.warning)
    build(app, "build_report_copy_document")
    build(app, "build_clean_workbook")
    assert not app.warning
    assert "Regenerated region narrative" in docx_text(app)
    workbook = io.BytesIO(app.session_state["clean_excel_bytes"])
    assert set(pd.read_excel(workbook, sheet_name="Regions")["Overall Observation"]) == {"Regenerated region narrative"}
