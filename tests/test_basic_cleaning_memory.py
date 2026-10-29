from pathlib import Path
from unittest.mock import patch

import pandas as pd
from pandas.testing import assert_frame_equal
from streamlit.testing.v1 import AppTest

from processing.standard_cleaning import run_standard_cleaning
from utils.io import normalize_uploaded_dataframe


ROOT = Path(__file__).resolve().parents[1]


def baseline(raw):
    normalized = normalize_uploaded_dataframe(raw)
    for column in ["Sentiment", "Continent", "Country", "Prov/State", "City", "Language"]:
        if column in normalized.columns:
            normalized[column] = normalized[column].astype("category")
    return normalized


def cleaning_page():
    raw = pd.read_csv(ROOT / "tests/fixtures/agility/agility_golden_corpus.csv")
    app = AppTest.from_file(str(ROOT / "pages/2-Basic_Cleaning.py"), default_timeout=30)
    app.session_state["df_untouched"] = raw
    app.session_state["df_traditional"] = baseline(raw)
    app.session_state["upload_step"] = True
    return app, raw


def button(app, label):
    return next(item for item in app.button if item.label == label)


def refresh_after_form_submission(app):
    # AppTest 1.56 retains removed form widgets after st.rerun; do not replay
    # their stale widget states when rendering the completed page again.
    return app._run()


def test_page_reruns_and_completed_cleaning_reuse_summary_without_normalizing():
    app, raw = cleaning_page()
    with patch("utils.io.normalize_uploaded_dataframe", wraps=normalize_uploaded_dataframe) as normalize:
        app.run()
        assert not app.exception
        assert normalize.call_count == 1
        app.run()
        assert not app.exception
        assert normalize.call_count == 1
        summary = app.session_state["_basic_cleaning_source_summary"]
        assert summary["row_count"] == len(raw)
        assert not any(isinstance(value, pd.DataFrame) for value in summary.values())

        button(app, "Run Basic Cleaning").click().run()
        assert not app.exception
        assert app.session_state["standard_step"] is True
        assert normalize.call_count == 2
        refresh_after_form_submission(app)
        assert not app.exception
        assert normalize.call_count == 2
        assert len(app.session_state["df_traditional"]) == 49
        assert len(app.session_state["df_social"]) == 47
        assert len(app.session_state["df_dupes"]) == 4
        assert_frame_equal(app.session_state["df_untouched"], raw)


def test_summary_refreshes_after_source_replacement_and_reset_restores_baseline():
    app, raw = cleaning_page()
    app.run()
    replacement = raw.iloc[:-1].copy()
    app.session_state["df_untouched"] = replacement
    app.run()
    assert not app.exception
    assert app.session_state["_basic_cleaning_source_summary"]["row_count"] == len(replacement)

    button(app, "Run Basic Cleaning").click().run()
    assert not app.exception
    refresh_after_form_submission(app)
    button(app, "Reset Basic Cleaning").click().run()
    assert not app.exception
    assert app.session_state["standard_step"] is False
    assert_frame_equal(app.session_state["df_traditional"], baseline(replacement))


def test_staged_cleaning_starts_from_raw_data_and_keeps_all_three_steps():
    app, raw = cleaning_page()
    # This models a restored session whose working data differs from the source.
    app.session_state["df_traditional"] = baseline(raw.iloc[:1])
    with patch("pandas.DataFrame.memory_usage") as memory:
        memory.return_value = pd.Series([60 * 1024 * 1024])
        app.run()
    assert not app.exception
    button(app, "Run Basic Cleaning").click().run()
    assert not app.exception
    assert app.session_state["basic_cleaning_stage"] == 1
    expected = run_standard_cleaning(baseline(raw), add_flags=False)
    for key in ["df_traditional", "df_social", "df_dupes"]:
        assert_frame_equal(app.session_state[key], expected[key])
    button(app, "Continue").click().run()
    assert not app.exception
    assert app.session_state["basic_cleaning_stage"] == 2
    button(app, "Finish Basic Cleaning").click().run()
    assert not app.exception
    assert app.session_state["standard_step"] is True
    assert len(app.session_state["df_traditional"]) == 49
