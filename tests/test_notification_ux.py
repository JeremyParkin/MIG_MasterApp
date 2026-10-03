from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_authors_auto_assign_confirmation_uses_post_rerun_toast() -> None:
    source = (ROOT / "ui" / "authors_view.py").read_text(encoding="utf-8")

    assert "authors_outlets_toast_message" in source
    assert 'st.toast(authors_outlets_toast_message, icon="✅")' in source
    assert 'st.toast(status_message, icon="✅")' in source
    assert "Auto-assigned {prefetch_summary['auto_assigned_now']} perfect match(es)" in source
    assert "st.success(f\"Auto-assigned {prefetch_summary['auto_assigned_now']} perfect match(es)" not in source


def test_translation_completion_relies_on_durable_post_rerun_state() -> None:
    source = (ROOT / "pages" / "5-Translation.py").read_text(encoding="utf-8")

    assert 'st.success("Done translating headlines!")' not in source
    assert 'st.success("Done translating snippets!")' not in source
    assert "st.session_state.translated_headline = True" in source
    assert "st.session_state.translated_snippet = True" in source


def test_analysis_context_suggestions_confirmation_uses_toast() -> None:
    source = (ROOT / "pages" / "Analysis_Context.py").read_text(encoding="utf-8")

    assert "AI context suggestions were added to the fields below." in source
    assert 'icon="✅"' in source
    assert "st.success(\"AI context suggestions were added to the fields below." not in source
    assert "st.session_state.analysis_context_suggestion_success = False" in source
