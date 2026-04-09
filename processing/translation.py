# translation.py
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Iterable

import pandas as pd
from deep_translator import GoogleTranslator
from deep_translator.exceptions import RequestError
from titlecase import titlecase


TRANSLATION_COLUMNS = ["Headline", "Snippet", "Contextual Snippet", "Language"]


def init_translation_state(session_state) -> None:
    if "translated_headline" not in session_state:
        session_state.translated_headline = False

    if "translated_snippet" not in session_state:
        session_state.translated_snippet = False

    if "translated_summary" not in session_state:
        session_state.translated_summary = False


def ensure_translation_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure expected translation columns exist."""
    if df is None or df.empty:
        return pd.DataFrame(columns=TRANSLATION_COLUMNS)

    out = df.copy()
    for col in TRANSLATION_COLUMNS:
        if col not in out.columns:
            out[col] = ""

    return out


def count_non_english_records(df: pd.DataFrame) -> int:
    if df is None or df.empty or "Language" not in df.columns:
        return 0
    return int((df["Language"] != "English").sum())


def get_non_english_records(
    df: pd.DataFrame,
    display_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Return only non-English rows for preview."""
    if df is None or df.empty or "Language" not in df.columns:
        return pd.DataFrame()

    out = df[df["Language"] != "English"].copy()

    if display_columns is None:
        display_columns = ["Outlet", "Headline", "Snippet", "Language", "Country"]

    existing_cols = [c for c in display_columns if c in out.columns]
    return out[existing_cols].copy()


def translate_text(text: str) -> tuple[str, bool]:
    text = str(text or "")
    if not text.strip():
        return text, False

    try:
        translated = GoogleTranslator(source="auto", target="en").translate(text[:3900])
        return translated, False
    except RequestError:
        return text, True
    except Exception:
        return text, True


def build_translation_map(texts: Iterable[str], max_workers: int = 5) -> tuple[dict[str, str], int]:
    """
    Translate unique texts and return:
    - mapping of original -> translated
    - count of failed translations
    """
    unique_texts = list({str(t) for t in texts if str(t).strip()})

    if not unique_texts:
        return {}, 0

    def worker(text: str) -> tuple[str, str, bool]:
        translated, failed = translate_text(text)
        return text, translated, failed

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        results = list(ex.map(worker, unique_texts))

    translation_map = {original: translated for original, translated, _ in results}
    failed_count = sum(1 for _, _, failed in results if failed)

    return translation_map, failed_count



def translate_column(df: pd.DataFrame, column_name: str) -> tuple[pd.DataFrame, int]:
    """Replace non-English values in a column with English translations."""
    out = df.copy()

    if (
        out is None
        or out.empty
        or column_name not in out.columns
        or "Language" not in out.columns
    ):
        return out, 0

    non_english_mask = out["Language"] != "English"
    texts_to_translate = out.loc[non_english_mask, column_name].dropna().astype(str)

    translation_map, failed_count = build_translation_map(texts_to_translate)
    if translation_map:
        out[column_name] = out[column_name].replace(translation_map)

    return out, failed_count



def apply_headline_titlecase_to_traditional(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply titlecase to translated traditional headlines, excluding broadcast.
    Mirrors the legacy behavior.
    """
    out = df.copy()

    if out.empty or "Headline" not in out.columns or "Type" not in out.columns:
        return out

    broadcast_array = ["RADIO", "TV"]
    broadcast = out.loc[out["Type"].isin(broadcast_array)].copy()
    non_broadcast = out.loc[~out["Type"].isin(broadcast_array)].copy()

    non_broadcast["Headline"] = non_broadcast["Headline"].fillna("").map(titlecase)

    out = pd.concat([non_broadcast, broadcast], ignore_index=True)
    return out


def translate_headlines(
    df_traditional: pd.DataFrame,
    df_social: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    trad = ensure_translation_columns(df_traditional).copy()
    social = ensure_translation_columns(df_social).copy()

    if "Original Headline" not in trad.columns:
        trad["Original Headline"] = trad["Headline"]
    if "Original Headline" not in social.columns:
        social["Original Headline"] = social["Headline"]

    trad, _ = translate_column(trad, "Headline")
    trad = apply_headline_titlecase_to_traditional(trad)

    social, _ = translate_column(social, "Headline")

    return trad, social



def translate_snippets(
    df_traditional: pd.DataFrame,
    df_social: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    trad = ensure_translation_columns(df_traditional).copy()
    social = ensure_translation_columns(df_social).copy()

    trad, _ = translate_column(trad, "Snippet")
    social, _ = translate_column(social, "Snippet")

    return trad, social


def headline_translation_done(df_traditional: pd.DataFrame, df_social: pd.DataFrame, translated_headline_flag: bool) -> bool:
    return (
        translated_headline_flag
        or "Original Headline" in df_traditional.columns
        or "Original Headline" in df_social.columns
    )