from __future__ import annotations

import pandas as pd

from processing.standard_cleaning import SOCIAL_TYPES


def _nonblank_text_mask(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("").str.strip().ne("")


def _format_percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def build_data_quality_warnings(df: pd.DataFrame) -> list[str]:
    if df is None or df.empty:
        return []

    warnings_list: list[str] = []
    total_rows = len(df)

    def add_missing_or_sparse_text_warning(
        column: str,
        *,
        label: str,
        blank_warn_threshold: float,
        impact: str,
    ) -> None:
        if column not in df.columns:
            warnings_list.append(f"`{label}` is missing. {impact}")
            return
        nonblank_mask = _nonblank_text_mask(df[column])
        blank_count = int((~nonblank_mask).sum())
        blank_rate = blank_count / total_rows if total_rows else 0.0
        if blank_rate >= blank_warn_threshold:
            warnings_list.append(
                f"`{label}` is blank on {blank_count:,} of {total_rows:,} rows ({_format_percent(blank_rate)}). {impact}"
            )

    def add_numeric_warning(
        column: str,
        *,
        label: str,
        impact: str,
        subset_mask: pd.Series | None = None,
    ) -> None:
        working = df
        if subset_mask is not None:
            working = df[subset_mask.reindex(df.index, fill_value=False)].copy()
        if working.empty:
            return
        if column not in working.columns:
            warnings_list.append(f"`{label}` is missing. {impact}")
            return
        raw = working[column]
        nonblank_mask = _nonblank_text_mask(raw)
        nonblank_count = int(nonblank_mask.sum())
        if nonblank_count == 0:
            warnings_list.append(f"`{label}` is present but blank on all relevant rows. {impact}")
            return
        coerced = pd.to_numeric(raw, errors="coerce")
        invalid_mask = nonblank_mask & coerced.isna()
        invalid_count = int(invalid_mask.sum())
        invalid_rate = invalid_count / nonblank_count if nonblank_count else 0.0
        if invalid_count > 0:
            warnings_list.append(
                f"`{label}` has {invalid_count:,} non-numeric value(s) out of {nonblank_count:,} populated rows ({_format_percent(invalid_rate)}). {impact}"
            )

    add_missing_or_sparse_text_warning(
        "Type",
        label="Type",
        blank_warn_threshold=0.25,
        impact="Media-type cleanup, effective reach, Top Stories, and Regions may be weaker.",
    )
    add_missing_or_sparse_text_warning(
        "Headline",
        label="Headline",
        blank_warn_threshold=0.25,
        impact="Grouping, Top Stories, Missing Authors, and several AI review workflows may be weaker.",
    )
    add_missing_or_sparse_text_warning(
        "Outlet",
        label="Outlet",
        blank_warn_threshold=0.25,
        impact="Deduping, outlet analysis, author-outlet assignment, and coverage flags may be weaker.",
    )
    add_missing_or_sparse_text_warning(
        "URL",
        label="URL",
        blank_warn_threshold=0.50,
        impact="Validation, example selection, and report-copy linking may be weaker.",
    )

    if "Date" not in df.columns:
        warnings_list.append("`Date` is missing after upload normalization. Trend charts, date scope, and time-based grouping may be limited.")
    else:
        parsed_dates = pd.to_datetime(df["Date"], errors="coerce")
        invalid_dates = int(parsed_dates.isna().sum())
        if invalid_dates > 0:
            invalid_rate = invalid_dates / total_rows if total_rows else 0.0
            warnings_list.append(
                f"`Date` is missing or invalid on {invalid_dates:,} of {total_rows:,} rows ({_format_percent(invalid_rate)}). Trend charts, date scope, and time-based grouping may be incomplete."
            )

    add_numeric_warning(
        "Impressions",
        label="Impressions",
        impact="Effective Reach and visibility-based rankings may be incomplete.",
    )

    social_mask = pd.Series(False, index=df.index)
    if "Type" in df.columns:
        normalized_types = df["Type"].fillna("").astype(str).str.strip().str.upper()
        social_mask = normalized_types.isin(SOCIAL_TYPES)
    social_count = int(social_mask.sum())
    if social_count > 0:
        add_numeric_warning(
            "Engagements",
            label="Engagements",
            impact="Social Effective Reach cannot be calculated correctly for those rows.",
            subset_mask=social_mask,
        )

    geography_cols = [col for col in ["Country", "Prov/State", "City"] if col in df.columns]
    if not geography_cols:
        warnings_list.append("Geography columns are missing (`Country`, `Prov/State`, `City`). Regions analysis may be empty or low value.")
    else:
        geo_has_value = pd.Series(False, index=df.index)
        for col in geography_cols:
            geo_has_value = geo_has_value | _nonblank_text_mask(df[col])
        geo_blank_count = int((~geo_has_value).sum())
        geo_blank_rate = geo_blank_count / total_rows if total_rows else 0.0
        if geo_blank_rate >= 0.90:
            warnings_list.append(
                f"Geography fields are blank on {geo_blank_count:,} of {total_rows:,} rows ({_format_percent(geo_blank_rate)}). Regions analysis may be sparse or empty."
            )

    return warnings_list
