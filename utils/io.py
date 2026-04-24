# io.py
from __future__ import annotations

import re
import pandas as pd
import streamlit as st


def _normalize_numeric_upload_series(series: pd.Series, *, prefer_integer: bool = False) -> pd.Series:
    """
    Coerce uploaded numeric columns safely.

    If `prefer_integer` is True, keep nullable Int64 only when every non-null
    parsed value is effectively whole-numbered; otherwise preserve as float.
    """
    numeric = pd.to_numeric(series, errors="coerce")
    if not prefer_integer:
        return numeric.fillna(0)

    non_null = numeric.dropna()
    if non_null.empty:
        return numeric.fillna(0).astype("Int64")

    whole_number_mask = (non_null % 1).abs() < 1e-9
    if bool(whole_number_mask.all()):
        return numeric.fillna(0).astype("Int64")

    return numeric.fillna(0)

def detect_original_ave_col(df: pd.DataFrame) -> str | None:
    """
    Detect the original AVE column name from uploaded data, such as:
    AVE(USD), AVE(CAD), or AVE.
    """
    ave_candidates = [
        col for col in df.columns if re.match(r"^AVE\([A-Z]{2,3}\)$", str(col))
    ]
    if not ave_candidates and "AVE" in df.columns:
        ave_candidates = ["AVE"]
    return ave_candidates[0] if ave_candidates else None


def normalize_uploaded_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize uploaded CSV/XLSX data into a consistent internal schema
    regardless of whether the file is raw, app-produced, or partially transformed.
    """
    df = df.copy()

    # Drop near-empty rows early
    df = df.dropna(thresh=3).copy()

    # Standardize likely variant column names into one schema
    rename_map = {
        "Media Type": "Type",
        "Coverage Snippet": "Snippet",
        "Province/State": "Prov/State",
    }
    df.rename(columns=rename_map, inplace=True)

    # Add Mentions if missing
    if "Mentions" not in df.columns:
        df["Mentions"] = 1

    # Normalize Impressions safely
    if "Impressions" in df.columns:
        df["Impressions"] = _normalize_numeric_upload_series(
            df["Impressions"],
            prefer_integer=True,
        )

    # Normalize AVE column variants to internal working name: AVE
    ave_candidates = [
        col for col in df.columns if re.match(r"^AVE\([A-Z]{2,3}\)$", str(col))
    ]
    if not ave_candidates and "AVE" in df.columns:
        ave_candidates = ["AVE"]

    if ave_candidates:
        original_ave_col = ave_candidates[0]
        df.rename(columns={original_ave_col: "AVE"}, inplace=True)
        df["AVE"] = pd.to_numeric(df["AVE"], errors="coerce").fillna(0)

    # Build unified Date column if needed
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    elif "Published Date" in df.columns and "Published Time" in df.columns:
        published_date = pd.to_datetime(df["Published Date"], errors="coerce")
        published_time = df["Published Time"].fillna("").astype(str).str.strip()

        df["Date"] = pd.to_datetime(
            published_date.dt.strftime("%Y-%m-%d").fillna("") + " " + published_time,
            errors="coerce"
        )
    elif "Published Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Published Date"], errors="coerce")

    # Drop raw date columns once Date exists
    if "Date" in df.columns:
        df.drop(["Published Date", "Published Time"], axis=1, inplace=True, errors="ignore")

    # Normalize text-like columns safely
    text_columns = [
        "Headline", "Snippet", "Outlet", "Author", "URL", "Type",
        "Sentiment", "Continent", "Country", "Prov/State", "City", "Language"
    ]
    for col in text_columns:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)

    # Drop columns not needed for app processing
    df.drop(
        [
            "Timezone",
            "Word Count",
            "Duration",
            "Image URLs",
            "Folders",
            "Notes",
            "County",
            "Saved Date",
            "Edited Date",
        ],
        axis=1,
        inplace=True,
        errors="ignore"
    )

    # Move Date to front
    if "Date" in df.columns:
        date_col = df.pop("Date")
        df.insert(0, "Date", date_col)

    return df


def build_upload_quality_report(df_raw: pd.DataFrame, df_normalized: pd.DataFrame) -> dict:
    """
    Build a lightweight upload-quality report for issues that were recoverable
    during normalization but may affect downstream workflows.
    """
    report = {
        "warnings": [],
        "date_issue_examples": pd.DataFrame(),
        "date_issue_indices": [],
        "date_issue_row_numbers": [],
    }

    if df_raw is None or df_raw.empty:
        return report

    raw = df_raw.copy()
    normalized = df_normalized.copy() if isinstance(df_normalized, pd.DataFrame) else pd.DataFrame()

    def _non_blank_text(series: pd.Series) -> pd.Series:
        return series.fillna("").astype(str).str.strip()

    date_source_mask = pd.Series(False, index=raw.index)
    date_source_description = None

    if "Published Date" in raw.columns and "Published Time" in raw.columns:
        published_date_text = _non_blank_text(raw["Published Date"])
        published_time_text = _non_blank_text(raw["Published Time"])
        date_source_mask = (published_date_text != "") | (published_time_text != "")
        date_source_description = "Published Date / Published Time"
        date_example_df = pd.DataFrame(
            {
                "Published Date": published_date_text,
                "Published Time": published_time_text,
            },
            index=raw.index,
        )
    elif "Published Date" in raw.columns:
        published_date_text = _non_blank_text(raw["Published Date"])
        date_source_mask = published_date_text != ""
        date_source_description = "Published Date"
        date_example_df = pd.DataFrame({"Published Date": published_date_text}, index=raw.index)
    elif "Date" in raw.columns:
        raw_date_text = _non_blank_text(raw["Date"])
        date_source_mask = raw_date_text != ""
        date_source_description = "Date"
        date_example_df = pd.DataFrame({"Date": raw_date_text}, index=raw.index)
    else:
        date_example_df = pd.DataFrame(index=raw.index)

    if "Date" in normalized.columns and bool(date_source_mask.any()):
        normalized_dates = pd.to_datetime(normalized["Date"], errors="coerce")
        invalid_mask = date_source_mask & normalized_dates.isna()
        invalid_row_numbers = (raw.index[invalid_mask] + 2).tolist()
        invalid_indices = raw.index[invalid_mask].tolist()

        if invalid_row_numbers:
            report["date_issue_indices"] = invalid_indices
            report["date_issue_row_numbers"] = invalid_row_numbers
            example_numbers = invalid_row_numbers[:5]
            example_text = ", ".join(str(num) for num in example_numbers)
            if len(invalid_row_numbers) > 5:
                example_text += ", ..."

            report["warnings"].append(
                {
                    "title": "Some date values could not be parsed",
                    "message": (
                        f"{len(invalid_row_numbers)} uploaded row(s) had values in {date_source_description} "
                        f"that could not be converted into `Date`. Example row numbers: {example_text}."
                    ),
                }
            )

            examples = date_example_df.loc[invalid_mask].copy().head(5)
            examples.insert(0, "Source Row", (examples.index + 2).astype(int))
            report["date_issue_examples"] = examples.reset_index(drop=True)

    return report


def read_uploaded_file(uploaded_file) -> pd.DataFrame:
    """
    Read uploaded CSV or XLSX into a dataframe.
    """
    if uploaded_file is None:
        return pd.DataFrame()

    file_name = uploaded_file.name.lower()

    if file_name.endswith(".xlsx"):
        excel_file = pd.ExcelFile(uploaded_file)
        sheet_names = excel_file.sheet_names

        if len(sheet_names) > 1:
            sheet = st.selectbox("Select a sheet:", sheet_names)
        else:
            sheet = sheet_names[0]

        return pd.read_excel(excel_file, sheet_name=sheet)

    if file_name.endswith(".csv"):
        chunk_list = []
        for chunk in pd.read_csv(uploaded_file, chunksize=5000):
            chunk_list.append(chunk)
        return pd.concat(chunk_list, ignore_index=True)

    return pd.DataFrame()
