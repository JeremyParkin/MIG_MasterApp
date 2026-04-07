from __future__ import annotations

import re
import pandas as pd
import streamlit as st

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
        df["Impressions"] = pd.to_numeric(
            df["Impressions"], errors="coerce"
        ).fillna(0).astype("Int64")

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