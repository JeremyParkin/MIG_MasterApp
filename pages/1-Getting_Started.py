# 1-Getting_Started.py

import re
import warnings
import pandas as pd
import streamlit as st
from ui.page_help import set_page_help_context
from utils.formatting import format_number
from utils.io import (
    detect_original_ave_col,
    normalize_uploaded_dataframe,
    read_uploaded_file,
)
from utils.dataframe_helpers import top_x_by_mentions
from utils.session import init_getting_started_state, clear_all_session_state
from ui.charts import build_time_series_area_chart

warnings.filterwarnings("ignore")

st.title("Getting Started")
st.caption("Upload an Agility export, normalize the file structure, and set the client/reporting context for the rest of the workflow.")
set_page_help_context(st.session_state, "Getting Started")

init_getting_started_state()

# ----------------------------
# Upload step
# ----------------------------
if not st.session_state.upload_step:
    uploaded_file = st.file_uploader(
        label="Upload your CSV or XLSX*",
        type=["csv", "xlsx"],
        accept_multiple_files=False,
        help="Use CSV files exported from the Agility Platform or XLSX files produced by this app.",
    )

    if uploaded_file is not None:
        st.session_state.df_untouched = read_uploaded_file(uploaded_file)
        st.session_state.uploaded_filename = uploaded_file.name

    client = st.text_input(
        "Client organization name*",
        placeholder="eg. Air Canada",
        key="client",
        help="Required to build export file name.",
    )
    period = st.text_input(
        "Reporting period or focus*",
        placeholder="eg. March 2022",
        key="period",
        help="Required to build export file name.",
    )

    submitted = st.button("Submit", type="primary")

    if submitted and (client == "" or period == "" or uploaded_file is None):
        st.error("Missing required form inputs above.")

    elif submitted:
        with st.spinner("Converting file format."):
            st.session_state.original_ave_col = detect_original_ave_col(
                st.session_state.df_untouched
            )

            st.session_state.df_traditional = normalize_uploaded_dataframe(
                st.session_state.df_untouched
            )

            if "AVE" in st.session_state.df_traditional.columns:
                st.session_state.ave_col = "AVE"

            category_columns = [
                "Sentiment",
                "Continent",
                "Country",
                "Prov/State",
                "City",
                "Language",
            ]
            for column in category_columns:
                if column in st.session_state.df_traditional.columns:
                    st.session_state.df_traditional[column] = (
                        st.session_state.df_traditional[column].astype("category")
                    )

            st.session_state.export_name = f"{client} - {period}"
            st.session_state.client_name = client
            st.session_state.upload_step = True
            st.rerun()


# ----------------------------
# Post-upload view
# ----------------------------
if st.session_state.upload_step:
    uploaded_filename = str(st.session_state.get("uploaded_filename", "") or "").strip()
    st.success(f"File uploaded: {uploaded_filename}" if uploaded_filename else "File uploaded.")

    if st.button("Start Over?"):
        clear_all_session_state()
        st.rerun()

    st.header("Initial Stats")

    df_display = st.session_state.df_traditional.copy()

    col1, col2, col3 = st.columns(3, gap="medium")

    with col1:
        st.metric(label="Mentions", value="{:,}".format(len(df_display)))

        impressions = df_display["Impressions"].sum() if "Impressions" in df_display.columns else 0
        st.metric(label="Impressions", value=format_number(impressions))

        if "Type" in df_display.columns:
            st.write(df_display["Type"].value_counts())
        else:
            st.info("No media type column available.")

    with col2:
        st.subheader("Top Authors")
        if "Author" in df_display.columns:
            authors_df = df_display.copy()
            authors_df["Author"] = authors_df["Author"].fillna("").astype(str).str.strip()
            authors_df = authors_df[authors_df["Author"] != ""].copy()

            if not authors_df.empty:
                original_top_authors = top_x_by_mentions(authors_df, "Author")
                # st.write(original_top_authors)
                st.dataframe(
                    original_top_authors,
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info("No non-blank authors available.")
        else:
            st.info("No Author column available.")

    with col3:
        st.subheader("Top Outlets")
        if "Outlet" in df_display.columns:
            outlets_df = df_display.copy()
            outlets_df["Outlet"] = outlets_df["Outlet"].fillna("").astype(str).str.strip()
            outlets_df = outlets_df[outlets_df["Outlet"] != ""].copy()

            if not outlets_df.empty:
                original_top_outlets = top_x_by_mentions(outlets_df, "Outlet")
                st.dataframe(
                    original_top_outlets,
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info("No non-blank outlets available.")
        else:
            st.info("No Outlet column available.")

    df_trend = df_display.copy()

    if "Date" in df_trend.columns:
        df_trend["Date"] = pd.to_datetime(df_trend["Date"], errors="coerce").dt.date
    else:
        df_trend["Date"] = pd.NaT

    if df_trend["Date"].notna().any():
        summary_stats = df_trend.groupby("Date", dropna=True).agg(
            Mentions=("Mentions", "sum"),
            Impressions=("Impressions", "sum") if "Impressions" in df_trend.columns else ("Date", "size"),
        ).reset_index()

        trend_col1, trend_col2 = st.columns(2, gap="large")

        with trend_col1:
            st.subheader("Mention Trend")
            mention_chart = build_time_series_area_chart(
                df=summary_stats,
                x_col="Date",
                y_col="Mentions",
                title="",
                height=250,
            )
            if mention_chart is not None:
                st.altair_chart(mention_chart, use_container_width=True)
            else:
                st.info("Trend chart unavailable in this environment.")

        with trend_col2:
            st.subheader("Impressions Trend")
            if "Impressions" in summary_stats.columns:
                impressions_chart = build_time_series_area_chart(
                    df=summary_stats,
                    x_col="Date",
                    y_col="Impressions",
                    title="",
                    height=250,
                )
                if impressions_chart is not None:
                    st.altair_chart(impressions_chart, use_container_width=True)
                else:
                    st.info("Impressions chart unavailable in this environment.")


    else:
        st.info("No date information available to display mention and impressions trends.")
