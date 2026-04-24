# 1-Getting_Started.py

import re
import warnings
import pandas as pd
import streamlit as st
from ui.page_help import set_page_help_context
from utils.formatting import format_number
from utils.io import (
    build_upload_quality_report,
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
            st.session_state.upload_quality_report = build_upload_quality_report(
                st.session_state.df_untouched,
                st.session_state.df_traditional,
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
    conf_col, button_col = st.columns([4,1])
    with conf_col:
        st.success(f"File uploaded: {uploaded_filename}" if uploaded_filename else "File uploaded.")
    with button_col:
        if st.button("Start Over?"):
            clear_all_session_state()
            st.rerun()

    upload_quality_report = st.session_state.get("upload_quality_report") or {}
    upload_warnings = upload_quality_report.get("warnings", [])
    if upload_warnings:
        date_issue_indices = upload_quality_report.get("date_issue_indices") or []
        media_type_issue_indices = upload_quality_report.get("media_type_issue_indices") or []
        for warning in upload_warnings:
            if warning.get("title") == "Some date values could not be parsed" and 0 < len(date_issue_indices) <= 5:
                warning_col, action_col = st.columns([7, 1], gap="small", vertical_alignment="center")
                with warning_col:
                    st.warning(warning.get("message", "Some uploaded values could not be normalized cleanly."))
                with action_col:
                    drop_label = f"Drop {len(date_issue_indices)} row{'' if len(date_issue_indices) == 1 else 's'}"
                    if st.button(drop_label, key="drop_invalid_upload_date_rows", type="secondary", use_container_width=True):
                        st.session_state.df_traditional = st.session_state.df_traditional.drop(index=date_issue_indices, errors="ignore")
                        st.session_state.df_untouched = st.session_state.df_untouched.drop(index=date_issue_indices, errors="ignore")
                        st.session_state.upload_quality_report = build_upload_quality_report(
                            st.session_state.df_untouched,
                            st.session_state.df_traditional,
                        )
                        st.success(
                            f"Dropped {len(date_issue_indices)} uploaded row"
                            f"{'' if len(date_issue_indices) == 1 else 's'} with invalid date/time values."
                        )
                        st.rerun()
            elif warning.get("title") == "Some rows are missing media type" and 0 < len(media_type_issue_indices) <= 5:
                warning_col, action_col = st.columns([7, 1], gap="small", vertical_alignment="center")
                with warning_col:
                    st.warning(warning.get("message", "Some uploaded values could not be normalized cleanly."))
                with action_col:
                    drop_label = f"Drop {len(media_type_issue_indices)} row{'' if len(media_type_issue_indices) == 1 else 's'}"
                    if st.button(drop_label, key="drop_missing_media_type_rows", type="secondary", use_container_width=True):
                        st.session_state.df_traditional = st.session_state.df_traditional.drop(index=media_type_issue_indices, errors="ignore")
                        st.session_state.df_untouched = st.session_state.df_untouched.drop(index=media_type_issue_indices, errors="ignore")
                        st.session_state.upload_quality_report = build_upload_quality_report(
                            st.session_state.df_untouched,
                            st.session_state.df_traditional,
                        )
                        st.success(
                            f"Dropped {len(media_type_issue_indices)} uploaded row"
                            f"{'' if len(media_type_issue_indices) == 1 else 's'} with missing media type values."
                        )
                        st.rerun()
            else:
                st.warning(warning.get("message", "Some uploaded values could not be normalized cleanly."))

        date_issue_examples = upload_quality_report.get("date_issue_examples")
        if isinstance(date_issue_examples, pd.DataFrame) and not date_issue_examples.empty:
            with st.expander("Review example rows with upload date issues", expanded=False):
                st.caption("These source row numbers refer to the uploaded file and can help you find the problematic values.")
                st.dataframe(date_issue_examples, use_container_width=True, hide_index=True)
        media_type_issue_examples = upload_quality_report.get("media_type_issue_examples")
        if isinstance(media_type_issue_examples, pd.DataFrame) and not media_type_issue_examples.empty:
            with st.expander("Review example rows with missing media type", expanded=False):
                st.caption("These source row numbers refer to the uploaded file and can help you find the problematic values.")
                st.dataframe(media_type_issue_examples, use_container_width=True, hide_index=True)

    df_display = st.session_state.df_traditional.copy()
    impressions = df_display["Impressions"].sum() if "Impressions" in df_display.columns else 0

    st.divider()

    header_col, metric_col1, metric_col2 = st.columns([3, 1, 1], gap="medium", vertical_alignment="bottom")
    with header_col:
        st.header("Initial Stats")
    with metric_col1:
        st.metric(label="Mentions", value="{:,}".format(len(df_display)))
    with metric_col2:
        st.metric(label="Impressions", value=format_number(impressions))

    st.divider()

    col1, col2, col3 = st.columns(3, gap="small")

    with col1:
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

    with col2:
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

    with col3:
        st.subheader("Media Type")
        if "Type" in df_display.columns:
            media_type_df = (
                df_display["Type"]
                .fillna("")
                .astype(str)
                .str.strip()
                .loc[lambda s: s != ""]
                .value_counts()
                .rename_axis("Type")
                .reset_index(name="count")
            )
            if not media_type_df.empty:
                st.dataframe(
                    media_type_df,
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info("No non-blank media types available.")
        else:
            st.info("No media type column available.")

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
