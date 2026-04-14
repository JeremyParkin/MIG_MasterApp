# download_exports.py

from __future__ import annotations

import io
from datetime import datetime
from typing import Any

import pandas as pd

from processing.author_insights import build_author_metrics
from processing.outlet_insights import build_outlet_metrics


# ---------- Core helpers ----------

def rename_ave(df: pd.DataFrame, original_ave_col: str | None = None) -> pd.DataFrame:
    """Restore internal AVE column to original uploaded AVE column name for export."""
    export_ave_name = original_ave_col or "AVE"

    if "AVE" in df.columns:
        return df.rename(columns={"AVE": export_ave_name})
    return df


def explode_tags(df: pd.DataFrame) -> pd.DataFrame:
    """Explode comma-separated platform Tags to one-hot columns, ignoring blanks."""
    if df is None or df.empty or "Tags" not in df.columns:
        return df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame()

    out = df.copy()

    tags = out["Tags"].fillna("").astype(str).str.strip()
    tags = tags.replace({"nan": "", "None": ""})
    out["Tags"] = tags

    dummies = out["Tags"].str.get_dummies(sep=",")
    dummies.columns = [c.strip() for c in dummies.columns]

    dummies = dummies.loc[:, [c for c in dummies.columns if c and c.lower() != "nan"]]

    if not dummies.empty:
        dummies = dummies.astype("category")
        out = out.join(dummies, how="left", rsuffix=" (tag)")

    return out


def build_authors_export_table(
    df: pd.DataFrame,
    existing_assignments: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Rebuild author summary from current df_traditional and preserve any assigned outlets.
    Excludes blank author names.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["Author", "Outlet", "Mentions", "Impressions"])

    available_cols = [c for c in ["Author", "Mentions", "Impressions"] if c in df.columns]
    if "Author" not in available_cols:
        return pd.DataFrame(columns=["Author", "Outlet", "Mentions", "Impressions"])

    working = df[available_cols].copy()

    if "Mentions" not in working.columns:
        working["Mentions"] = 1
    if "Impressions" not in working.columns:
        working["Impressions"] = 0

    working["Author"] = working["Author"].fillna("").astype(str).str.strip()
    working = working[working["Author"] != ""].copy()

    rebuilt = (
        working.groupby("Author", as_index=False)[["Mentions", "Impressions"]]
        .sum()
    )

    if existing_assignments is not None and len(existing_assignments) > 0 and "Outlet" in existing_assignments.columns:
        assignment_map = (
            existing_assignments[["Author", "Outlet"]]
            .copy()
            .fillna("")
        )
        assignment_map["Author"] = assignment_map["Author"].fillna("").astype(str).str.strip()
        assignment_map = assignment_map[assignment_map["Author"] != ""].copy()
        assignment_map = assignment_map.drop_duplicates(subset=["Author"], keep="last")

        rebuilt = rebuilt.merge(assignment_map, on="Author", how="left")
        rebuilt["Outlet"] = rebuilt["Outlet"].fillna("")
    else:
        rebuilt.insert(loc=1, column="Outlet", value="")

    return rebuilt[["Author", "Outlet", "Mentions", "Impressions"]].copy()


def build_author_insights_export_table(session_state) -> pd.DataFrame:
    df_traditional = session_state.get("df_traditional", pd.DataFrame()).copy()
    auth_outlet_table = session_state.get("auth_outlet_table", pd.DataFrame()).copy()
    auth_outlet_table = auth_outlet_table if isinstance(auth_outlet_table, pd.DataFrame) and not auth_outlet_table.empty else None

    summary, _story_level = build_author_metrics(df_traditional, auth_outlet_table=auth_outlet_table)
    if summary.empty:
        return pd.DataFrame(columns=["Author", "Assigned Outlet", "Mentions", "Unique Mentions", "Impressions", "Effective Reach", "Good Outlet Rate", "Coverage Themes"])

    selected_authors = [
        str(author).strip()
        for author in session_state.get("author_insights_selected_authors", [])
        if str(author).strip()
    ]
    summaries = dict(session_state.get("author_insights_summaries", {}))

    if selected_authors:
        export_df = summary[summary["Author"].isin(selected_authors)].copy()
        export_df["SortOrder"] = export_df["Author"].map({name: idx for idx, name in enumerate(selected_authors)})
        export_df = export_df.sort_values("SortOrder").drop(columns=["SortOrder"])
    else:
        export_df = summary.sort_values(["Mention_Total", "Impressions", "Unique_Stories"], ascending=False).copy()

    export_df["Coverage Themes"] = export_df["Author"].map(lambda author: summaries.get(author, ""))
    export_df["Good Outlet Rate"] = (
        export_df["Good_Outlet_Stories"] / export_df["Unique_Stories"].replace(0, pd.NA)
    ).fillna(0.0) * 100

    return export_df[[
        "Author",
        "Assigned Outlet",
        "Mention_Total",
        "Unique_Stories",
        "Impressions",
        "Effective_Reach",
        "Good Outlet Rate",
        "Coverage Themes",
    ]].rename(
        columns={
            "Mention_Total": "Mentions",
            "Unique_Stories": "Unique Mentions",
            "Effective_Reach": "Effective Reach",
        }
    ).copy()


def build_outlets_export_table(session_state) -> pd.DataFrame:
    df_traditional = session_state.get("df_traditional", pd.DataFrame()).copy()
    summary, _story_level = build_outlet_metrics(df_traditional)
    if summary.empty:
        return pd.DataFrame(columns=["Outlet", "Media Types", "Mentions", "Unique Mentions", "Impressions", "Effective Reach", "Good Outlet Rate", "Coverage Themes"])

    selected_outlets = [
        str(outlet).strip()
        for outlet in session_state.get("outlet_insights_selected_outlets", [])
        if str(outlet).strip()
    ]
    summaries = dict(session_state.get("outlet_insights_summaries", {}))

    if selected_outlets:
        export_df = summary[summary["Outlet"].isin(selected_outlets)].copy()
        export_df["SortOrder"] = export_df["Outlet"].map({name: idx for idx, name in enumerate(selected_outlets)})
        export_df = export_df.sort_values("SortOrder").drop(columns=["SortOrder"])
    else:
        export_df = summary.sort_values(["Mention_Total", "Impressions", "Unique_Mentions"], ascending=False).copy()

    export_df["Coverage Themes"] = export_df["Outlet"].map(lambda outlet: summaries.get(outlet, ""))

    return export_df[[
        "Outlet",
        "Top_Types",
        "Mention_Total",
        "Unique_Mentions",
        "Impressions",
        "Effective_Reach",
        "Good_Outlet_Rate",
        "Coverage Themes",
    ]].rename(
        columns={
            "Top_Types": "Media Types",
            "Mention_Total": "Mentions",
            "Unique_Mentions": "Unique Mentions",
            "Effective_Reach": "Effective Reach",
            "Good_Outlet_Rate": "Good Outlet Rate",
        }
    ).copy()


def get_currency_symbol(original_ave_col: str | None) -> str:
    """Infer currency symbol from original uploaded AVE column name. Fallback to $."""
    original_ave_col = str(original_ave_col or "AVE")

    if "(EUR)" in original_ave_col:
        return "€"
    if "(GBP)" in original_ave_col:
        return "£"
    if "(JPY)" in original_ave_col:
        return "¥"

    return "$"


def apply_sheet_column_formats(
    worksheet,
    df: pd.DataFrame,
    number_format,
    currency_format,
    original_ave_col: str | None = None,
):
    """Apply column widths and formats by column name."""
    if df is None or df.empty:
        return

    original_ave_col = str(original_ave_col or "AVE")

    column_rules = {
        "Group ID": {"width": 10},
        "Prime Example": {"width": 10},
        "Date": {"width": 18},
        "Author": {"width": 24},
        "Outlet": {"width": 28},
        "Headline": {"width": 40},
        "Title": {"width": 40},
        "Type": {"width": 14},
        "Media Type": {"width": 14},
        "URL": {"width": 30},
        "Example URL": {"width": 30},
        "Snippet": {"width": 55},
        "Coverage Snippet": {"width": 55},
        "Summary": {"width": 45},
        "Content": {"width": 70},
        "Country": {"width": 14},
        "Prov/State": {"width": 14},
        "Language": {"width": 12},
        "Sentiment": {"width": 12},
        "Assigned Sentiment": {"width": 16},
        "AI Sentiment": {"width": 16},
        "AI Sentiment Confidence": {"width": 12, "format": number_format},
        "AI Sentiment Rationale": {"width": 45},
        "Hybrid Sentiment": {"width": 16},
        "Hybrid Sentiment Confidence": {"width": 12, "format": number_format},
        "AI Tags": {"width": 30},
        "AI Tag Rationale": {"width": 45},
        "Tag_Processed": {"width": 12},
        "Mentions": {"width": 12, "format": number_format},
        "Impressions": {"width": 14, "format": number_format},
        "Effective Reach": {"width": 16, "format": number_format},
        "AVE": {"width": 14, "format": currency_format},
        original_ave_col: {"width": 14, "format": currency_format},
        "Example Outlet": {"width": 20},
        "Chart Callout": {"width": 40},
        "Top Story Summary": {"width": 55},
        "Entity Sentiment": {"width": 45},
        "Unique Mentions": {"width": 14, "format": number_format},
        "Effective Reach": {"width": 16, "format": number_format},
        "Good Outlet Rate": {"width": 14},
        "Coverage Themes": {"width": 65},
        "Media Types": {"width": 24},
        "Field": {"width": 32},
        "Value": {"width": 40},
    }

    worksheet.set_default_row(22)

    for col_name in df.columns:
        col_idx = df.columns.get_loc(col_name)
        rule = column_rules.get(col_name)
        if rule:
            worksheet.set_column(
                col_idx,
                col_idx,
                rule.get("width"),
                rule.get("format"),
            )

    worksheet.freeze_panes(1, 0)


# ---------- AI scope / merge logic ----------

def tagging_exists(session_state) -> bool:
    df = session_state.get("df_tagging_rows", pd.DataFrame())
    return isinstance(df, pd.DataFrame) and not df.empty


def sentiment_exists(session_state) -> bool:
    df = session_state.get("df_sentiment_rows", pd.DataFrame())
    return isinstance(df, pd.DataFrame) and not df.empty


def tagging_full_scope(session_state) -> bool:
    return (
        session_state.get("tagging_config_step", False)
        and session_state.get("tagging_sample_mode") == "full"
    )


def sentiment_full_scope(session_state) -> bool:
    return (
        session_state.get("sentiment_config_step", False)
        and session_state.get("sentiment_sample_mode") == "full"
    )


def tagging_processed_complete_enough(session_state) -> bool:
    df_unique = session_state.get("df_tagging_unique", pd.DataFrame())
    if not isinstance(df_unique, pd.DataFrame) or df_unique.empty:
        return False
    if "Tag_Processed" not in df_unique.columns:
        return False
    return bool(df_unique["Tag_Processed"].fillna(False).any())


def sentiment_processed_complete_enough(session_state) -> bool:
    df_unique = session_state.get("df_sentiment_unique", pd.DataFrame())
    if not isinstance(df_unique, pd.DataFrame) or df_unique.empty:
        return False

    has_ai = "AI Sentiment" in df_unique.columns and bool(df_unique["AI Sentiment"].notna().any())
    has_assigned = "Assigned Sentiment" in df_unique.columns and bool(df_unique["Assigned Sentiment"].notna().any())

    return has_ai or has_assigned


def should_merge_tagging_into_clean_trad(session_state) -> bool:
    return tagging_exists(session_state) and tagging_full_scope(session_state) and tagging_processed_complete_enough(session_state)


def should_merge_sentiment_into_clean_trad(session_state) -> bool:
    return sentiment_exists(session_state) and sentiment_full_scope(session_state) and sentiment_processed_complete_enough(session_state)


def build_tagging_sample_export(session_state) -> pd.DataFrame:
    df = session_state.get("df_tagging_rows", pd.DataFrame())
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()

    out = df.copy()

    priority_cols = [
        "Group ID",
        "Prime Example",
        "Date",
        "Headline",
        "Outlet",
        "Type",
        "Mentions",
        "Impressions",
        "Effective Reach",
        "AI Tags",
        "AI Tag Rationale",
        "Tag_Processed",
    ]

    existing_priority = [c for c in priority_cols if c in out.columns]
    remaining_cols = [c for c in out.columns if c not in existing_priority]

    return out[existing_priority + remaining_cols].copy()


def build_sentiment_sample_export(session_state) -> pd.DataFrame:
    df = session_state.get("df_sentiment_rows", pd.DataFrame())
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()

    out = df.copy()

    priority_cols = [
        "Group ID",
        "Prime Example",
        "Date",
        "Headline",
        "Outlet",
        "Type",
        "Mentions",
        "Impressions",
        "Effective Reach",
        "Assigned Sentiment",
        "AI Sentiment",
        "AI Sentiment Confidence",
        "AI Sentiment Rationale",
        "Hybrid Sentiment",
        "Hybrid Sentiment Confidence",
    ]

    existing_priority = [c for c in priority_cols if c in out.columns]
    remaining_cols = [c for c in out.columns if c not in existing_priority]

    return out[existing_priority + remaining_cols].copy()


def merge_full_scope_ai_columns_into_clean_trad(session_state, traditional: pd.DataFrame) -> pd.DataFrame:
    out = traditional.copy()

    if should_merge_tagging_into_clean_trad(session_state):
        tag_rows = session_state.get("df_tagging_rows", pd.DataFrame()).copy()
        tag_cols = [c for c in ["Group ID", "AI Tags", "AI Tag Rationale", "Tag_Processed"] if c in tag_rows.columns]
        binary_tag_cols = [c for c in tag_rows.columns if str(c).startswith("AI Tag: ")]
        tag_cols = tag_cols + binary_tag_cols

        if "Group ID" in tag_cols:
            tag_map = tag_rows[tag_cols].drop_duplicates(subset=["Group ID"], keep="last")
            cols_to_drop = [c for c in tag_cols if c != "Group ID" and c in out.columns]
            out = out.drop(columns=cols_to_drop, errors="ignore")
            out = out.merge(tag_map, on="Group ID", how="left")

    if should_merge_sentiment_into_clean_trad(session_state):
        sent_rows = session_state.get("df_sentiment_rows", pd.DataFrame()).copy()
        sent_cols = [
            c for c in [
                "Group ID",
                "Assigned Sentiment",
                "AI Sentiment",
                "AI Sentiment Confidence",
                "AI Sentiment Rationale",
                "Review AI Sentiment",
                "Review AI Confidence",
                "Review AI Rationale",
            ] if c in sent_rows.columns
        ]
        if "Group ID" in sent_cols:
            sent_map = sent_rows[sent_cols].drop_duplicates(subset=["Group ID"], keep="last")
            cols_to_drop = [c for c in sent_cols if c != "Group ID" and c in out.columns]
            out = out.drop(columns=cols_to_drop, errors="ignore")
            out = out.merge(sent_map, on="Group ID", how="left")

    return out


# ---------- Metadata sheet ----------

def build_export_metadata_sheet(session_state) -> pd.DataFrame:
    original_rows = len(session_state.get("df_untouched", pd.DataFrame()))
    clean_trad_rows = len(session_state.get("df_traditional", pd.DataFrame()))
    clean_social_rows = len(session_state.get("df_social", pd.DataFrame()))
    dupes_rows = len(session_state.get("df_dupes", pd.DataFrame()))
    row_check = clean_trad_rows + clean_social_rows + dupes_rows

    tag_rows = session_state.get("df_tagging_rows", pd.DataFrame())
    tag_unique = session_state.get("df_tagging_unique", pd.DataFrame())

    sent_rows = session_state.get("df_sentiment_rows", pd.DataFrame())
    sent_unique = session_state.get("df_sentiment_unique", pd.DataFrame())

    tag_processed_groups = 0
    if isinstance(tag_unique, pd.DataFrame) and not tag_unique.empty and "Tag_Processed" in tag_unique.columns:
        tag_processed_groups = int(tag_unique["Tag_Processed"].fillna(False).sum())

    sent_processed_groups = 0
    if isinstance(sent_unique, pd.DataFrame) and not sent_unique.empty:
        has_ai = "AI Sentiment" in sent_unique.columns and sent_unique["AI Sentiment"].notna()
        has_assigned = "Assigned Sentiment" in sent_unique.columns and sent_unique["Assigned Sentiment"].notna()
        sent_processed_groups = int((has_ai if isinstance(has_ai, pd.Series) else False) | (has_assigned if isinstance(has_assigned, pd.Series) else False)).sum() if isinstance(has_ai, pd.Series) and isinstance(has_assigned, pd.Series) else int(has_ai.sum() if isinstance(has_ai, pd.Series) else 0)

    rows = [
        ("Export Timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        ("Export Name", session_state.get("export_name", "")),
        ("Client Name", session_state.get("client_name", "")),
        ("Original Rows", original_rows),
        ("Clean Traditional Rows", clean_trad_rows),
        ("Clean Social Rows", clean_social_rows),
        ("Deleted Duplicate Rows", dupes_rows),
        ("Row Check Total", row_check),
        ("Row Check Matches Original", "Yes" if row_check == original_rows else "No"),
        ("Canonical Group ID Present", "Yes" if "Group ID" in session_state.get("df_traditional", pd.DataFrame()).columns else "No"),
        ("Prime Example Present", "Yes" if "Prime Example" in session_state.get("df_traditional", pd.DataFrame()).columns else "No"),

        ("Tagging Run", "Yes" if tagging_exists(session_state) else "No"),
        ("Tagging Scope", session_state.get("tagging_sample_mode", "")),
        ("Tagging Sample Size Used", session_state.get("tagging_sample_size", "")),
        ("Tagging Processed Groups", tag_processed_groups),
        ("Tagging Rows in Working Set", len(tag_rows) if isinstance(tag_rows, pd.DataFrame) else 0),
        ("Tagging Merged Into Clean Trad", "Yes" if should_merge_tagging_into_clean_trad(session_state) else "No"),

        ("Sentiment Run", "Yes" if sentiment_exists(session_state) else "No"),
        ("Sentiment Scope", session_state.get("sentiment_sample_mode", "")),
        ("Sentiment Sample Size Used", session_state.get("sentiment_sample_size", "")),
        ("Sentiment Processed Groups", sent_processed_groups),
        ("Sentiment Rows in Working Set", len(sent_rows) if isinstance(sent_rows, pd.DataFrame) else 0),
        ("Sentiment Merged Into Clean Trad", "Yes" if should_merge_sentiment_into_clean_trad(session_state) else "No"),
    ]

    return pd.DataFrame(rows, columns=["Field", "Value"])


# ---------- Sentiment helpers ----------

def add_final_sentiment_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame()

    out = df.copy()

    base_ai_sent = out.get("AI Sentiment", pd.Series(index=out.index, dtype="object"))
    base_ai_conf = out.get("AI Sentiment Confidence", pd.Series(index=out.index, dtype="object"))
    base_ai_rat = out.get("AI Sentiment Rationale", pd.Series(index=out.index, dtype="object"))

    review_ai_sent = out.get("Review AI Sentiment", pd.Series(index=out.index, dtype="object"))
    review_ai_conf = out.get("Review AI Confidence", pd.Series(index=out.index, dtype="object"))
    review_ai_rat = out.get("Review AI Rationale", pd.Series(index=out.index, dtype="object"))

    review_sent_clean = review_ai_sent.fillna("").astype(str).str.strip()

    out["AI Sentiment"] = review_ai_sent.where(review_sent_clean != "", base_ai_sent)
    out["AI Sentiment Confidence"] = review_ai_conf.where(review_sent_clean != "", base_ai_conf)
    out["AI Sentiment Rationale"] = review_ai_rat.where(review_sent_clean != "", base_ai_rat)

    assigned = out.get("Assigned Sentiment", pd.Series(index=out.index, dtype="object"))
    ai = out.get("AI Sentiment", pd.Series(index=out.index, dtype="object"))

    assigned_clean = assigned.fillna("").astype(str).str.strip()
    ai_clean = ai.fillna("").astype(str).str.strip()

    out["Hybrid Sentiment"] = assigned_clean.where(assigned_clean != "", ai_clean)
    out["Hybrid Sentiment"] = out["Hybrid Sentiment"].replace("", pd.NA)

    if "AI Sentiment Confidence" in out.columns:
        out["Hybrid Sentiment Confidence"] = out["AI Sentiment Confidence"]
        out.loc[assigned_clean != "", "Hybrid Sentiment Confidence"] = pd.NA

    return out


# ---------- Workbook builder ----------

def build_clean_workbook_bytes(session_state) -> bytes:
    traditional = session_state.get("df_traditional", pd.DataFrame()).copy()
    social = session_state.get("df_social", pd.DataFrame()).copy()
    top_stories = session_state.get("added_df", pd.DataFrame()).copy()
    dupes = session_state.get("df_dupes", pd.DataFrame()).copy()
    raw = session_state.get("df_untouched", pd.DataFrame()).copy()

    for df in [traditional, social, dupes]:
        if isinstance(df, pd.DataFrame) and "Published Date" in df.columns:
            df.drop(columns=["Published Date"], inplace=True)

    original_ave_col = session_state.get("original_ave_col") or "AVE"

    traditional = merge_full_scope_ai_columns_into_clean_trad(session_state, traditional)
    traditional = add_final_sentiment_columns(traditional)
    traditional = explode_tags(traditional)

    social = explode_tags(social)

    output = io.BytesIO()

    with pd.ExcelWriter(output, engine="xlsxwriter", datetime_format="yyyy-mm-dd") as writer:
        workbook = writer.book
        cleaned_exports: list[tuple[str, pd.DataFrame, Any]] = []

        number_format = workbook.add_format({"num_format": "#,##0"})
        currency_symbol = get_currency_symbol(original_ave_col)
        currency_format = workbook.add_format({"num_format": f"{currency_symbol}#,##0.00"})

        # CLEAN TRAD
        if len(traditional) > 0:
            trad_export = rename_ave(traditional.copy(), original_ave_col=original_ave_col)
            if "Impressions" in trad_export.columns:
                trad_export = trad_export.sort_values(by=["Impressions"], ascending=False)
            trad_export.to_excel(writer, sheet_name="CLEAN TRAD", startrow=1, header=False, index=False)
            ws = writer.sheets["CLEAN TRAD"]
            ws.set_tab_color("black")
            cleaned_exports.append(("CLEAN TRAD", trad_export, ws))

        # CLEAN SOCIAL
        if len(social) > 0:
            social_export = rename_ave(social.copy(), original_ave_col=original_ave_col)
            if "Impressions" in social_export.columns:
                social_export = social_export.sort_values(by=["Impressions"], ascending=False)
            social_export.to_excel(writer, sheet_name="CLEAN SOCIAL", startrow=1, header=False, index=False)
            ws = writer.sheets["CLEAN SOCIAL"]
            ws.set_tab_color("black")
            cleaned_exports.append(("CLEAN SOCIAL", social_export, ws))

        # TAGGING SAMPLE
        if (
            tagging_exists(session_state)
            and not should_merge_tagging_into_clean_trad(session_state)
            and tagging_processed_complete_enough(session_state)
        ):
            tagging_export = rename_ave(build_tagging_sample_export(session_state), original_ave_col=original_ave_col)
            if not tagging_export.empty:
                tagging_export.to_excel(writer, sheet_name="TAGGING SAMPLE", header=True, index=False)
                ws = writer.sheets["TAGGING SAMPLE"]
                ws.set_tab_color("#7f8c8d")
                cleaned_exports.append(("TAGGING SAMPLE", tagging_export, ws))

        # SENTIMENT SAMPLE
        if (
            sentiment_exists(session_state)
            and not should_merge_sentiment_into_clean_trad(session_state)
            and sentiment_processed_complete_enough(session_state)
        ):
            sentiment_export = add_final_sentiment_columns(
                rename_ave(build_sentiment_sample_export(session_state), original_ave_col=original_ave_col)
            )
            if not sentiment_export.empty:
                sentiment_export.to_excel(writer, sheet_name="SENTIMENT SAMPLE", header=True, index=False)
                ws = writer.sheets["SENTIMENT SAMPLE"]
                ws.set_tab_color("#7f8c8d")
                cleaned_exports.append(("SENTIMENT SAMPLE", sentiment_export, ws))

        # AUTHORS
        authors = build_author_insights_export_table(session_state)
        if authors.empty:
            authors = build_authors_export_table(
                session_state.get("df_traditional", pd.DataFrame()).copy(),
                existing_assignments=session_state.get("auth_outlet_table", pd.DataFrame()).copy()
                if len(session_state.get("auth_outlet_table", pd.DataFrame())) > 0 else None,
            )
        if len(authors) > 0:
            selected_authors = [
                str(author).strip()
                for author in session_state.get("author_insights_selected_authors", [])
                if str(author).strip()
            ]
            if not selected_authors:
                authors = authors.sort_values(by=["Mentions", "Impressions"], ascending=False).copy()
            authors.to_excel(writer, sheet_name="Authors", header=True, index=False)
            ws = writer.sheets["Authors"]
            ws.set_tab_color("green")
            cleaned_exports.append(("Authors", authors, ws))

        # OUTLETS
        outlets = build_outlets_export_table(session_state)
        if len(outlets) > 0:
            outlets.to_excel(writer, sheet_name="Outlets", header=True, index=False)
            ws = writer.sheets["Outlets"]
            ws.set_tab_color("green")
            cleaned_exports.append(("Outlets", outlets, ws))

        # TOP STORIES
        if len(top_stories) > 0:
            top_stories_export = top_stories.copy()
            if all(c in top_stories_export.columns for c in ["Mentions", "Impressions"]):
                top_stories_export = top_stories_export.sort_values(by=["Mentions", "Impressions"], ascending=False)

            desired_top_story_columns = [
                "Headline",
                "Date",
                "Mentions",
                "Impressions",
                "Example Outlet",
                "Example URL",
                "Chart Callout",
                "Top Story Summary",
                "Entity Sentiment",
            ]
            existing_top_story_columns = [c for c in desired_top_story_columns if c in top_stories_export.columns]
            top_stories_export = top_stories_export[existing_top_story_columns].copy()

            top_stories_export.to_excel(writer, sheet_name="Top Stories", header=True, index=False)
            ws = writer.sheets["Top Stories"]
            ws.set_tab_color("green")
            cleaned_exports.append(("Top Stories", top_stories_export, ws))

        # DLTD DUPES
        if len(dupes) > 0:
            dupes_export = rename_ave(dupes.copy(), original_ave_col=original_ave_col)
            dupes_export.to_excel(writer, sheet_name="DLTD DUPES", header=True, index=False)
            ws = writer.sheets["DLTD DUPES"]
            ws.set_tab_color("#c26f4f")
            cleaned_exports.append(("DLTD DUPES", dupes_export, ws))

        # RAW
        raw_export = rename_ave(raw.copy(), original_ave_col=original_ave_col)
        raw_export.drop(["Mentions"], axis=1, inplace=True, errors="ignore")
        raw_export.to_excel(writer, sheet_name="RAW", header=True, index=False)
        ws = writer.sheets["RAW"]
        ws.set_tab_color("#c26f4f")
        cleaned_exports.append(("RAW", raw_export, ws))

        # EXPORT METADATA
        metadata_export = build_export_metadata_sheet(session_state)
        metadata_export.to_excel(writer, sheet_name="EXPORT METADATA", header=True, index=False)
        ws = writer.sheets["EXPORT METADATA"]
        ws.set_tab_color("#4f81bd")
        cleaned_exports.append(("EXPORT METADATA", metadata_export, ws))

        # Apply Excel table structures + formatting
        row_level_sheets = {"CLEAN TRAD", "CLEAN SOCIAL", "TAGGING SAMPLE", "SENTIMENT SAMPLE", "DLTD DUPES"}

        for sheet_name, clean_df, ws in cleaned_exports:
            if clean_df.empty:
                continue

            max_row, max_col = clean_df.shape
            column_settings = [{"header": column} for column in clean_df.columns]
            ws.add_table(0, 0, max_row, max_col - 1, {"columns": column_settings})

            apply_sheet_column_formats(
                worksheet=ws,
                df=clean_df,
                number_format=number_format,
                currency_format=currency_format,
                original_ave_col=original_ave_col,
            )

            if "Mentions" in clean_df.columns and sheet_name in row_level_sheets:
                col_idx = clean_df.columns.get_loc("Mentions")
                ws.set_column(col_idx, col_idx, None, None, {"hidden": True})

    output.seek(0)
    return output.getvalue()
# from __future__ import annotations
#
# import io
# from datetime import datetime
#
# import pandas as pd
#
#
# # ---------- Core helpers ----------
#
# def rename_ave(df: pd.DataFrame, original_ave_col: str | None = None) -> pd.DataFrame:
#     """Restore internal AVE column to original uploaded AVE column name for export."""
#     export_ave_name = original_ave_col or "AVE"
#
#     if "AVE" in df.columns:
#         return df.rename(columns={"AVE": export_ave_name})
#     return df
#
#
# def explode_tags(df: pd.DataFrame) -> pd.DataFrame:
#     """Explode comma-separated Tags to one-hot columns, ignoring blanks."""
#     if df is None or df.empty or "Tags" not in df.columns:
#         return df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame()
#
#     out = df.copy()
#
#     tags = out["Tags"].fillna("").astype(str).str.strip()
#     tags = tags.replace({"nan": "", "None": ""})
#     out["Tags"] = tags
#
#     dummies = out["Tags"].str.get_dummies(sep=",")
#     dummies.columns = [c.strip() for c in dummies.columns]
#
#     # drop blank / garbage columns if they somehow appear
#     dummies = dummies.loc[:, [c for c in dummies.columns if c and c.lower() != "nan"]]
#
#     if not dummies.empty:
#         dummies = dummies.astype("category")
#         out = out.join(dummies, how="left", rsuffix=" (tag)")
#
#     return out
#
#
#
#
# def build_authors_export_table(
#     df: pd.DataFrame,
#     existing_assignments: pd.DataFrame | None = None,
# ) -> pd.DataFrame:
#     """
#     Rebuild author summary from current df_traditional and preserve any assigned outlets.
#     Excludes blank author names.
#     """
#     if df is None or df.empty:
#         return pd.DataFrame(columns=["Author", "Outlet", "Mentions", "Impressions"])
#
#     available_cols = [c for c in ["Author", "Mentions", "Impressions"] if c in df.columns]
#     if "Author" not in available_cols:
#         return pd.DataFrame(columns=["Author", "Outlet", "Mentions", "Impressions"])
#
#     working = df[available_cols].copy()
#
#     if "Mentions" not in working.columns:
#         working["Mentions"] = 1
#     if "Impressions" not in working.columns:
#         working["Impressions"] = 0
#
#     working["Author"] = working["Author"].fillna("").astype(str).str.strip()
#     working = working[working["Author"] != ""].copy()
#
#     rebuilt = (
#         working.groupby("Author", as_index=False)[["Mentions", "Impressions"]]
#         .sum()
#     )
#
#     if existing_assignments is not None and len(existing_assignments) > 0 and "Outlet" in existing_assignments.columns:
#         assignment_map = (
#             existing_assignments[["Author", "Outlet"]]
#             .copy()
#             .fillna("")
#         )
#         assignment_map["Author"] = assignment_map["Author"].fillna("").astype(str).str.strip()
#         assignment_map = assignment_map[assignment_map["Author"] != ""].copy()
#         assignment_map = assignment_map.drop_duplicates(subset=["Author"], keep="last")
#
#         rebuilt = rebuilt.merge(assignment_map, on="Author", how="left")
#         rebuilt["Outlet"] = rebuilt["Outlet"].fillna("")
#     else:
#         rebuilt.insert(loc=1, column="Outlet", value="")
#
#     return rebuilt[["Author", "Outlet", "Mentions", "Impressions"]].copy()
#
#
# def get_currency_symbol(original_ave_col: str | None) -> str:
#     """Infer currency symbol from original uploaded AVE column name. Fallback to $."""
#     original_ave_col = str(original_ave_col or "AVE")
#
#     if "(EUR)" in original_ave_col:
#         return "€"
#     if "(GBP)" in original_ave_col:
#         return "£"
#     if "(JPY)" in original_ave_col:
#         return "¥"
#
#     return "$"
#
#
# def apply_sheet_column_formats(
#     worksheet,
#     df: pd.DataFrame,
#     number_format,
#     currency_format,
#     original_ave_col: str | None = None,
# ):
#     """Apply column widths and formats by column name."""
#     if df is None or df.empty:
#         return
#
#     original_ave_col = str(original_ave_col or "AVE")
#
#     column_rules = {
#         "Group ID": {"width": 10},
#         "Date": {"width": 12},
#         "Published Date": {"width": 12},
#         "Author": {"width": 24},
#         "Outlet": {"width": 28},
#         "Headline": {"width": 40},
#         "Title": {"width": 40},
#         "Type": {"width": 14},
#         "Media Type": {"width": 14},
#         "URL": {"width": 30},
#         "Example URL": {"width": 30},
#         "Snippet": {"width": 55},
#         "Coverage Snippet": {"width": 55},
#         "Summary": {"width": 45},
#         "Content": {"width": 70},
#         "Country": {"width": 14},
#         "Prov/State": {"width": 14},
#         "Language": {"width": 12},
#         "Sentiment": {"width": 12},
#         "Assigned Sentiment": {"width": 16},
#         "AI Sentiment": {"width": 16},
#         "AI Sentiment Confidence": {"width": 12, "format": number_format},
#         "AI Sentiment Rationale": {"width": 45},
#         "AI Tags": {"width": 30},
#         "AI Tag Rationale": {"width": 45},
#         "Tag_Processed": {"width": 12},
#         "Mentions": {"width": 12, "format": number_format},
#         "Impressions": {"width": 14, "format": number_format},
#         "Effective Reach": {"width": 16, "format": number_format},
#         "AVE": {"width": 14, "format": currency_format},
#         original_ave_col: {"width": 14, "format": currency_format},
#         "Example Outlet": {"width": 20},
#         "Chart Callout": {"width": 40},
#         "Top Story Summary": {"width": 55},
#         "Entity Sentiment": {"width": 45},
#         "Field": {"width": 32},
#         "Value": {"width": 40},
#     }
#
#     worksheet.set_default_row(22)
#
#     for col_name in df.columns:
#         col_idx = df.columns.get_loc(col_name)
#         rule = column_rules.get(col_name)
#         if rule:
#             worksheet.set_column(
#                 col_idx,
#                 col_idx,
#                 rule.get("width"),
#                 rule.get("format"),
#             )
#
#     worksheet.freeze_panes(1, 0)
#
#
# # ---------- AI scope / merge logic ----------
#
# def tagging_exists(session_state) -> bool:
#     df = session_state.get("df_tagging_rows", pd.DataFrame())
#     return isinstance(df, pd.DataFrame) and not df.empty
#
#
# def sentiment_exists(session_state) -> bool:
#     df = session_state.get("df_sentiment_rows", pd.DataFrame())
#     return isinstance(df, pd.DataFrame) and not df.empty
#
#
# def tagging_full_scope(session_state) -> bool:
#     return (
#         session_state.get("tagging_config_step", False)
#         and session_state.get("tagging_sample_mode") == "full"
#     )
#
#
# def sentiment_full_scope(session_state) -> bool:
#     return (
#         session_state.get("sentiment_config_step", False)
#         and session_state.get("sentiment_sample_mode") == "full"
#     )
#
#
# def tagging_processed_complete_enough(session_state) -> bool:
#     df_unique = session_state.get("df_tagging_unique", pd.DataFrame())
#     if not isinstance(df_unique, pd.DataFrame) or df_unique.empty:
#         return False
#     if "Tag_Processed" not in df_unique.columns:
#         return False
#     return bool(df_unique["Tag_Processed"].fillna(False).all())
#
#
# def sentiment_processed_complete_enough(session_state) -> bool:
#     df_unique = session_state.get("df_sentiment_unique", pd.DataFrame())
#     if not isinstance(df_unique, pd.DataFrame) or df_unique.empty:
#         return False
#     if "AI Sentiment" not in df_unique.columns:
#         return False
#     return bool(df_unique["AI Sentiment"].notna().all())
#
#
# def should_merge_tagging_into_clean_trad(session_state) -> bool:
#     return tagging_exists(session_state) and tagging_full_scope(session_state) and tagging_processed_complete_enough(session_state)
#
#
# def should_merge_sentiment_into_clean_trad(session_state) -> bool:
#     return sentiment_exists(session_state) and sentiment_full_scope(session_state) and sentiment_processed_complete_enough(session_state)
#
#
# def build_tagging_sample_export(session_state) -> pd.DataFrame:
#     df = session_state.get("df_tagging_rows", pd.DataFrame())
#     if not isinstance(df, pd.DataFrame) or df.empty:
#         return pd.DataFrame()
#
#     out = df.copy()
#
#     # Put the most important AI columns near the front if they exist
#     priority_cols = [
#         "Group ID",
#         "Date",
#         "Headline",
#         "Outlet",
#         "Type",
#         "Mentions",
#         "Impressions",
#         "Effective Reach",
#         "AI Tags",
#         "AI Tag Rationale",
#         "Tag_Processed",
#     ]
#
#     existing_priority = [c for c in priority_cols if c in out.columns]
#     remaining_cols = [c for c in out.columns if c not in existing_priority]
#
#     return out[existing_priority + remaining_cols].copy()
#
#
# def build_sentiment_sample_export(session_state) -> pd.DataFrame:
#     df = session_state.get("df_sentiment_rows", pd.DataFrame())
#     if not isinstance(df, pd.DataFrame) or df.empty:
#         return pd.DataFrame()
#
#     out = df.copy()
#
#     priority_cols = [
#         "Group ID",
#         "Date",
#         "Headline",
#         "Outlet",
#         "Type",
#         "Mentions",
#         "Impressions",
#         "Effective Reach",
#         "Assigned Sentiment",
#         "AI Sentiment",
#         "AI Sentiment Confidence",
#         "AI Sentiment Rationale",
#     ]
#
#     existing_priority = [c for c in priority_cols if c in out.columns]
#     remaining_cols = [c for c in out.columns if c not in existing_priority]
#
#     return out[existing_priority + remaining_cols].copy()
#
# # def build_sentiment_sample_export(session_state) -> pd.DataFrame:
# #     df = session_state.get("df_sentiment_rows", pd.DataFrame())
# #     if not isinstance(df, pd.DataFrame) or df.empty:
# #         return pd.DataFrame()
# #
# #     out = df.copy()
# #     desired = [
# #         "Group ID", "Date", "Headline", "Outlet", "Type", "Language",
# #         "Mentions", "Impressions", "Effective Reach",
# #         "Assigned Sentiment", "AI Sentiment", "AI Sentiment Confidence", "AI Sentiment Rationale",
# #         "URL", "Snippet",
# #     ]
# #     existing = [c for c in desired if c in out.columns]
# #     return out[existing].copy()
#
#
# def merge_full_scope_ai_columns_into_clean_trad(session_state, traditional: pd.DataFrame) -> pd.DataFrame:
#     out = traditional.copy()
#
#     if should_merge_tagging_into_clean_trad(session_state):
#         tag_rows = session_state.get("df_tagging_rows", pd.DataFrame()).copy()
#         tag_cols = [c for c in ["Group ID", "AI Tags", "AI Tag Rationale", "Tag_Processed"] if c in tag_rows.columns]
#         if "Group ID" in tag_cols:
#             tag_map = tag_rows[tag_cols].drop_duplicates(subset=["Group ID"], keep="last")
#             cols_to_drop = [c for c in tag_cols if c != "Group ID" and c in out.columns]
#             out = out.drop(columns=cols_to_drop, errors="ignore")
#             out = out.merge(tag_map, on="Group ID", how="left")
#
#     if should_merge_sentiment_into_clean_trad(session_state):
#         sent_rows = session_state.get("df_sentiment_rows", pd.DataFrame()).copy()
#         sent_cols = [c for c in ["Group ID", "Assigned Sentiment", "AI Sentiment", "AI Sentiment Confidence", "AI Sentiment Rationale"] if c in sent_rows.columns]
#         if "Group ID" in sent_cols:
#             sent_map = sent_rows[sent_cols].drop_duplicates(subset=["Group ID"], keep="last")
#             cols_to_drop = [c for c in sent_cols if c != "Group ID" and c in out.columns]
#             out = out.drop(columns=cols_to_drop, errors="ignore")
#             out = out.merge(sent_map, on="Group ID", how="left")
#
#     return out
#
#
# # ---------- Metadata sheet ----------
#
# def build_export_metadata_sheet(session_state) -> pd.DataFrame:
#     original_rows = len(session_state.get("df_untouched", pd.DataFrame()))
#     clean_trad_rows = len(session_state.get("df_traditional", pd.DataFrame()))
#     clean_social_rows = len(session_state.get("df_social", pd.DataFrame()))
#     dupes_rows = len(session_state.get("df_dupes", pd.DataFrame()))
#     row_check = clean_trad_rows + clean_social_rows + dupes_rows
#
#     tag_rows = session_state.get("df_tagging_rows", pd.DataFrame())
#     tag_unique = session_state.get("df_tagging_unique", pd.DataFrame())
#
#     sent_rows = session_state.get("df_sentiment_rows", pd.DataFrame())
#     sent_unique = session_state.get("df_sentiment_unique", pd.DataFrame())
#
#     tag_processed_groups = 0
#     if isinstance(tag_unique, pd.DataFrame) and not tag_unique.empty and "Tag_Processed" in tag_unique.columns:
#         tag_processed_groups = int(tag_unique["Tag_Processed"].fillna(False).sum())
#
#     sent_processed_groups = 0
#     if isinstance(sent_unique, pd.DataFrame) and not sent_unique.empty and "AI Sentiment" in sent_unique.columns:
#         sent_processed_groups = int(sent_unique["AI Sentiment"].notna().sum())
#
#     rows = [
#         ("Export Timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
#         ("Export Name", session_state.get("export_name", "")),
#         ("Client Name", session_state.get("client_name", "")),
#         ("Original Rows", original_rows),
#         ("Clean Traditional Rows", clean_trad_rows),
#         ("Clean Social Rows", clean_social_rows),
#         ("Deleted Duplicate Rows", dupes_rows),
#         ("Row Check Total", row_check),
#         ("Row Check Matches Original", "Yes" if row_check == original_rows else "No"),
#         ("Canonical Group ID Present", "Yes" if "Group ID" in session_state.get("df_traditional", pd.DataFrame()).columns else "No"),
#
#         ("Tagging Run", "Yes" if tagging_exists(session_state) else "No"),
#         ("Tagging Scope", session_state.get("tagging_sample_mode", "")),
#         ("Tagging Sample Size Used", session_state.get("tagging_sample_size", "")),
#         ("Tagging Processed Groups", tag_processed_groups),
#         ("Tagging Rows in Working Set", len(tag_rows) if isinstance(tag_rows, pd.DataFrame) else 0),
#         ("Tagging Merged Into Clean Trad", "Yes" if should_merge_tagging_into_clean_trad(session_state) else "No"),
#
#         ("Sentiment Run", "Yes" if sentiment_exists(session_state) else "No"),
#         ("Sentiment Scope", session_state.get("sentiment_sample_mode", "")),
#         ("Sentiment Sample Size Used", session_state.get("sentiment_sample_size", "")),
#         ("Sentiment Processed Groups", sent_processed_groups),
#         ("Sentiment Rows in Working Set", len(sent_rows) if isinstance(sent_rows, pd.DataFrame) else 0),
#         ("Sentiment Merged Into Clean Trad", "Yes" if should_merge_sentiment_into_clean_trad(session_state) else "No"),
#     ]
#
#     return pd.DataFrame(rows, columns=["Field", "Value"])
#
#
# # ---------- Workbook builder ----------
#
# def build_clean_workbook_bytes(session_state) -> bytes:
#     traditional = session_state.get("df_traditional", pd.DataFrame()).copy()
#     social = session_state.get("df_social", pd.DataFrame()).copy()
#     top_stories = session_state.get("added_df", pd.DataFrame()).copy()
#     dupes = session_state.get("df_dupes", pd.DataFrame()).copy()
#     raw = session_state.get("df_untouched", pd.DataFrame()).copy()
#
#     # Remove Published Date (redundant once Date is normalized)
#     for df in [traditional, social, dupes]:
#         if isinstance(df, pd.DataFrame) and "Published Date" in df.columns:
#             df.drop(columns=["Published Date"], inplace=True)
#
#     original_ave_col = session_state.get("original_ave_col") or "AVE"
#
#     traditional = merge_full_scope_ai_columns_into_clean_trad(session_state, traditional)
#     traditional = add_final_sentiment_columns(traditional)
#     traditional = explode_tags(traditional)
#
#     sentiment_export = add_final_sentiment_columns(
#         rename_ave(build_sentiment_sample_export(session_state), original_ave_col=original_ave_col)
#     )
#
#     social = explode_tags(social)
#
#     output = io.BytesIO()
#
#     with pd.ExcelWriter(output, engine="xlsxwriter", datetime_format="yyyy-mm-dd") as writer:
#         workbook = writer.book
#         cleaned_exports: list[tuple[str, pd.DataFrame, Any]] = []
#
#         number_format = workbook.add_format({"num_format": "#,##0"})
#         currency_symbol = get_currency_symbol(original_ave_col)
#         currency_format = workbook.add_format({"num_format": f"{currency_symbol}#,##0.00"})
#
#         # CLEAN TRAD
#         if len(traditional) > 0:
#             trad_export = rename_ave(traditional.copy(), original_ave_col=original_ave_col)
#             if "Impressions" in trad_export.columns:
#                 trad_export = trad_export.sort_values(by=["Impressions"], ascending=False)
#             trad_export.to_excel(writer, sheet_name="CLEAN TRAD", startrow=1, header=False, index=False)
#             ws = writer.sheets["CLEAN TRAD"]
#             ws.set_tab_color("black")
#             cleaned_exports.append(("CLEAN TRAD", trad_export, ws))
#
#         # CLEAN SOCIAL
#         if len(social) > 0:
#             social_export = rename_ave(social.copy(), original_ave_col=original_ave_col)
#             if "Impressions" in social_export.columns:
#                 social_export = social_export.sort_values(by=["Impressions"], ascending=False)
#             social_export.to_excel(writer, sheet_name="CLEAN SOCIAL", startrow=1, header=False, index=False)
#             ws = writer.sheets["CLEAN SOCIAL"]
#             ws.set_tab_color("black")
#             cleaned_exports.append(("CLEAN SOCIAL", social_export, ws))
#
#         # TAGGING SAMPLE
#         if (
#                 tagging_exists(session_state)
#                 and not should_merge_tagging_into_clean_trad(session_state)
#                 and tagging_processed_complete_enough(session_state)
#         ):
#         # if tagging_exists(session_state) and not should_merge_tagging_into_clean_trad(session_state):
#             tagging_export = rename_ave(build_tagging_sample_export(session_state), original_ave_col=original_ave_col)
#             if not tagging_export.empty:
#                 tagging_export.to_excel(writer, sheet_name="TAGGING SAMPLE", header=True, index=False)
#                 ws = writer.sheets["TAGGING SAMPLE"]
#                 ws.set_tab_color("#7f8c8d")
#                 cleaned_exports.append(("TAGGING SAMPLE", tagging_export, ws))
#
#         # SENTIMENT SAMPLE
#         if (
#                 sentiment_exists(session_state)
#                 and not should_merge_sentiment_into_clean_trad(session_state)
#                 and sentiment_processed_complete_enough(session_state)
#         ):
#         # if sentiment_exists(session_state) and not should_merge_sentiment_into_clean_trad(session_state):
#             sentiment_export = rename_ave(build_sentiment_sample_export(session_state), original_ave_col=original_ave_col)
#             if not sentiment_export.empty:
#                 sentiment_export.to_excel(writer, sheet_name="SENTIMENT SAMPLE", header=True, index=False)
#                 ws = writer.sheets["SENTIMENT SAMPLE"]
#                 ws.set_tab_color("#7f8c8d")
#                 cleaned_exports.append(("SENTIMENT SAMPLE", sentiment_export, ws))
#
#         # AUTHORS
#         authors = build_authors_export_table(
#             session_state.get("df_traditional", pd.DataFrame()).copy(),
#             existing_assignments=session_state.get("auth_outlet_table", pd.DataFrame()).copy()
#             if len(session_state.get("auth_outlet_table", pd.DataFrame())) > 0 else None,
#         )
#         if len(authors) > 0:
#             authors = authors.sort_values(by=["Mentions", "Impressions"], ascending=False).copy()
#             authors.to_excel(writer, sheet_name="Authors", header=True, index=False)
#             ws = writer.sheets["Authors"]
#             ws.set_tab_color("green")
#             cleaned_exports.append(("Authors", authors, ws))
#
#         # TOP STORIES
#         if len(top_stories) > 0:
#             top_stories_export = top_stories.copy()
#             if all(c in top_stories_export.columns for c in ["Mentions", "Impressions"]):
#                 top_stories_export = top_stories_export.sort_values(by=["Mentions", "Impressions"], ascending=False)
#
#             desired_top_story_columns = [
#                 "Headline",
#                 "Date",
#                 "Mentions",
#                 "Impressions",
#                 "Example Outlet",
#                 "Example URL",
#                 "Chart Callout",
#                 "Top Story Summary",
#                 "Entity Sentiment",
#             ]
#             existing_top_story_columns = [c for c in desired_top_story_columns if c in top_stories_export.columns]
#             top_stories_export = top_stories_export[existing_top_story_columns].copy()
#
#             top_stories_export.to_excel(writer, sheet_name="Top Stories", header=True, index=False)
#             ws = writer.sheets["Top Stories"]
#             ws.set_tab_color("green")
#             cleaned_exports.append(("Top Stories", top_stories_export, ws))
#
#         # DLTD DUPES
#         if len(dupes) > 0:
#             dupes_export = rename_ave(dupes.copy(), original_ave_col=original_ave_col)
#             dupes_export.to_excel(writer, sheet_name="DLTD DUPES", header=True, index=False)
#             ws = writer.sheets["DLTD DUPES"]
#             ws.set_tab_color("#c26f4f")
#             cleaned_exports.append(("DLTD DUPES", dupes_export, ws))
#
#         # RAW
#         raw_export = rename_ave(raw.copy(), original_ave_col=original_ave_col)
#         raw_export.drop(["Mentions"], axis=1, inplace=True, errors="ignore")
#         raw_export.to_excel(writer, sheet_name="RAW", header=True, index=False)
#         ws = writer.sheets["RAW"]
#         ws.set_tab_color("#c26f4f")
#         cleaned_exports.append(("RAW", raw_export, ws))
#
#         # EXPORT METADATA
#         metadata_export = build_export_metadata_sheet(session_state)
#         metadata_export.to_excel(writer, sheet_name="EXPORT METADATA", header=True, index=False)
#         ws = writer.sheets["EXPORT METADATA"]
#         ws.set_tab_color("#4f81bd")
#         cleaned_exports.append(("EXPORT METADATA", metadata_export, ws))
#
#         # Apply Excel table structures + formatting
#         for _, clean_df, ws in cleaned_exports:
#             if clean_df.empty:
#                 continue
#
#             max_row, max_col = clean_df.shape
#             column_settings = [{"header": column} for column in clean_df.columns]
#             ws.add_table(0, 0, max_row, max_col - 1, {"columns": column_settings})
#
#             if "Mentions" in clean_df.columns:
#                 col_idx = clean_df.columns.get_loc("Mentions")
#                 ws.set_column(col_idx, col_idx, None, None, {"hidden": True})
#
#             apply_sheet_column_formats(
#                 worksheet=ws,
#                 df=clean_df,
#                 number_format=number_format,
#                 currency_format=currency_format,
#                 original_ave_col=original_ave_col,
#             )
#
#     output.seek(0)
#     return output.getvalue()
#
# def add_final_sentiment_columns(df: pd.DataFrame) -> pd.DataFrame:
#     if df is None or df.empty:
#         return df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame()
#
#     out = df.copy()
#
#     # --- Resolve AI fields (prefer review if present) ---
#     ai_sent = None
#     ai_conf = None
#     ai_rat = None
#
#     if "Review AI Sentiment" in out.columns:
#         ai_sent = out["Review AI Sentiment"]
#         ai_conf = out.get("Review AI Confidence")
#         ai_rat = out.get("Review AI Rationale")
#     else:
#         ai_sent = out.get("AI Sentiment")
#         ai_conf = out.get("AI Sentiment Confidence")
#         ai_rat = out.get("AI Sentiment Rationale")
#
#     if ai_sent is not None:
#         out["AI Sentiment"] = ai_sent
#     if ai_conf is not None:
#         out["AI Sentiment Confidence"] = ai_conf
#     if ai_rat is not None:
#         out["AI Sentiment Rationale"] = ai_rat
#
#     # --- Hybrid sentiment ---
#     assigned = out.get("Assigned Sentiment", pd.Series(index=out.index, dtype="object"))
#     ai = out.get("AI Sentiment", pd.Series(index=out.index, dtype="object"))
#
#     assigned_clean = assigned.fillna("").astype(str).str.strip()
#     ai_clean = ai.fillna("").astype(str).str.strip()
#
#     out["Hybrid Sentiment"] = assigned_clean.where(assigned_clean != "", ai_clean)
#     out["Hybrid Sentiment"] = out["Hybrid Sentiment"].replace("", pd.NA)
#
#     # Confidence
#     if "AI Sentiment Confidence" in out.columns:
#         out["Hybrid Sentiment Confidence"] = out["AI Sentiment Confidence"]
#         out.loc[assigned_clean != "", "Hybrid Sentiment Confidence"] = pd.NA
#
#     return out

# def add_final_sentiment_columns(df: pd.DataFrame) -> pd.DataFrame:
#     if df is None or df.empty:
#         return df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame()
#
#     out = df.copy()
#
#     assigned = out["Assigned Sentiment"] if "Assigned Sentiment" in out.columns else pd.Series(index=out.index, dtype="object")
#     ai = out["AI Sentiment"] if "AI Sentiment" in out.columns else pd.Series(index=out.index, dtype="object")
#
#     assigned_clean = assigned.fillna("").astype(str).str.strip()
#     ai_clean = ai.fillna("").astype(str).str.strip()
#
#     out["Final Sentiment"] = assigned_clean.where(assigned_clean != "", ai_clean)
#     out["Final Sentiment"] = out["Final Sentiment"].replace("", pd.NA)
#
#     if "AI Sentiment Confidence" in out.columns:
#         out["Final Sentiment Confidence"] = out["AI Sentiment Confidence"]
#         if "Assigned Sentiment" in out.columns:
#             out.loc[assigned_clean != "", "Final Sentiment Confidence"] = pd.NA
#
#     return out
