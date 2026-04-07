from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd


TOP_STORY_DEFAULT_COLUMNS = {
    "Group ID": pd.NA,
    "Headline": "",
    "Date": pd.NaT,
    "Mentions": 1,
    "Impressions": 0,
    "Type": "",
    "Outlet": "",
    "URL": "",
    "Snippet": "",
    "Tags": "",
    "Coverage Flags": "",
}


def normalize_top_stories_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize dataframe enough for Top Stories page to work safely."""
    df = df.copy()

    for col, default in TOP_STORY_DEFAULT_COLUMNS.items():
        if col not in df.columns:
            df[col] = default

    text_columns = ["Headline", "Type", "Outlet", "URL", "Snippet", "Tags", "Coverage Flags"]
    for col in text_columns:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)

    if "Mentions" in df.columns:
        df["Mentions"] = pd.to_numeric(df["Mentions"], errors="coerce").fillna(1).astype(int)

    if "Impressions" in df.columns:
        df["Impressions"] = pd.to_numeric(df["Impressions"], errors="coerce").fillna(0)

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.date

    return df


def pick_best_story_row(group: pd.DataFrame) -> pd.Series | None:
    """Pick the best example row for a grouped story."""
    if group.empty:
        return None

    working = group.copy()
    working["Outlet"] = working["Outlet"].fillna("").astype(str)
    working["Type"] = working["Type"].fillna("").astype(str)
    working["Snippet"] = working["Snippet"].fillna("").astype(str)
    working["URL"] = working["URL"].fillna("").astype(str)
    working["Headline"] = working["Headline"].fillna("").astype(str)
    working["Impressions"] = pd.to_numeric(working["Impressions"], errors="coerce").fillna(0)

    preferred_wire_pattern = r"Reuters|Associated Press|Canadian Press"
    preferred_wire_group = working[
        working["Outlet"].str.contains(preferred_wire_pattern, case=False, na=False, regex=True)
    ]

    if not preferred_wire_group.empty:
        return preferred_wire_group.loc[preferred_wire_group["Impressions"].idxmax()]

    is_broadcast = working["Type"].isin(["TV", "RADIO", "PODCAST"]).any()

    middle_tier_keywords = [
        "MarketWatch", "Seeking Alpha", "News Break", "Dispatchist",
        "MarketScreener", "StreetInsider", "Head Topics"
    ]
    bottom_tier_keywords = [
        "Yahoo", "MSN", "AOL", "Newswire", "Saltwire", "Market Wire",
        "Business Wire", "TD Ameritrade", "PR Wire", "Chinese Wire",
        "News Wire", "Presswire"
    ]

    middle_pattern = "|".join(re.escape(x) for x in middle_tier_keywords)
    bottom_pattern = "|".join(re.escape(x) for x in bottom_tier_keywords)
    combined_pattern = "|".join(re.escape(x) for x in (middle_tier_keywords + bottom_tier_keywords))

    if is_broadcast:
        return working.loc[working["Impressions"].idxmax()]

    top_tier_group = working[
        ~working["Outlet"].str.contains(combined_pattern, case=False, na=False, regex=True)
    ]
    middle_tier_group = working[
        working["Outlet"].str.contains(middle_pattern, case=False, na=False, regex=True)
        & ~working["Outlet"].str.contains(bottom_pattern, case=False, na=False, regex=True)
    ]

    if not top_tier_group.empty:
        return top_tier_group.loc[top_tier_group["Impressions"].idxmax()]
    if not middle_tier_group.empty:
        return middle_tier_group.loc[middle_tier_group["Impressions"].idxmax()]

    return working.loc[working["Impressions"].idxmax()]


def build_grouped_story_candidates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group filtered row-level stories by Group ID and attach exemplar row details.
    """
    df = normalize_top_stories_df(df)

    if df.empty or "Group ID" not in df.columns:
        return pd.DataFrame(columns=[
            "Group ID",
            "Headline",
            "Date",
            "Mentions",
            "Impressions",
            "Example Outlet",
            "Example URL",
            "Example Type",
            "Example Snippet",
        ])

    rows: list[dict[str, Any]] = []

    for group_id, group in df.groupby("Group ID", dropna=False):
        mentions = pd.to_numeric(group["Mentions"], errors="coerce").fillna(0).sum()
        impressions = pd.to_numeric(group["Impressions"], errors="coerce").fillna(0).sum()

        best_row = pick_best_story_row(group)
        if best_row is None:
            continue

        rows.append({
            "Group ID": group_id,
            "Headline": best_row.get("Headline", ""),
            "Date": best_row.get("Date", pd.NaT),
            "Mentions": int(mentions),
            "Impressions": impressions,
            "Example Outlet": best_row.get("Outlet", ""),
            "Example URL": best_row.get("URL", ""),
            "Example Type": best_row.get("Type", ""),
            "Example Snippet": best_row.get("Snippet", ""),
        })

    if not rows:
        return pd.DataFrame(columns=[
            "Group ID",
            "Headline",
            "Date",
            "Mentions",
            "Impressions",
            "Example Outlet",
            "Example URL",
            "Example Type",
            "Example Snippet",
        ])

    result = pd.DataFrame(rows)
    result["Date"] = pd.to_datetime(result["Date"], errors="coerce").dt.date
    result["Impressions"] = pd.to_numeric(result["Impressions"], errors="coerce").fillna(0)
    result["Mentions"] = pd.to_numeric(result["Mentions"], errors="coerce").fillna(0).astype(int)

    return result[[
        "Group ID",
        "Headline",
        "Date",
        "Mentions",
        "Impressions",
        "Example Outlet",
        "Example URL",
        "Example Type",
        "Example Snippet",
    ]]


def tokenize_boolean_query(query: str):
    token_pattern = r'"[^"]+"|\(|\)|\bAND\b|\bOR\b|\bNOT\b|[^\s()]+'
    tokens = re.findall(token_pattern, query, flags=re.IGNORECASE)
    return [token.strip() for token in tokens if token.strip()]


def normalize_tokens_for_implicit_and(tokens):
    normalized = []

    def is_operand(tok):
        upper = tok.upper()
        return tok not in ("(", ")") and upper not in ("AND", "OR", "NOT")

    def ends_expr(tok):
        return tok == ")" or is_operand(tok)

    def starts_expr(tok):
        upper = tok.upper()
        return tok == "(" or is_operand(tok) or upper == "NOT"

    for i, token in enumerate(tokens):
        if i > 0:
            prev = tokens[i - 1]
            if ends_expr(prev) and starts_expr(token):
                normalized.append("AND")
        normalized.append(token)

    return normalized


def boolean_query_to_postfix(tokens):
    precedence = {"NOT": 3, "AND": 2, "OR": 1}
    right_associative = {"NOT"}
    output = []
    operators = []

    for token in tokens:
        upper = token.upper()

        if upper in precedence:
            while (
                operators
                and operators[-1] != "("
                and (
                    precedence.get(operators[-1], 0) > precedence[upper]
                    or (
                        precedence.get(operators[-1], 0) == precedence[upper]
                        and upper not in right_associative
                    )
                )
            ):
                output.append(operators.pop())
            operators.append(upper)

        elif token == "(":
            operators.append(token)

        elif token == ")":
            while operators and operators[-1] != "(":
                output.append(operators.pop())
            if operators and operators[-1] == "(":
                operators.pop()

        else:
            output.append(token)

    while operators:
        output.append(operators.pop())

    return output


def evaluate_boolean_query(series: pd.Series, query: str) -> pd.Series:
    series = series.fillna("").astype(str)

    query = str(query).strip()
    if not query:
        return pd.Series(True, index=series.index)

    tokens = tokenize_boolean_query(query)
    if not tokens:
        return pd.Series(True, index=series.index)

    tokens = normalize_tokens_for_implicit_and(tokens)
    postfix = boolean_query_to_postfix(tokens)
    stack = []

    for token in postfix:
        upper = token.upper()

        if upper == "NOT":
            if len(stack) < 1:
                return pd.Series(False, index=series.index)
            operand = stack.pop()
            stack.append(~operand)

        elif upper == "AND":
            if len(stack) < 2:
                return pd.Series(False, index=series.index)
            right = stack.pop()
            left = stack.pop()
            stack.append(left & right)

        elif upper == "OR":
            if len(stack) < 2:
                return pd.Series(False, index=series.index)
            right = stack.pop()
            left = stack.pop()
            stack.append(left | right)

        else:
            term = token[1:-1] if len(token) >= 2 and token.startswith('"') and token.endswith('"') else token
            mask = series.str.contains(term, case=False, na=False, regex=False)
            stack.append(mask)

    if len(stack) != 1:
        return pd.Series(False, index=series.index)

    return stack[0]


def apply_filters(
    df: pd.DataFrame,
    start_date,
    end_date,
    exclude_types,
    exclude_coverage_flags,
    advanced_filters,
):
    working = normalize_top_stories_df(df.copy())

    if start_date is not None:
        working = working[working["Date"].notna()]
        working = working[working["Date"] >= start_date]

    if end_date is not None:
        working = working[working["Date"].notna()]
        working = working[working["Date"] <= end_date]

    if exclude_types:
        working = working[~working["Type"].isin(exclude_types)]

    if exclude_coverage_flags and "Coverage Flags" in working.columns:
        working = working[~working["Coverage Flags"].isin(exclude_coverage_flags)]

    for condition in advanced_filters:
        column = condition.get("column")
        value = condition.get("value")

        if not column or not value or column not in working.columns:
            continue

        series = working[column].fillna("").astype(str)
        mask = evaluate_boolean_query(series, value)
        working = working[mask]

    return working


def save_selected_rows(
    updated_data: pd.DataFrame,
    full_candidate_df: pd.DataFrame,
    added_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Return updated saved top stories dataset, preserving hidden/internal columns
    by selecting from the full candidate dataframe using Group ID.
    """
    if "Top Story" not in updated_data.columns or "Group ID" not in updated_data.columns:
        return added_df

    selected_ids = (
        updated_data.loc[updated_data["Top Story"] == True, "Group ID"]
        .dropna()
        .tolist()
    )

    if not selected_ids:
        return added_df

    selected_rows = full_candidate_df.loc[
        full_candidate_df["Group ID"].isin(selected_ids)
    ].copy()

    new_added_df = pd.concat([added_df, selected_rows], ignore_index=True)

    if not new_added_df.empty and "Group ID" in new_added_df.columns:
        new_added_df.drop_duplicates(subset=["Group ID"], keep="last", inplace=True)
        new_added_df.reset_index(drop=True, inplace=True)

    return new_added_df
#
# def save_selected_rows(
#     updated_data: pd.DataFrame,
#     added_df: pd.DataFrame,
# ) -> pd.DataFrame:
#     """Return updated saved top stories dataset."""
#     if "Top Story" not in updated_data.columns:
#         return added_df
#
#     selected_rows = updated_data.loc[updated_data["Top Story"] == True].copy()
#     selected_rows.drop(columns=["Top Story"], inplace=True, errors="ignore")
#
#     new_added_df = pd.concat([added_df, selected_rows], ignore_index=True)
#
#     if not new_added_df.empty and "Group ID" in new_added_df.columns:
#         new_added_df.drop_duplicates(subset=["Group ID"], keep="last", inplace=True)
#         new_added_df.reset_index(drop=True, inplace=True)
#
#     return new_added_df


def remove_saved_candidates_from_display(
    display_df: pd.DataFrame,
    added_df: pd.DataFrame,
) -> pd.DataFrame:
    if display_df.empty or added_df.empty or "Group ID" not in display_df.columns or "Group ID" not in added_df.columns:
        return display_df

    existing_ids = set(added_df["Group ID"].dropna().tolist())
    return display_df[~display_df["Group ID"].isin(existing_ids)].copy()


def reset_generated_candidates(session_state) -> None:
    session_state.df_grouped = pd.DataFrame()
    session_state.filtered_df = pd.DataFrame()
    session_state.top_stories_generated = False