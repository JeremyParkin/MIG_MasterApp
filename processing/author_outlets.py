# author_outlets.py
from __future__ import annotations

import urllib.parse

import numpy as np
import pandas as pd
import requests
from requests.structures import CaseInsensitiveDict
from unidecode import unidecode


FORMAT_DICT = {
    "AVE": "${0:,.0f}",
    "Audience Reach": "{:,.0f}",
    "Impressions": "{:,.0f}",
    "Mentions": "{:,.0f}",
}


def init_author_outlets_state(session_state) -> None:
    if "last_outlet_assignment" not in session_state:
        session_state.last_outlet_assignment = None

    if "auth_outlet_skipped" not in session_state:
        session_state.auth_outlet_skipped = 0

    if "top_auths_by" not in session_state:
        session_state.top_auths_by = "Mentions"


def undo_last_outlet_assignment(session_state) -> None:
    last_assignment = session_state.get("last_outlet_assignment")

    if not last_assignment:
        return

    author_name = last_assignment.get("author_name")
    previous_outlet = last_assignment.get("previous_outlet", "")
    previous_skip = last_assignment.get("previous_skip", session_state.auth_outlet_skipped)

    if not author_name:
        session_state.last_outlet_assignment = None
        return

    session_state.auth_outlet_table = session_state.auth_outlet_table.copy()
    session_state.auth_outlet_table.loc[
        session_state.auth_outlet_table["Author"] == author_name,
        "Outlet"
    ] = previous_outlet

    session_state.auth_outlet_skipped = previous_skip
    session_state.last_outlet_assignment = None


def reset_outlet_skips(session_state) -> None:
    session_state.auth_outlet_skipped = 0


def fetch_outlet(author_name: str, secrets) -> tuple[dict | None, dict]:
    contact_url = "https://mediadatabase.agilitypr.com/api/v4/contacts/search"

    headers = CaseInsensitiveDict()
    headers["Content-Type"] = "application/json"
    headers["Accept"] = "application/json"
    headers["Authorization"] = secrets["authorization"]
    headers["client_id"] = secrets["client_id"]
    headers["userclient_id"] = secrets["userclient_id"]

    data = f"""
    {{
      "aliases": [
        "{author_name}"
      ]
    }}
    """

    debug_info = {
        "request_author": author_name,
        "url": contact_url,
        "status_code": None,
        "ok": False,
        "error": "",
        "response_text_preview": "",
        "json_keys": [],
    }

    try:
        contact_resp = requests.post(contact_url, headers=headers, data=data, timeout=30)
        debug_info["status_code"] = contact_resp.status_code
        debug_info["ok"] = contact_resp.ok
        debug_info["response_text_preview"] = contact_resp.text[:1000]

        if not contact_resp.ok:
            debug_info["error"] = f"HTTP {contact_resp.status_code}"
            return None, debug_info

        try:
            parsed = contact_resp.json()
            if isinstance(parsed, dict):
                debug_info["json_keys"] = list(parsed.keys())
            return parsed, debug_info
        except Exception as e:
            debug_info["error"] = f"JSON decode failed: {e}"
            return None, debug_info

    except Exception as e:
        debug_info["error"] = f"Request failed: {e}"
        return None, debug_info


def prepare_traditional_for_author_outlets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "Mentions" in df.columns:
        df["Mentions"] = pd.to_numeric(df["Mentions"], errors="coerce").fillna(0).astype(int)

    return df


def build_auth_outlet_table(
    df: pd.DataFrame,
    top_auths_by: str,
    existing_assignments: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Rebuild the author-outlet summary table from df_traditional and preserve
    any existing outlet assignments where possible.
    """
    required_cols = ["Author", "Mentions", "Impressions"]
    existing_cols = [c for c in required_cols if c in df.columns]
    working = df[existing_cols].copy()

    if "Author" not in working.columns:
        return pd.DataFrame(columns=["Author", "Outlet", "Mentions", "Impressions"])

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

    if (
        existing_assignments is not None
        and len(existing_assignments) > 0
        and "Outlet" in existing_assignments.columns
    ):
        assignment_map = (
            existing_assignments[["Author", "Outlet"]]
            .copy()
            .fillna("")
        )

        assignment_map["Author"] = assignment_map["Author"].fillna("").astype(str).str.strip()
        assignment_map = assignment_map[assignment_map["Author"] != ""].copy()

        assignment_map["has_outlet"] = assignment_map["Outlet"].str.strip().ne("")
        assignment_map = (
            assignment_map.sort_values(["Author", "has_outlet"], ascending=[True, False])
            .drop_duplicates(subset=["Author"], keep="first")
            .drop(columns=["has_outlet"])
        )

        rebuilt = rebuilt.merge(assignment_map, on="Author", how="left")
        rebuilt["Outlet"] = rebuilt["Outlet"].fillna("")
    else:
        rebuilt.insert(loc=1, column="Outlet", value="")

    if top_auths_by == "Mentions":
        rebuilt = rebuilt.sort_values(["Mentions", "Impressions"], ascending=False).reset_index(drop=True)
    else:
        rebuilt = rebuilt.sort_values(["Impressions", "Mentions"], ascending=False).reset_index(drop=True)

    desired_order = ["Author", "Outlet", "Mentions", "Impressions"]
    rebuilt = rebuilt[[c for c in desired_order if c in rebuilt.columns]].copy()

    return rebuilt


def get_auth_outlet_todo(auth_outlet_table: pd.DataFrame) -> pd.DataFrame:
    if auth_outlet_table is None or auth_outlet_table.empty or "Outlet" not in auth_outlet_table.columns:
        return pd.DataFrame()

    return auth_outlet_table.loc[auth_outlet_table["Outlet"] == ""].copy().reset_index(drop=True)


def apply_author_name_fix(
    session_state,
    old_name: str,
    new_name: str,
) -> None:
    """
    Apply a corrected author name to df_traditional and rebuild auth_outlet_table.
    Keep the user on the renamed author so outlet assignment can happen next.
    """
    old_name = str(old_name).strip()
    new_name = str(new_name).strip()

    if not old_name or not new_name or old_name == new_name:
        return

    session_state.df_traditional = session_state.df_traditional.copy()
    session_state.df_traditional.loc[
        session_state.df_traditional["Author"] == old_name,
        "Author"
    ] = new_name

    existing_assignments = (
        session_state.auth_outlet_table.copy()
        if len(session_state.auth_outlet_table) > 0
        else None
    )

    session_state.auth_outlet_table = build_auth_outlet_table(
        session_state.df_traditional.copy(),
        session_state.top_auths_by,
        existing_assignments=existing_assignments,
    )

    auth_outlet_todo = get_auth_outlet_todo(session_state.auth_outlet_table)
    matching_rows = auth_outlet_todo.index[auth_outlet_todo["Author"] == new_name].tolist()

    if matching_rows:
        session_state.auth_outlet_skipped = matching_rows[0]
    else:
        session_state.auth_outlet_skipped = max(
            0,
            min(session_state.auth_outlet_skipped, len(auth_outlet_todo) - 1),
        )

    session_state.author_fix_input = new_name
    session_state.last_author_for_fix = new_name


def get_matched_authors_df(search_results, outlets_in_coverage_list) -> tuple[pd.DataFrame, list[str], list[str]]:
    """
    Build dataframe of matched authors from database response.
    Also returns db_outlets and possibles lists.
    """
    db_outlets: list[str] = []
    possibles: list[str] = []

    if not search_results or "results" not in search_results or not search_results["results"]:
        return pd.DataFrame(), db_outlets, possibles

    response_results = search_results["results"]
    outlet_results = []

    for result in response_results:
        first = result.get("firstName", "") or ""
        last = result.get("lastName", "") or ""
        auth_name = f"{first} {last}".strip()

        primary_employment = result.get("primaryEmployment") or {}
        job_title = primary_employment.get("jobTitle", "") or ""
        outlet = primary_employment.get("outletName", "") or ""

        country_obj = result.get("country")
        country = country_obj.get("name", "") if country_obj else ""

        outlet_results.append((auth_name, job_title, outlet, country))

    matched_authors = pd.DataFrame.from_records(
        outlet_results,
        columns=["Name", "Title", "Outlet", "Country"],
    ).copy()

    if len(matched_authors) == 0:
        return matched_authors, db_outlets, possibles

    matched_authors.loc[matched_authors["Outlet"] == "[Freelancer]", "Outlet"] = "Freelance"

    db_outlets = matched_authors["Outlet"].tolist()
    possibles = matched_authors["Outlet"].tolist()

    matching_outlets = set(outlets_in_coverage_list).intersection(set(possibles))

    if len(matching_outlets) > 0 and len(possibles) > 1:
        matched_authors_top = matched_authors[matched_authors["Outlet"].isin(matching_outlets)].copy()
        matched_authors_bottom = matched_authors[~matched_authors["Outlet"].isin(matching_outlets)].copy()
        matched_authors = pd.concat([matched_authors_top, matched_authors_bottom], ignore_index=True)
        possibles = matched_authors["Outlet"].tolist()

    matching_outlet = [outlet for outlet in outlets_in_coverage_list if outlet in possibles]
    if len(matching_outlet) == 1:
        index = possibles.index(matching_outlet[0])
        possibles = [matching_outlet[0]] + possibles[:index] + possibles[index + 1:]

    return matched_authors, db_outlets, possibles


def get_outlets_in_coverage(df_traditional: pd.DataFrame, author_name: str) -> pd.DataFrame:
    if df_traditional is None or df_traditional.empty:
        return pd.DataFrame(columns=["Outlet", "Hits"])

    if "Author" not in df_traditional.columns or "Outlet" not in df_traditional.columns:
        return pd.DataFrame(columns=["Outlet", "Hits"])

    outlets_in_coverage = (
        df_traditional.loc[df_traditional["Author"] == author_name, "Outlet"]
        .value_counts()
        .rename_axis("Outlet")
        .reset_index(name="Hits")
        .copy()
    )

    return outlets_in_coverage


def build_outlet_assignment_payload(
    auth_outlet_table: pd.DataFrame,
    author_name: str,
    previous_skip: int,
) -> dict:
    previous_outlet_series = auth_outlet_table.loc[
        auth_outlet_table["Author"] == author_name,
        "Outlet",
    ]

    previous_outlet = previous_outlet_series.iloc[0] if len(previous_outlet_series) > 0 else ""

    return {
        "author_name": author_name,
        "previous_outlet": previous_outlet,
        "previous_skip": previous_skip,
    }


def assign_outlet(
    auth_outlet_table: pd.DataFrame,
    author_name: str,
    new_outlet: str,
) -> pd.DataFrame:
    updated = auth_outlet_table.copy()
    updated.loc[updated["Author"] == author_name, "Outlet"] = new_outlet
    return updated


def get_author_search_urls(author_name: str) -> tuple[str, str]:
    encoded_author_name = urllib.parse.quote(author_name)
    muckrack_url = f"https://www.google.com/search?q=site%3Amuckrack.com+{encoded_author_name}"
    linkedin_url = f'https://www.google.com/search?q=site%3Alinkedin.com+%22{encoded_author_name}%22+journalist'
    return muckrack_url, linkedin_url


def get_search_author_name(author_name: str) -> str:
    return unidecode(str(author_name or "").strip())