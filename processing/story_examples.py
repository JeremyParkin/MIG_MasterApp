# story_examples.py
from __future__ import annotations

import pandas as pd


def pick_best_story_row(group: pd.DataFrame) -> pd.Series | None:
    """Pick the best representative row for a grouped story."""
    if group.empty:
        return None

    working = group.copy()

    for col in ["Outlet", "Type", "Snippet", "URL", "Headline", "Coverage Flags"]:
        if col not in working.columns:
            working[col] = ""
        working[col] = working[col].fillna("").astype(str)

    if "Impressions" not in working.columns:
        working["Impressions"] = 0

    working["Impressions"] = pd.to_numeric(working["Impressions"], errors="coerce").fillna(0)

    # Highest-priority legit news agencies
    preferred_wire_pattern = r"Reuters|Associated Press|Canadian Press"
    working["_is_preferred_wire"] = working["Outlet"].str.contains(
        preferred_wire_pattern,
        case=False,
        na=False,
        regex=True,
    )

    # Broadcast: keep it simple, but still let preferred wires win if somehow present
    is_broadcast = working["Type"].str.upper().isin(["TV", "RADIO", "PODCAST"]).any()

    # Coverage flag buckets
    flags = working["Coverage Flags"].str.strip()

    working["_is_good_outlet"] = flags.eq("Good Outlet")
    working["_is_aggregator"] = flags.eq("Aggregator")
    working["_is_newswire"] = flags.eq("Newswire?")
    working["_is_stocks"] = flags.eq("Stocks / Financials?")
    working["_is_market_report"] = flags.eq("Market Report Spam?")
    working["_is_advertorial"] = flags.eq("Advertorial?")
    working["_is_unflagged"] = flags.eq("")

    # Quality rank: lower is better
    # 0 = Reuters/AP/Canadian Press
    # 1 = Good Outlet
    # 2 = Other non-flagged
    # 3 = Aggregator
    # 4 = Newswire? / press-release distribution
    # 5 = Stocks / Financials?
    # 6 = Market Report Spam?
    # 7 = Advertorial?
    working["_quality_rank"] = 2

    working.loc[working["_is_advertorial"], "_quality_rank"] = 7
    working.loc[working["_is_market_report"], "_quality_rank"] = 6
    working.loc[working["_is_stocks"], "_quality_rank"] = 5
    working.loc[working["_is_newswire"], "_quality_rank"] = 4
    working.loc[working["_is_aggregator"], "_quality_rank"] = 3
    working.loc[working["_is_unflagged"], "_quality_rank"] = 2
    working.loc[working["_is_good_outlet"], "_quality_rank"] = 1
    working.loc[working["_is_preferred_wire"], "_quality_rank"] = 0

    if "Date" in working.columns:
        working["_date_dt"] = pd.to_datetime(working["Date"], errors="coerce")
    else:
        working["_date_dt"] = pd.NaT

    # For broadcast, still use the same rank ordering, but mostly impressions decide within tier
    sort_cols = ["_quality_rank", "Impressions", "_date_dt"]
    ascending = [True, False, True]

    working = working.sort_values(
        by=sort_cols,
        ascending=ascending,
        na_position="last",
    )

    return working.iloc[0]
#
# from __future__ import annotations
#
# import re
# from typing import Any
#
# import pandas as pd
#
#
# def pick_best_story_row(group: pd.DataFrame) -> pd.Series | None:
#     """Pick the best representative row for a grouped story."""
#     if group.empty:
#         return None
#
#     working = group.copy()
#     working["Outlet"] = working["Outlet"].fillna("").astype(str)
#     working["Type"] = working["Type"].fillna("").astype(str)
#     working["Snippet"] = working["Snippet"].fillna("").astype(str)
#     working["URL"] = working["URL"].fillna("").astype(str)
#     working["Headline"] = working["Headline"].fillna("").astype(str)
#     working["Impressions"] = pd.to_numeric(working["Impressions"], errors="coerce").fillna(0)
#
#     preferred_wire_pattern = r"Reuters|Associated Press|Canadian Press"
#     preferred_wire_group = working[
#         working["Outlet"].str.contains(preferred_wire_pattern, case=False, na=False, regex=True)
#     ]
#
#     if not preferred_wire_group.empty:
#         return preferred_wire_group.loc[preferred_wire_group["Impressions"].idxmax()]
#
#     is_broadcast = working["Type"].isin(["TV", "RADIO", "PODCAST"]).any()
#
#     middle_tier_keywords = [
#         "MarketWatch", "Seeking Alpha", "News Break", "Dispatchist",
#         "MarketScreener", "StreetInsider", "Head Topics"
#     ]
#     bottom_tier_keywords = [
#         "Yahoo", "MSN", "AOL", "Newswire", "Saltwire", "Market Wire",
#         "Business Wire", "TD Ameritrade", "PR Wire", "Chinese Wire",
#         "News Wire", "Presswire"
#     ]
#
#     middle_pattern = "|".join(re.escape(x) for x in middle_tier_keywords)
#     bottom_pattern = "|".join(re.escape(x) for x in bottom_tier_keywords)
#     combined_pattern = "|".join(re.escape(x) for x in (middle_tier_keywords + bottom_tier_keywords))
#
#     if is_broadcast:
#         return working.loc[working["Impressions"].idxmax()]
#
#     top_tier_group = working[
#         ~working["Outlet"].str.contains(combined_pattern, case=False, na=False, regex=True)
#     ]
#     middle_tier_group = working[
#         working["Outlet"].str.contains(middle_pattern, case=False, na=False, regex=True)
#         & ~working["Outlet"].str.contains(bottom_pattern, case=False, na=False, regex=True)
#     ]
#
#     if not top_tier_group.empty:
#         return top_tier_group.loc[top_tier_group["Impressions"].idxmax()]
#     if not middle_tier_group.empty:
#         return middle_tier_group.loc[middle_tier_group["Impressions"].idxmax()]
#
#     return working.loc[working["Impressions"].idxmax()]