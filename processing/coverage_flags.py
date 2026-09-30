# coverage_flags.py

import pandas as pd
import re

FLAG_SPLIT_PATTERN = r"[;,]\s*|\|\s*"
STORY_FAMILY_FLAGS_COL = "Story Family Flags"
STORY_FAMILY_PRESS_RELEASE_EVIDENCE_COL = "Story Family Press Release Evidence"

PRESS_RELEASE_DISTRIBUTOR_TERMS = [
    "pressrelease", "accesswire", "business wire", "businesswire", "CNW",
    "presswire", "openPR", "pr-gateway", "Prlog", "PRWEB", "Pressebox",
    "Presseportal", "RTTNews", "SBWIRE", "issuewire", "prunderground",
]
PRESS_RELEASE_SNIPPET_TERMS = [*PRESS_RELEASE_DISTRIBUTOR_TERMS, "newswire"]
PRESS_RELEASE_AUTHOR_PATTERN = r"newswire|press\s*release|distribution|newsfile"
PRESS_RELEASE_URL_PATTERN = r"/pr\.|news-release|press-release|newswise\.com"


def split_coverage_flags(value: object) -> list[str]:
    if value is None or value is pd.NA:
        return []
    try:
        if bool(pd.isna(value)):
            return []
    except (TypeError, ValueError):
        pass
    raw = str(value).strip()
    if not raw:
        return []
    parts = re.split(FLAG_SPLIT_PATTERN, raw)
    return [part.strip() for part in parts if part.strip()]


def has_coverage_flag(value: object, target_flag: str) -> bool:
    target = str(target_flag or "").strip()
    if not target:
        return False
    return target in split_coverage_flags(value)


def effective_coverage_flags(row: pd.Series | dict | object) -> list[str]:
    """Return direct and derived story-family flags without changing source truth."""
    if not isinstance(row, (pd.Series, dict)):
        return split_coverage_flags(row)

    direct = split_coverage_flags(row.get("Coverage Flags", ""))
    family = split_coverage_flags(row.get(STORY_FAMILY_FLAGS_COL, ""))
    return list(dict.fromkeys([*direct, *family]))


def has_effective_coverage_flag(row: pd.Series | dict | object, target_flag: str) -> bool:
    target = str(target_flag or "").strip()
    return bool(target) and target in effective_coverage_flags(row)


def _text_series(df: pd.DataFrame, column: str) -> pd.Series:
    return df.get(column, pd.Series("", index=df.index, dtype="object")).fillna("").astype(str)


def _original_type_is_press_release(df: pd.DataFrame) -> pd.Series:
    original_type = _text_series(df, "Original Type")
    return original_type.str.upper().str.replace("_", " ", regex=False).str.replace(r"\s+", " ", regex=True).str.strip().eq("PRESS RELEASE")


def get_strong_press_release_evidence(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Identify deterministic evidence that may classify an entire story family.

    Snippet-only matches intentionally remain direct row-level signals: less reliable
    evidence should not spread across a canonical Group ID.
    """
    outlet = _text_series(df, "Outlet")
    url = _text_series(df, "URL")
    author = _text_series(df, "Author")
    source_type = _original_type_is_press_release(df)
    distributor_outlet = outlet.str.contains(
        "|".join(re.escape(term) for term in PRESS_RELEASE_DISTRIBUTOR_TERMS),
        case=False,
        na=False,
        regex=True,
    ) | outlet.str.contains("EurekAlert", case=False, na=False, regex=False)
    release_url = url.str.contains(PRESS_RELEASE_URL_PATTERN, case=False, na=False, regex=True)
    release_author = author.str.contains(PRESS_RELEASE_AUTHOR_PATTERN, case=False, na=False, regex=True)

    strong = source_type | distributor_outlet | release_url | release_author
    evidence = pd.Series("", index=df.index, dtype="object")
    evidence = evidence.mask(source_type, "Source type")
    evidence = evidence.mask(~source_type & release_url, "Press-release URL")
    evidence = evidence.mask(~source_type & ~release_url & release_author, "Press-release author")
    evidence = evidence.mask(
        ~source_type & ~release_url & ~release_author & distributor_outlet,
        "Distribution outlet",
    )
    return strong, evidence


def apply_story_family_press_release_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Annotate canonical groups while leaving direct row-level flags untouched."""
    out = df.copy()
    if "Group ID" not in out.columns:
        return out

    out[STORY_FAMILY_FLAGS_COL] = ""
    out[STORY_FAMILY_PRESS_RELEASE_EVIDENCE_COL] = ""
    strong, evidence = get_strong_press_release_evidence(out)
    if not strong.any():
        return out

    strong_evidence = out.loc[strong, ["Group ID"]].copy()
    strong_evidence["Evidence"] = evidence.loc[strong].astype(str)
    evidence_by_group = (
        strong_evidence.groupby("Group ID", dropna=False)["Evidence"]
        .agg(lambda values: "; ".join(dict.fromkeys(value for value in values if value)))
    )
    group_evidence = out["Group ID"].map(evidence_by_group).fillna("")
    family_press_release = group_evidence.ne("")
    out.loc[family_press_release, STORY_FAMILY_FLAGS_COL] = "Press Release"
    out.loc[family_press_release, STORY_FAMILY_PRESS_RELEASE_EVIDENCE_COL] = group_evidence.loc[family_press_release]
    return out


def extract_relevant_text(snippet: str) -> str:
    words = str(snippet or "").split()
    if len(words) > 250:
        return " ".join(words[:125] + words[-125:])
    return str(snippet or "")

def add_coverage_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    stock_moves_phrases = [
        "ADVFN", "ARIVA.DE", "Benzinga", "Barchart", "Daily Advent", "ETF Daily News",
        "FinanzNachrichten.de", "Finanzen.at", "Finanzen.ch", "FONDS exclusiv",
        "Market Beat", "Market Newsdesk", "Market Newswire", "Market Screener",
        "Market Wire News", "MarketBeat", "MarketScreener", "MarketWatch", "Nasdaq",
        "Seeking Alpha", "Stock Observer", "Stock Titan", "Stockhouse", "Stockstar", "Zacks"
    ]

    aggregators_list = [
        "Yahoo", "MSN", "News Break", "Google News", "Apple News", "Flipboard",
        "Pocket", "Feedly", "SmartNews", "StumbleUpon", "Ground News", "DNyuz",
        "Mirage News", "Newstex Blogs", "Trading View", "AOL", "Legacy.com", "World Atlas"
    ]

    user_generated_domains = [
        "medium.com",
        "substack.com",
        "slideshare.net",
    ]

    outlet_names = [
        "Associated Press", "National Post", "The Canadian Press", "The Globe and Mail", "Toronto Star",
        "Calgary Herald", "Edmonton Journal", "Montreal Gazette", "Ottawa Citizen",
        "The Chronicle Herald", "The Telegram", "Vancouver Sun", "Winnipeg Free Press",
        "The Globe", "Toronto Life", "BlogTO", "CBC News", "CBC ", "CityNews", "City ",
        "Citytv ", "CTV ", "CP24", "Daily Hive", "Global News", "La Presse", "Le Devoir",
        "Le Journal de Montréal", "Radio-Canada", "BNN Bloomberg", "Financial Post",
        "rabble.ca", "The Tyee", "The Walrus", "CHCH", "CHEK News", "NOW Magazine",
        "The Georgia Straight", "HuffPost Canada", "iPolitics", "TVO.org", "OMNI Television",
        "Sing Tao Daily", "APTN National News", "Calgary Sun", "Edmonton Sun",
        "Hamilton Spectator", "Kingston Whig-Standard", "London Free Press", "Ottawa Sun",
        "Regina Leader-Post", "Sault Star", "StarPhoenix", "Sudbury Star", "The Province",
        "Toronto Sun", "Windsor Star", "Winnipeg Sun", "Bloomberg", "Financial Times",
        "Macleans", "Reuters", "Journal de Quebec", "L'Actualite", "Le Droit", "Le Soleil",
        "Les Affaires", "TVA Nouvelles", "Times Colonist", "The New York Times",
        "The Washington Post", "USA Today", "Los Angeles Times", "Chicago Tribune",
        "The Boston Globe", "The Dallas Morning News", "The Philadelphia Inquirer",
        "San Francisco Chronicle", "Miami Herald", "The Seattle Times", "Houston Chronicle",
        "The Salt Lake Tribune", "Deseret News", "Albany Times Union", "Arkansas Democrat-Gazette",
        "Austin American-Statesman", "Bakersfield Californian", "Buffalo News",
        "Charleston Gazette-Mail", "The Columbus Dispatch", "The Fresno Bee", "Hartford Courant",
        "Idaho Statesman", "Las Vegas Review-Journal", "The Ledger", "Lexington Herald-Leader",
        "The Modesto Bee", "The Morning Call", "New Haven Register", "Omaha World-Herald",
        "Palm Beach Post", "Patriot-News", "Pittsburgh Post-Gazette", "Richmond Times-Dispatch",
        "The Sacramento Bee", "The Spokesman-Review", "Syracuse Post-Standard", "The Tennessean",
        "The Trentonian", "Tulsa World", "The Virginian-Pilot", "The Wichita Eagle",
        "The Star-Ledger", "The News & Observer", "The News Tribune", "Reno Gazette-Journal",
        "The Clarion-Ledger", "The State", "Daily Press", "The Ann Arbor News", "The Day",
        "The Press-Enterprise", "South Florida Sun Sentinel", "The Providence Journal",
        "Daily Herald", "The Times-Picayune/The New Orleans Advocate", "The Star Press",
        "The Pueblo Chieftain", "The Record", "The Roanoke Times", "The Daily Breeze",
        "The Vindicator", "Waco Tribune-Herald", "Yakima Herald-Republic", "York Daily Record",
        "NPR", "ABC News", "NBC News", "CBS News", "CNN", "Fox News", "CNBC",
        "The Wall Street Journal", "Barron's", "ProPublica", "The Atlantic", "Politico",
        "Vox", "Slate", "The Nation", "Mother Jones", "The Hill", "Axios", "BuzzFeed News",
        "Vice News", "HuffPost", "The Verge", "Univision", "Telemundo", "Indian Country Today",
        "The Detroit News", "New York Post", "San Diego Union-Tribune", "The Baltimore Sun",
        "Orlando Sentinel", "The Denver Post", "The Plain Dealer", "The Charlotte Observer",
        "St. Louis Post-Dispatch", "The Kansas City Star", "The Tampa Bay Times",
        "The Star Tribune", "Milwaukee Journal Sentinel", "The Indianapolis Star",
        "The Courier-Journal", "The Times", "The Guardian", "The Daily Telegraph",
        "The Independent", "The Sun", "The Daily Mail", "The Mirror", "The Observer",
        "The Sunday Times", "The Evening Standard", "Yorkshire Post", "The Scotsman",
        "Manchester Evening News", "Liverpool Echo", "Birmingham Mail", "Wales Online",
        "Belfast Telegraph", "The Herald Scotland", "ITV News", "BBC News", "Channel 4 News",
        "Sky News", "Reuters UK", "City A.M.", "The Economist", "The Spectator",
        "New Statesman", "The Week", "Prospect Magazine", "The Conversation UK",
        "HuffPost UK", "Metro", "The Register", "PinkNews", "Al Jazeera English (UK)",
        "The National (Scotland)", "The Courier (Dundee)", "Cambridge News",
        "Eastern Daily Press", "Oxford Mail", "Swindon Advertiser", "The Argus (Brighton)",
        "Kent Online", "Lincolnshire Echo", "Gloucestershire Live", "The Waterloo Region Record",
    ]

    for col in [
        "Newswire Flag", "Market Report Flag", "Financial Outlet Flag",
        "Advertorial Flag", "Possible Advertorial Flag", "Good Outlet Flag", "Aggregator Flag",
        "User-Generated Flag", "Coverage Flags"
    ]:

    # for col in [
    #     "Newswire Flag", "Market Report Flag", "Stock Moves Flag",
    #     "Advertorial Flag", "Good Outlet Flag", "Aggregator Flag", "Coverage Flags"
    # ]:
        df[col] = ""

    if "Snippet" not in df.columns:
        df["Snippet"] = ""
    if "Author" not in df.columns:
        df["Author"] = ""
    if "Outlet" not in df.columns:
        df["Outlet"] = ""
    if "URL" not in df.columns:
        df["URL"] = ""
    if "Headline" not in df.columns:
        df["Headline"] = ""

    df["Snippet_Limited"] = df["Snippet"].apply(extract_relevant_text)

    headline_series = df["Headline"].fillna("").astype(str)
    outlet_series = df["Outlet"].fillna("").astype(str)

    newswire_mask = df["Snippet_Limited"].str.contains(
        "|".join(re.escape(phrase) for phrase in PRESS_RELEASE_SNIPPET_TERMS),
        case=False,
        na=False,
        regex=True,
    )

    newswire_mask = (
        newswire_mask
        | outlet_series.str.contains(
            "|".join(re.escape(phrase) for phrase in PRESS_RELEASE_SNIPPET_TERMS),
            case=False,
            na=False,
            regex=True,
        )
        | df["Outlet"].str.contains("EurekAlert", case=False, na=False)
        | df["URL"].str.contains(r"/pr\.|news-release|press-release|newswise\.com", case=False, na=False, regex=True)
        | df["Author"].str.contains(PRESS_RELEASE_AUTHOR_PATTERN, case=False, na=False, regex=True)
    )

    advertorial_snippet_mask = df["Snippet_Limited"].str.contains(
        r"advertorial|sponsored content|brandpoint",
        case=False,
        na=False,
        regex=True,
    )
    advertorial_url_mask = df["URL"].str.contains(
        r"sponsored|advertorial|brandpoint|paid[-_/ ]?post|paid[-_/ ]?content|partner[-_/ ]?content",
        case=False,
        na=False,
        regex=True,
    )
    advertorial_author_mask = (
        df["Author"].str.fullmatch("Brandpoint", case=False, na=False)
        | df["Author"].str.contains(
            r"sponsored content|partner content|paid content|brand studio|content studio",
            case=False,
            na=False,
            regex=True,
        )
    )
    advertorial_mask = advertorial_url_mask | advertorial_author_mask
    possible_advertorial_mask = advertorial_snippet_mask & ~advertorial_mask

    df.drop(columns=["Snippet_Limited"], inplace=True, errors="ignore")

    financial_outlet_mask = outlet_series.str.contains(
        "|".join(re.escape(phrase) for phrase in stock_moves_phrases),
        case=False,
        na=False,
        regex=True,
    )
    market_report_mask = headline_series.str.contains(r"\bmarket\b", case=False, na=False, regex=True) & (
        headline_series.str.contains(r"\bglobal\b", case=False, na=False, regex=True)
        | headline_series.str.contains(r"\b20\d{2}\b", case=False, na=False, regex=True)
    )

    reputable_outlet_pattern = r"(?<!\w)(?:%s)(?!\w)" % "|".join(map(re.escape, outlet_names))
    reputable_outlet_mask = outlet_series.str.contains(
        reputable_outlet_pattern,
        case=False,
        na=False,
        regex=True,
    )

    aggregator_mask = outlet_series.str.contains(
        "|".join(re.escape(name) for name in aggregators_list),
        case=False,
        na=False,
        regex=True,
    )

    user_generated_mask = df["URL"].str.contains(
        "|".join(re.escape(domain) for domain in user_generated_domains),
        case=False,
        na=False,
        regex=True,
    )

    df.loc[market_report_mask, "Market Report Flag"] = "Market Report Spam"
    source_press_release_mask = _original_type_is_press_release(df)
    df.loc[(newswire_mask & ~market_report_mask) | source_press_release_mask, "Newswire Flag"] = "Press Release"
    df.loc[~newswire_mask & ~market_report_mask & financial_outlet_mask, "Financial Outlet Flag"] = "Financial Outlet"
    df.loc[~newswire_mask & ~market_report_mask & advertorial_mask, "Advertorial Flag"] = "Advertorial"
    # Disabled for now: snippet-only advertorial hints are too noisy to surface as a live flag.
    # df.loc[~newswire_mask & ~advertorial_mask & possible_advertorial_mask, "Possible Advertorial Flag"] = "Possible Advertorial?"
    df.loc[aggregator_mask, "Aggregator Flag"] = "Aggregator"
    df.loc[user_generated_mask, "User-Generated Flag"] = "User-Generated"

    df.loc[
        ~newswire_mask & ~advertorial_mask & ~market_report_mask & ~financial_outlet_mask & reputable_outlet_mask,
        "Good Outlet Flag",
    ] = "Good Outlet"

    def combine_flags(row):
        ordered_flags = [
            row.get("Newswire Flag", ""),
            row.get("Advertorial Flag", ""),
            row.get("Possible Advertorial Flag", ""),
            row.get("Good Outlet Flag", ""),
            row.get("Market Report Flag", ""),
            row.get("Financial Outlet Flag", ""),
            row.get("Aggregator Flag", ""),
            row.get("User-Generated Flag", ""),
        ]
        cleaned = [str(flag).strip() for flag in ordered_flags if str(flag or "").strip()]
        return " | ".join(cleaned)

    df["Coverage Flags"] = df.apply(combine_flags, axis=1)


    flag_columns = [
        "Newswire Flag", "Advertorial Flag", "Possible Advertorial Flag", "Good Outlet Flag",
        "Market Report Flag", "Financial Outlet Flag", "Aggregator Flag",
        "User-Generated Flag"
    ]
    df.drop(columns=[c for c in flag_columns if c in df.columns], inplace=True)

    return df
