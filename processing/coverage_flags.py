# coverage_flags.py

import pandas as pd
import re

def extract_relevant_text(snippet: str) -> str:
    words = str(snippet or "").split()
    if len(words) > 250:
        return " ".join(words[:125] + words[-125:])
    return str(snippet or "")

def add_coverage_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    newswire_phrases = [
        "pressrelease", "accesswire", "business wire", "businesswire", "CNW",
        "newswire", "presswire", "openPR", "pr-gateway", "Prlog", "PRWEB",
        "Pressebox", "Presseportal", "RTTNews", "SBWIRE", "issuewire", "prunderground"
    ]

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

    newswire_mask = df["Snippet_Limited"].str.contains(
        "|".join(re.escape(phrase) for phrase in newswire_phrases),
        case=False,
        na=False,
        regex=True,
    )

    newswire_author_pattern = r"newswire|press\s*release|distribution|newsfile"

    newswire_mask = (
        newswire_mask
        | df["Outlet"].str.contains("EurekAlert", case=False, na=False)
        | df["URL"].str.contains(r"/pr\.|news-release|press-release|newswise\.com", case=False, na=False, regex=True)
        | df["Author"].str.contains(newswire_author_pattern, case=False, na=False, regex=True)
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

    financial_outlet_mask = df["Outlet"].str.contains(
        "|".join(re.escape(phrase) for phrase in stock_moves_phrases),
        case=False,
        na=False,
        regex=True,
    )
    market_report_mask = (
        df["Headline"].str.contains(r"\bglobal\b", case=False, na=False, regex=True)
        & df["Headline"].str.contains(r"\bmarket\b", case=False, na=False, regex=True)
    )

    reputable_outlet_mask = df["Outlet"].str.contains(
        "|".join(map(re.escape, outlet_names)),
        case=False,
        na=False,
    )

    aggregator_mask = df["Outlet"].str.contains(
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

    df.loc[newswire_mask, "Newswire Flag"] = "Press Release"
    df.loc[~newswire_mask & market_report_mask, "Market Report Flag"] = "Market Report Spam"
    df.loc[~newswire_mask & ~market_report_mask & financial_outlet_mask, "Financial Outlet Flag"] = "Financial Outlet"
    df.loc[~newswire_mask & advertorial_mask, "Advertorial Flag"] = "Advertorial"
    # Disabled for now: snippet-only advertorial hints are too noisy to surface as a live flag.
    # df.loc[~newswire_mask & ~advertorial_mask & possible_advertorial_mask, "Possible Advertorial Flag"] = "Possible Advertorial?"
    df.loc[aggregator_mask, "Aggregator Flag"] = "Aggregator"
    df.loc[user_generated_mask, "User-Generated Flag"] = "User-Generated"

    df.loc[
        ~newswire_mask & ~advertorial_mask & ~market_report_mask & ~financial_outlet_mask & reputable_outlet_mask,
        "Good Outlet Flag",
    ] = "Good Outlet"

    def combine_flags(row):
        if row.get("Newswire Flag"):
            return row["Newswire Flag"]
        elif row.get("Advertorial Flag"):
            return row["Advertorial Flag"]
        elif row.get("Good Outlet Flag"):
            return row["Good Outlet Flag"]
        elif row.get("Possible Advertorial Flag"):
            return row["Possible Advertorial Flag"]
        elif row.get("Aggregator Flag"):
            return row["Aggregator Flag"]
        elif row.get("User-Generated Flag"):
            return row["User-Generated Flag"]
        elif row.get("Market Report Flag"):
            return row["Market Report Flag"]
        elif row.get("Financial Outlet Flag"):
            return row["Financial Outlet Flag"]
        return ""

    df["Coverage Flags"] = df.apply(combine_flags, axis=1)


    flag_columns = [
        "Newswire Flag", "Advertorial Flag", "Possible Advertorial Flag", "Good Outlet Flag",
        "Market Report Flag", "Financial Outlet Flag", "Aggregator Flag",
        "User-Generated Flag"
    ]
    df.drop(columns=[c for c in flag_columns if c in df.columns], inplace=True)

    return df
