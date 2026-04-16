from __future__ import annotations

import html
import importlib
import urllib.parse
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import streamlit as st
import processing.author_outlets as author_outlets_module
from ui.insight_blocks import build_linked_example_blocks_html

from processing.author_insights import (
    DEFAULT_AUTHOR_SUMMARY_MODEL,
    build_author_headline_table,
    build_author_metrics,
    generate_author_summary,
    init_author_insights_state,
)
from processing.missing_authors import (
    apply_author_fix,
    build_fixable_headline_table,
    build_last_author_fix_payload,
    fixable_headline_stats,
    get_available_visible_flags,
    get_headline_authors,
    get_possible_authors,
    init_missing_authors_state,
    prepare_author_working_df,
    undo_last_author_fix,
)
from utils.dataframe_helpers import top_x_by_mentions

author_outlets = importlib.reload(author_outlets_module)

FORMAT_DICT = author_outlets.FORMAT_DICT
apply_author_name_fix = author_outlets.apply_author_name_fix
assign_outlet = author_outlets.assign_outlet
build_author_outlet_cache_entry = author_outlets.build_author_outlet_cache_entry
build_auth_outlet_table = author_outlets.build_auth_outlet_table
build_outlet_assignment_payload = author_outlets.build_outlet_assignment_payload
find_strict_auto_assign_outlet = author_outlets.find_strict_auto_assign_outlet
get_auth_outlet_todo = author_outlets.get_auth_outlet_todo
get_author_search_urls = author_outlets.get_author_search_urls
init_author_outlet_prefetch_state = author_outlets.init_author_outlet_prefetch_state
make_author_cache_key = author_outlets.make_author_cache_key
init_author_outlets_state = author_outlets.init_author_outlets_state
prepare_traditional_for_author_outlets = author_outlets.prepare_traditional_for_author_outlets
reset_outlet_skips = author_outlets.reset_outlet_skips
undo_last_outlet_assignment = author_outlets.undo_last_outlet_assignment

warnings.filterwarnings("ignore")

METRIC_FIELD_MAP = {
    "Mentions": "Mention_Total",
    "Impressions": "Impressions",
    "Effective Reach": "Effective_Reach",
}


def format_compact_integer(num: float | int) -> str:
    try:
        n = float(num)
    except Exception:
        return str(num)

    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.0f}B"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.0f}M"
    if n >= 1_000:
        return f"{n / 1_000:.0f}K"
    return str(int(n)) if n.is_integer() else str(round(n))


def render_authors_page() -> None:
    st.title("Authors")
    st.caption("Resolve missing authors, assign primary outlets, and curate final key authors and insights.")

    if not st.session_state.get("standard_step", False):
        st.error("Please complete Basic Cleaning before trying this step.")
        st.stop()

    init_missing_authors_state(st.session_state)
    init_author_outlets_state(st.session_state)
    init_author_outlet_prefetch_state(st.session_state)
    init_author_insights_state(st.session_state)
    st.session_state.setdefault("authors_section", "Missing")

    if st.session_state.get("pickle_load", False) is True and len(st.session_state.get("auth_outlet_table", [])) > 0:
        st.session_state.auth_outlet_table = st.session_state.auth_outlet_table.copy()
        st.session_state.auth_outlet_table["Outlet"] = st.session_state.auth_outlet_table["Outlet"].replace([None], "").fillna("")

    st.session_state.df_traditional = prepare_traditional_for_author_outlets(st.session_state.df_traditional)

    def rebuild_author_outlet_state() -> None:
        existing = st.session_state.auth_outlet_table.copy() if len(st.session_state.get("auth_outlet_table", [])) > 0 else None
        st.session_state.auth_outlet_table = build_auth_outlet_table(
            st.session_state.df_traditional.copy(),
            st.session_state.get("top_auths_by", "Mentions"),
            existing_assignments=existing,
        )

    rebuild_author_outlet_state()

    def get_cached_author_outlet_entry(author_name: str, force_refresh: bool = False) -> dict:
        cache = st.session_state.author_outlet_api_cache
        cache_key = make_author_cache_key(author_name)

        if not force_refresh and cache_key in cache:
            return cache[cache_key]

        entry = build_author_outlet_cache_entry(
            author_name=author_name,
            df_traditional=st.session_state.df_traditional,
            secrets=st.secrets,
        )
        cache[cache_key] = entry
        return entry

    def invalidate_author_outlet_cache(author_names: list[str] | None = None) -> None:
        if author_names is None:
            st.session_state.author_outlet_api_cache = {}
            return

        cache = st.session_state.author_outlet_api_cache
        for author_name in author_names:
            cache.pop(make_author_cache_key(author_name), None)

    def get_prefetch_target_authors(auth_outlet_todo: pd.DataFrame) -> list[str]:
        if auth_outlet_todo is None or auth_outlet_todo.empty:
            return []

        working = auth_outlet_todo.copy()
        for col in ["Author", "Mentions", "Impressions"]:
            if col not in working.columns:
                working[col] = 0 if col != "Author" else ""

        working["Author"] = working["Author"].fillna("").astype(str).str.strip()
        working = working[working["Author"] != ""].copy()
        if working.empty:
            return []

        by_mentions = (
            working.sort_values(["Mentions", "Impressions"], ascending=False)["Author"]
            .head(25)
            .tolist()
        )
        by_impressions = (
            working.sort_values(["Impressions", "Mentions"], ascending=False)["Author"]
            .head(25)
            .tolist()
        )

        target_authors = list(dict.fromkeys(by_mentions + by_impressions))

        current_index = int(st.session_state.get("auth_outlet_skipped", 0) or 0)
        if 0 <= current_index < len(working):
            current_batch = working.iloc[current_index: current_index + 10]["Author"].tolist()
            current_batch = [author for author in current_batch if make_author_cache_key(author) not in st.session_state.author_outlet_api_cache]
            target_authors = list(dict.fromkeys(target_authors + current_batch))

        return target_authors

    def prefetch_author_outlet_matches(auth_outlet_todo: pd.DataFrame, auto_assign: bool = False) -> dict:
        target_authors = get_prefetch_target_authors(auth_outlet_todo)
        cache = st.session_state.author_outlet_api_cache
        missing_authors = [author for author in target_authors if make_author_cache_key(author) not in cache]

        loaded_now = 0
        if missing_authors:
            max_workers = min(6, len(missing_authors))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        build_author_outlet_cache_entry,
                        author,
                        st.session_state.df_traditional,
                        st.secrets,
                    ): author
                    for author in missing_authors
                }
                for future in as_completed(futures):
                    author_name = futures[future]
                    cache[make_author_cache_key(author_name)] = future.result()
                    loaded_now += 1

        auto_assigned_now = []
        if auto_assign:
            for author_name in target_authors:
                if author_name not in set(auth_outlet_todo["Author"].tolist()):
                    continue
                entry = cache.get(make_author_cache_key(author_name))
                if not entry:
                    continue
                matched_outlet = find_strict_auto_assign_outlet(entry)
                if not matched_outlet:
                    continue
                current_outlet = st.session_state.auth_outlet_table.loc[
                    st.session_state.auth_outlet_table["Author"] == author_name, "Outlet"
                ]
                if len(current_outlet) > 0 and str(current_outlet.iloc[0]).strip():
                    continue
                st.session_state.auth_outlet_table = assign_outlet(
                    st.session_state.auth_outlet_table,
                    author_name,
                    matched_outlet,
                )
                auto_assigned_now.append({
                    "Author": author_name,
                    "Outlet": matched_outlet,
                })

        if auto_assigned_now:
            existing_rows = st.session_state.get("author_outlet_auto_assigned_rows", [])
            existing_map = {(row["Author"], row["Outlet"]) for row in existing_rows}
            for row in auto_assigned_now:
                if (row["Author"], row["Outlet"]) not in existing_map:
                    existing_rows.append(row)
            st.session_state.author_outlet_auto_assigned_rows = existing_rows

        summary = {
            "requested": len(target_authors),
            "loaded_now": loaded_now,
            "cached_total": len([author for author in target_authors if make_author_cache_key(author) in cache]),
            "auto_assigned_now": len(auto_assigned_now),
        }
        st.session_state.author_outlet_prefetch_summary = summary
        return summary

    def render_missing_authors_tab() -> None:
        st.session_state.authors_section = "Missing"
        if len(st.session_state.get("df_traditional", [])) == 0:
            st.info("No traditional media in data. Skip to the next section.")
            return

        hide_table_row_index = """
            <style>
            tbody th {display:none}
            .blank {display:none}
            </style>
        """
        st.markdown(hide_table_row_index, unsafe_allow_html=True)

        base_working_df = st.session_state.df_traditional.copy()
        visible_flags, visible_defaults = get_available_visible_flags(base_working_df)

        excluded_flags = st.multiselect(
            "Exclude coverage flags",
            options=visible_flags,
            default=visible_defaults,
            help="Exclude selected flagged coverage from the missing-author workflow.",
            key="authors_missing_excluded_flags",
        )

        author_working_df = prepare_author_working_df(
            st.session_state.df_traditional,
            excluded_flags=excluded_flags,
        )

        headline_table = build_fixable_headline_table(author_working_df)
        counter = st.session_state.auth_skip_counter
        reviewed = st.session_state.auth_reviewed_count

        if len(headline_table) == 0:
            st.success("No fixable missing-author headlines remain in the current filtered view.")
        elif counter < len(headline_table):
            headline_text = headline_table.iloc[counter]["Headline"]
            encoded_headline = urllib.parse.quote(f'"{headline_text}"')
            google_search_url = f"https://www.google.com/search?q={encoded_headline}"

            headline_authors_df = get_headline_authors(author_working_df, headline_text).copy()
            possibles = get_possible_authors(author_working_df, headline_text)

            if st.session_state.get("authors_missing_current_headline") != headline_text:
                st.session_state.authors_missing_current_headline = headline_text
                st.session_state.authors_missing_select_author = possibles[0] if possibles else ""
                st.session_state.authors_missing_manual_author = ""
            elif st.session_state.get("authors_missing_select_author") not in possibles:
                st.session_state.authors_missing_select_author = possibles[0] if possibles else ""

            but1, col3, but2, but4 = st.columns(4)

            with but1:
                if st.button("Skip to Next Headline", key="authors_missing_skip"):
                    st.session_state.auth_skip_counter = counter + 1
                    st.rerun()

            with col3:
                if counter > 0:
                    st.write(f"Skipped: {counter}")

            with but2:
                if counter > 0 and st.button("Reset Skip Counter", key="authors_missing_reset"):
                    st.session_state.auth_skip_counter = 0
                    st.rerun()

            with but4:
                if st.button(
                    "Undo Last Author Update",
                    key="authors_missing_undo",
                    disabled=st.session_state.get("last_author_fix") is None,
                ):
                    undo_last_author_fix(st.session_state)
                    st.rerun()

            col1, col2, col3 = st.columns([12, 1, 9])

            with col1:
                st.subheader("Headline")
                st.table(headline_table.iloc[[counter]])
                st.markdown(
                    f'&nbsp;&nbsp;» <a href="{google_search_url}" target="_blank" style="text-decoration:underline; color:lightblue;">Search Google for this headline</a>',
                    unsafe_allow_html=True,
                )

            with col2:
                st.write(" ")

            with col3:
                st.subheader("Authors in CSV")
                st.table(headline_authors_df)

            with st.form("authors_missing_fix_form", clear_on_submit=True):
                col1, col2, col3 = st.columns([8, 1, 8])

                with col1:
                    box_author = st.selectbox(
                        "Pick from possible Authors",
                        possibles,
                        key="authors_missing_select_author",
                        help="Pick from one of the authors already associated with this headline.",
                    )
                with col2:
                    st.write(" ")
                    st.subheader("OR")
                with col3:
                    string_author = st.text_input(
                        "Write in the author name",
                        key="authors_missing_manual_author",
                        help="Override above selection by writing in a custom name.",
                    )

                submitted = st.form_submit_button("Update Author", type="primary")

            if submitted:
                new_author = string_author.strip() if string_author.strip() else box_author
                if not new_author:
                    st.warning("Please choose or enter an author name.")
                else:
                    st.session_state.last_author_fix = build_last_author_fix_payload(
                        st.session_state.df_traditional,
                        headline_text,
                        previous_reviewed_count=st.session_state.auth_reviewed_count,
                    )
                    st.session_state.df_traditional = apply_author_fix(
                        st.session_state.df_traditional,
                        headline_text,
                        new_author,
                    )
                    invalidate_author_outlet_cache()
                    st.session_state.auth_reviewed_count = reviewed + 1
                    rebuild_author_outlet_state()
                    st.rerun()
        else:
            st.info("You've reached the end of the list!")

            top_end_col1, top_end_col2 = st.columns([1, 1])

            with top_end_col1:
                if counter > 0 and st.button("Reset Counter", key="authors_missing_reset_end"):
                    st.session_state.auth_skip_counter = 0
                    st.rerun()

            with top_end_col2:
                if st.button(
                    "Undo Last Author Update",
                    key="authors_missing_undo_end",
                    disabled=st.session_state.get("last_author_fix") is None,
                ):
                    undo_last_author_fix(st.session_state)
                    st.rerun()

            if counter == 0:
                st.success("✓ Nothing left to update here.")

        st.divider()

        col1, col2, col3 = st.columns(3)
        with col1:
            st.write("**Original Top Authors**")
            media_type_column = "Type" if "Type" in st.session_state.df_untouched.columns else "Media Type"
            filtered_df = st.session_state.df_untouched[
                st.session_state.df_untouched[media_type_column].isin(["PRINT", "ONLINE_NEWS", "ONLINE", "BLOGS", "PRESS_RELEASE"])
            ].copy()
            if "Mentions" not in filtered_df.columns:
                filtered_df["Mentions"] = 1
            st.dataframe(top_x_by_mentions(filtered_df, "Author"), use_container_width=True, hide_index=True)
        with col2:
            st.write("**Current Top Authors**")
            st.dataframe(top_x_by_mentions(author_working_df, "Author"), use_container_width=True, hide_index=True)
        with col3:
            st.write("**Fixable Author Stats**")
            remaining = fixable_headline_stats(
                author_working_df,
                counter=st.session_state.auth_skip_counter,
                primary="Headline",
                secondary="Author",
            )
            st.metric("Reviewed", len(headline_table) - remaining["remaining"] + reviewed if len(headline_table) > 0 else reviewed)
            st.metric("Updated", reviewed)
            st.metric("Remaining in this view", remaining["remaining"])

    def render_author_outlets_tab() -> None:
        st.session_state.authors_section = "Outlets"
        hide_table_row_index = """
            <style>
            tbody th {display:none}
            .blank {display:none}
            .row_heading.level0 {width:0; display:none}
            </style>
        """
        st.markdown(hide_table_row_index, unsafe_allow_html=True)

        control_col1, control_col2 = st.columns([1, 1], gap="medium")
        with control_col1:
            st.session_state.top_auths_by = st.selectbox(
                "Top Authors by:",
                ["Mentions", "Impressions", "Effective Reach"],
                key="authors_outlets_rank_by",
                on_change=lambda: reset_outlet_skips(st.session_state),
            )
        with control_col2:
            st.write("")
            auto_assign_requested = st.button(
                "Auto-assign perfect matches",
                key="authors_outlets_auto_assign",
                help="Assign only when there is exactly one clear overlap between coverage outlets and API result outlets in the prefetched author set.",
                use_container_width=True,
            )

        rebuild_author_outlet_state()
        auth_outlet_todo = get_auth_outlet_todo(st.session_state.auth_outlet_table)
        prefetch_summary = prefetch_author_outlet_matches(auth_outlet_todo, auto_assign=auto_assign_requested)

        if prefetch_summary.get("auto_assigned_now", 0) > 0:
            st.success(f"Auto-assigned {prefetch_summary['auto_assigned_now']} perfect match(es) from the prefetched author set.")
            st.rerun()

        auto_assigned_rows = st.session_state.get("author_outlet_auto_assigned_rows", [])
        if auto_assigned_rows:
            with st.expander("Auto-assigned this session", expanded=False):
                st.dataframe(pd.DataFrame(auto_assigned_rows), use_container_width=True, hide_index=True)

        if st.session_state.auth_outlet_skipped < len(auth_outlet_todo):
            original_author_name = auth_outlet_todo.iloc[st.session_state.auth_outlet_skipped]["Author"]

            if st.session_state.get("last_author_for_fix") != original_author_name:
                st.session_state.author_fix_input = original_author_name
                st.session_state.last_author_for_fix = original_author_name

            def apply_author_fix_callback() -> None:
                new_name = st.session_state.author_fix_input.strip()
                old_name = original_author_name
                if not new_name:
                    return
                if new_name != old_name:
                    apply_author_name_fix(st.session_state, old_name, new_name)
                    invalidate_author_outlet_cache([old_name, new_name])

            with st.expander("Author name fix tools", expanded=False):
                st.text_input(
                    "Correct author name",
                    key="author_fix_input",
                    on_change=apply_author_fix_callback,
                    help="Edit the name and press Enter to apply the correction to all matching rows.",
                )
                st.caption(
                    "This updates every instance of this author in the cleaned dataset and refreshes the author-outlet workflow."
                )

            header_col, skip_col, reset_col, undo_col = st.columns([2, 1, 1, 1])

            with header_col:
                st.markdown(
                    f"""
                    <h2 style="color: goldenrod; padding-top:0!important; margin-top:0;">
                        {original_author_name}
                    </h2>
                    """,
                    unsafe_allow_html=True,
                )

            with skip_col:
                st.write(" ")
                if st.button("Skip to Next Author", key="authors_outlets_skip"):
                    st.session_state.auth_outlet_skipped += 1
                    st.rerun()

            with reset_col:
                st.write(" ")
                if st.button("Reset Skips", key="authors_outlets_reset"):
                    st.session_state.auth_outlet_skipped = 0
                    st.rerun()

            with undo_col:
                st.write(" ")
                if st.button(
                    "Undo Last Outlet",
                    key="authors_outlets_undo",
                    disabled=st.session_state.get("last_outlet_assignment") is None,
                    help="Removes the most recently assigned outlet and returns that author to the queue.",
                ):
                    undo_last_outlet_assignment(st.session_state)
                    st.rerun()

            match_author_name = original_author_name
            cache_entry = get_cached_author_outlet_entry(original_author_name)
            api_debug = cache_entry.get("api_debug", {})

            def name_match(series):
                non_match = "color: #985331;"
                match = "color: goldenrod"
                return [non_match if cell_value != match_author_name else match for cell_value in series]

            outlets_in_coverage = cache_entry.get("outlets_in_coverage", pd.DataFrame())
            outlets_in_coverage_list = cache_entry.get("outlets_in_coverage_list", pd.Index(["Freelance"]))
            matched_authors = cache_entry.get("matched_authors", pd.DataFrame())
            db_outlets = cache_entry.get("db_outlets", [])
            possibles = cache_entry.get("possibles", [])

            form_block = st.container()
            info_block = st.container()

            with info_block:
                col1, col2, col3 = st.columns([8, 1, 16])

                with col1:
                    st.subheader("Outlets in CSV")

                    outlets_in_coverage_styled = outlets_in_coverage.style.apply(
                        lambda x: [
                            "background-color: goldenrod; color: black" if v in db_outlets else ""
                            for v in x
                        ],
                        axis=1,
                        subset="Outlet",
                    )

                    st.dataframe(
                        outlets_in_coverage_styled,
                        use_container_width=True,
                        hide_index=True,
                    )

                with col2:
                    st.write(" ")

                with col3:
                    st.subheader("Media Database Results")

                    show_debug = not api_debug.get("ok") or api_debug.get("error")

                    coverage_outlet_values = outlets_in_coverage["Outlet"].tolist()

                    if len(matched_authors) == 0:
                        st.warning("NO MATCH FOUND")

                        if show_debug:
                            with st.expander("API debug details", expanded=False):
                                st.write("Status code:", api_debug.get("status_code"))
                                st.write("Request ok:", api_debug.get("ok"))
                                st.write("Error:", api_debug.get("error"))
                                st.write("JSON keys:", api_debug.get("json_keys"))
                                st.write("Response preview:")
                                st.code(api_debug.get("response_text_preview", ""), language="json")
                    else:
                        matched_authors_display = matched_authors.copy()
                        preferred_cols = [c for c in ["Name", "Outlet", "Country", "Title"] if c in matched_authors_display.columns]
                        matched_authors_display = matched_authors_display[preferred_cols]

                        styled_matches = (
                            matched_authors_display.style
                            .apply(
                                lambda col: [
                                    "background-color: goldenrod; color: black"
                                    if v in coverage_outlet_values else ""
                                    for v in col
                                ],
                                subset=["Outlet"],
                                axis=0,
                            )
                            .apply(name_match, axis=0, subset=["Name"])
                        )

                        st.dataframe(
                            styled_matches,
                            use_container_width=True,
                            hide_index=True,
                        )

                    muckrack_url, linkedin_url = get_author_search_urls(match_author_name)
                    st.markdown(
                        f'&nbsp;&nbsp;» <a href="{muckrack_url}" target="_blank" style="text-decoration:underline; color:lightblue;">Search Muckrack for {match_author_name}</a>',
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f'&nbsp;&nbsp;» <a href="{linkedin_url}" target="_blank" style="text-decoration:underline; color:lightblue;">Search LinkedIn for {match_author_name}</a>',
                        unsafe_allow_html=True,
                    )

            with form_block:
                with st.form("authors_outlet_assign_form", clear_on_submit=True):
                    col1, col2, col3 = st.columns([8, 1, 8])

                    with col1:
                        if len(matched_authors) > 0:
                            box_outlet = st.selectbox(
                                "Pick outlet from DATABASE MATCHES",
                                possibles,
                                key="authors_outlet_pick",
                                help="Pick from one of the outlets associated with this author name.",
                            )
                        else:
                            box_outlet = st.selectbox(
                                'Pick outlet from COVERAGE or "Freelance"',
                                outlets_in_coverage_list,
                                key="authors_outlet_pick_fallback",
                            )

                    with col2:
                        st.write(" ")
                        st.subheader("OR")

                    with col3:
                        string_outlet = st.text_input(
                            "Write in an outlet name",
                            key="authors_outlet_manual",
                            help="Override the selection by writing a custom outlet name.",
                        )

                    submitted = st.form_submit_button("Assign Outlet", type="primary")

            if submitted:
                new_outlet = string_outlet.strip() if len(string_outlet.strip()) > 0 else box_outlet
                st.session_state.last_outlet_assignment = build_outlet_assignment_payload(
                    st.session_state.auth_outlet_table,
                    original_author_name,
                    previous_skip=st.session_state.auth_outlet_skipped,
                )
                st.session_state.auth_outlet_table = assign_outlet(
                    st.session_state.auth_outlet_table,
                    original_author_name,
                    new_outlet,
                )
                st.rerun()
        else:
            st.info("You've reached the end of the list!")
            st.write(f"Authors skipped: {st.session_state.auth_outlet_skipped}")

            if st.session_state.auth_outlet_skipped > 0:
                if st.button("Reset Counter", key="authors_outlets_reset_end"):
                    st.session_state.auth_outlet_skipped = 0
                    st.rerun()
            else:
                st.write("✓ Nothing left to update here.")

        st.divider()

        bottom_col1, bottom_col2, bottom_col3 = st.columns([8, 1, 4])

        with bottom_col1:
            st.subheader("Top Authors")
            table_df = st.session_state.auth_outlet_table[["Author", "Outlet", "Mentions", "Impressions", "Effective Reach"]].copy().fillna("")
            if st.session_state.top_auths_by == "Mentions":
                table_df = table_df.sort_values(["Mentions", "Impressions", "Effective Reach"], ascending=False).head(15)
            elif st.session_state.top_auths_by == "Impressions":
                table_df = table_df.sort_values(["Impressions", "Mentions", "Effective Reach"], ascending=False).head(15)
            else:
                table_df = table_df.sort_values(["Effective Reach", "Impressions", "Mentions"], ascending=False).head(15)
            st.dataframe(table_df.style.format(FORMAT_DICT, na_rep=" "), use_container_width=True, hide_index=True)

        with bottom_col2:
            st.write(" ")

        with bottom_col3:
            st.subheader("Outlets assigned")
            assigned = len(st.session_state.auth_outlet_table.loc[st.session_state.auth_outlet_table["Outlet"] != ""])
            st.metric(label="Assigned", value=assigned)

    def render_author_insights_tab(mode: str = "selection") -> None:
        st.session_state.authors_section = "Selection" if mode == "selection" else "Insights"
        rebuild_author_outlet_state()
        author_metrics, author_story_rows = build_author_metrics(
            st.session_state.df_traditional,
            auth_outlet_table=st.session_state.auth_outlet_table,
        )

        if author_metrics.empty:
            st.info("No author data available yet.")
            return

        st.session_state.author_insights_target_count = 10
        candidate_limit = 50
        st.session_state.setdefault("author_selection_assigned_only", False)

        rank_map = {
            "Mentions": ["Mention_Total", "Impressions", "Unique_Stories"],
            "Impressions": ["Impressions", "Mention_Total", "Unique_Stories"],
        }
        sort_cols = rank_map.get(st.session_state.get("top_auths_by", "Mentions"), rank_map["Mentions"])
        ranked_df = author_metrics.copy().sort_values(sort_cols, ascending=False).reset_index(drop=True)

        valid_authors = ranked_df["Author"].tolist()
        if "authors_insights_active_author" not in st.session_state or st.session_state["authors_insights_active_author"] not in valid_authors:
            st.session_state["authors_insights_active_author"] = valid_authors[0]
        if "authors_insights_inspect_author_current" not in st.session_state or st.session_state["authors_insights_inspect_author_current"] not in valid_authors:
            st.session_state["authors_insights_inspect_author_current"] = st.session_state["authors_insights_active_author"]
        if "authors_insights_inspect_author_split" not in st.session_state or st.session_state["authors_insights_inspect_author_split"] not in valid_authors:
            st.session_state["authors_insights_inspect_author_split"] = st.session_state["authors_insights_active_author"]

        def sync_active_author(widget_key: str) -> None:
            selected = st.session_state.get(widget_key)
            if selected in valid_authors:
                st.session_state["authors_insights_active_author"] = selected
                st.session_state["authors_insights_inspect_author_current"] = selected
                st.session_state["authors_insights_inspect_author_split"] = selected

        inspect_author = st.session_state["authors_insights_active_author"]
        inspect_row = ranked_df.loc[ranked_df["Author"] == inspect_author].iloc[0]
        headline_table = build_author_headline_table(author_story_rows, inspect_author, limit=5)

        valid_options = ranked_df["Author"].tolist()
        current_selected = [
            author for author in st.session_state.get("author_insights_selected_authors", [])
            if author in valid_options
        ]
        if current_selected != st.session_state.get("author_insights_selected_authors", []):
            st.session_state.author_insights_selected_authors = current_selected

        candidate_df = ranked_df[~ranked_df["Author"].isin(current_selected)].copy()
        if st.session_state.get("author_selection_assigned_only", False):
            candidate_df = candidate_df[candidate_df["Assigned Outlet"].fillna("").astype(str).str.strip() != ""].copy()
        suggested_authors = candidate_df.head(int(st.session_state.author_insights_target_count))["Author"].tolist()
        candidate_df = candidate_df.head(candidate_limit)[[
            "Author",
            "Assigned Outlet",
            "Mention_Total",
            "Impressions",
            "Syndicated_Pickups",
            "Good_Outlet_Stories",
            "Unique_Stories",
        ]].copy()
        candidate_df["Good Outlet Rate"] = (
            candidate_df["Good_Outlet_Stories"] / candidate_df["Unique_Stories"].replace(0, pd.NA)
        ).fillna(0.0) * 100
        candidate_df["Syndication Rate"] = (
            candidate_df["Syndicated_Pickups"] / candidate_df["Unique_Stories"].replace(0, pd.NA)
        ).fillna(0.0) * 100
        candidate_df["Keep"] = False

        selected_authors = st.session_state.get("author_insights_selected_authors", [])
        shortlist_df = ranked_df[ranked_df["Author"].isin(selected_authors)].copy()
        shortlist_df["SortOrder"] = shortlist_df["Author"].map({name: idx for idx, name in enumerate(selected_authors)})
        shortlist_df = shortlist_df.sort_values("SortOrder").drop(columns=["SortOrder"])

        summary_store = st.session_state.get("author_insights_summaries", {})
        shortlist_df["Good Outlet Rate"] = (
            shortlist_df["Good_Outlet_Stories"] / shortlist_df["Unique_Stories"].replace(0, pd.NA)
        ).fillna(0.0) * 100
        shortlist_df["Syndication Rate"] = (
            shortlist_df["Syndicated_Pickups"] / shortlist_df["Unique_Stories"].replace(0, pd.NA)
        ).fillna(0.0) * 100
        shortlist_df["Coverage Themes"] = shortlist_df["Author"].map(lambda author: summary_store.get(author, ""))

        shortlist_view = shortlist_df[[
            "Author",
            "Assigned Outlet",
            "Mention_Total",
            "Impressions",
            "Good Outlet Rate",
            "Coverage Themes",
        ]].copy()
        shortlist_view["Delete"] = False
        shortlist_output_df = shortlist_df[[
            "Author",
            "Assigned Outlet",
            "Mention_Total",
            "Unique_Stories",
            "Impressions",
            "Effective_Reach",
            "Coverage Themes",
            "Good Outlet Rate",
        ]].copy()

        def build_story_examples_html(
            df: pd.DataFrame,
            show_outlet: bool = True,
            show_media_type: bool = True,
            show_mentions: bool = True,
            show_impressions: bool = True,
            show_effective_reach: bool = True,
        ) -> str:
            if df.empty:
                return ""
            items = []
            for _, row in df.iterrows():
                items.append(
                    {
                        "headline": row.get("Headline", ""),
                        "url": row.get("Representative URL", ""),
                        "outlet": row.get("Representative Outlet", ""),
                        "example_type": row.get("Type", ""),
                        "mentions": int(pd.to_numeric(pd.Series([row.get("Story Mentions", 0)]), errors="coerce").fillna(0).iloc[0]),
                        "impressions": int(pd.to_numeric(pd.Series([row.get("Story Impressions", 0)]), errors="coerce").fillna(0).iloc[0]),
                        "effective_reach": int(pd.to_numeric(pd.Series([row.get("Story Effective Reach", 0)]), errors="coerce").fillna(0).iloc[0]),
                    }
                )

            return build_linked_example_blocks_html(
                items,
                show_outlet=show_outlet,
                show_media_type=show_media_type,
                show_mentions=show_mentions,
                show_impressions=show_impressions,
                show_effective_reach=show_effective_reach,
            )

        def render_candidate_selection_table(include_syndication: bool, key_suffix: str) -> None:
            del include_syndication
            st.subheader("Candidate Authors")
            st.caption('Check the "Keep" box for authors you want on the final shortlist, then click "Save Selected".')
            st.checkbox(
                "Show only authors with assigned outlets",
                key="author_selection_assigned_only",
                help="Hide unassigned authors from the candidate table and make Top Suggestion use only authors with assigned outlets.",
            )
            working_df = candidate_df.copy()
            working_df["Author Outlet"] = working_df.apply(
                lambda row: " | ".join(
                    [part for part in [str(row.get("Author", "") or "").strip(), str(row.get("Assigned Outlet", "") or "").strip()] if part]
                ),
                axis=1,
            )
            preferred_order = [
                "Author Outlet",
                "Mention_Total",
                "Impressions",
                "Keep",
            ]
            display_df = working_df[[col for col in preferred_order if col in working_df.columns]].copy()
            candidate_editor = st.data_editor(
                display_df,
                key=f"authors_candidate_editor_{key_suffix}",
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Author Outlet": st.column_config.Column("Author | Outlet", width="medium"),
                    "Mention_Total": st.column_config.NumberColumn("Mentions", width="small", format="%d"),
                    "Impressions": st.column_config.NumberColumn("Impressions", width="small", format="%,d"),
                    "Keep": st.column_config.CheckboxColumn("Keep", width="small"),
                },
            )

            candidate_action1, candidate_action2, candidate_action3 = st.columns([1, 1, 2], gap="small")
            with candidate_action1:
                if st.button("Use Top Suggestion", key=f"authors_insights_use_suggestion_{key_suffix}"):
                    st.session_state.authors_section = "Selection"
                    st.session_state.author_insights_selected_authors = suggested_authors
                    st.session_state.author_insights_summaries = {
                        k: v for k, v in st.session_state.get("author_insights_summaries", {}).items()
                        if k in suggested_authors
                    }
                    st.rerun()
            with candidate_action2:
                if st.button("Clear Selected", key=f"authors_insights_clear_shortlist_{key_suffix}"):
                    st.session_state.authors_section = "Selection"
                    st.session_state.author_insights_selected_authors = []
                    st.session_state.author_insights_summaries = {}
                    st.rerun()
            with candidate_action3:
                if st.button("Save Selected", key=f"authors_insights_save_selected_{key_suffix}", type="primary"):
                    st.session_state.authors_section = "Selection"
                    author_lookup = {
                        row["Author Outlet"]: row["Author"]
                        for _, row in working_df[["Author", "Author Outlet"]].drop_duplicates().iterrows()
                    }
                    newly_selected = [
                        author_lookup.get(label, "")
                        for label in candidate_editor.loc[candidate_editor["Keep"], "Author Outlet"].tolist()
                    ]
                    newly_selected = [name for name in newly_selected if name]
                    selected = list(dict.fromkeys(current_selected + newly_selected))
                    st.session_state.author_insights_selected_authors = selected
                    st.session_state.author_insights_summaries = {
                        k: v for k, v in st.session_state.get("author_insights_summaries", {}).items()
                        if k in selected
                    }
                    st.rerun()
            st.caption(
                f"Selected {len(selected_authors)} author(s). Target: {int(st.session_state.author_insights_target_count)}."
            )

        def render_shortlist_editor(key_suffix: str) -> None:
            if not selected_authors:
                return
            st.write("**Current shortlist**")
            shortlist_editor = st.data_editor(
                shortlist_view,
                use_container_width=True,
                hide_index=True,
                key=f"authors_shortlist_editor_{key_suffix}",
                column_config={
                    "Mention_Total": st.column_config.NumberColumn("Mentions", width="small", format="%d"),
                    "Impressions": st.column_config.NumberColumn("Impressions", width="small", format="%,d"),
                    "Good Outlet Rate": st.column_config.NumberColumn("Good Outlet Rate", width="small", format="%.0f%%"),
                    "Coverage Themes": st.column_config.Column("Coverage Themes", width="large"),
                    "Delete": st.column_config.CheckboxColumn("Delete", width="small"),
                },
            )
            rows_to_delete = shortlist_editor[shortlist_editor["Delete"]].index.tolist()
            if rows_to_delete:
                st.session_state.authors_section = "Selection"
                authors_to_remove = shortlist_df.iloc[rows_to_delete]["Author"].tolist()
                remaining = [
                    author for author in st.session_state.get("author_insights_selected_authors", [])
                    if author not in authors_to_remove
                ]
                st.session_state.author_insights_selected_authors = remaining
                st.session_state.author_insights_summaries = {
                    k: v for k, v in st.session_state.get("author_insights_summaries", {}).items()
                    if k in remaining
                }
                st.rerun()

        def render_generate_button(key_suffix: str) -> None:
            if st.button("Generate brand-relevant coverage themes", type="primary", key=f"authors_generate_theme_summaries_{key_suffix}"):
                st.session_state.authors_section = "Insights"
                summaries = dict(summary_store)
                client_name = str(st.session_state.get("client_name", "")).strip()
                with st.spinner("Generating author theme summaries..."):
                    for author_name in selected_authors:
                        author_row = shortlist_df.loc[shortlist_df["Author"] == author_name].iloc[0]
                        headline_df = build_author_headline_table(author_story_rows, author_name, limit=8)
                        try:
                            summary_text, _, _ = generate_author_summary(
                                author_name=author_name,
                                client_name=client_name,
                                author_row=author_row,
                                headline_df=headline_df,
                                api_key=st.secrets["key"],
                                model=DEFAULT_AUTHOR_SUMMARY_MODEL,
                            )
                            summaries[author_name] = summary_text
                        except Exception as e:
                            summaries[author_name] = f"Could not generate summary: {e}"
                st.session_state.author_insights_summaries = summaries
                st.rerun()

        def build_report_html(
            show_mentions: bool,
            show_unique_mentions: bool,
            show_impressions: bool,
            show_effective_reach: bool,
            show_headline_examples: bool,
        ) -> str:
            blocks = []
            for _, row in shortlist_df.iterrows():
                author_name = str(row.get("Author", "") or "").strip()
                outlet = str(row.get("Assigned Outlet", "") or "").strip()
                themes = " ".join(str(row.get("Coverage Themes", "") or "").split())
                mentions = int(row.get("Mention_Total", 0) or 0)
                unique_mentions = int(row.get("Unique_Stories", 0) or 0)
                impressions = int(row.get("Impressions", 0) or 0)
                effective_reach = int(row.get("Effective_Reach", 0) or 0)

                header = f"<strong>{html.escape(author_name)}</strong>"
                if outlet:
                    header += f' <span style="opacity:0.82;">|</span> <span style="font-style:italic; opacity:0.92;">{html.escape(outlet)}</span>'

                metric_parts = []
                if show_mentions:
                    metric_parts.append(f"Mentions: {mentions:,}")
                if show_unique_mentions:
                    metric_parts.append(f"Unique Mentions: {unique_mentions:,}")
                if show_impressions:
                    metric_parts.append(f"Impressions: {impressions:,}")
                if show_effective_reach:
                    metric_parts.append(f"Effective Reach: {effective_reach:,}")
                metrics = " | ".join(metric_parts)
                metrics_html = (
                    f'<div style="font-size:0.84rem; opacity:0.72; letter-spacing:0.01em;">{html.escape(metrics)}</div>'
                    if metrics else ""
                )

                body_html = html.escape(themes) if themes else ""
                examples_html = ""
                if show_headline_examples and author_name:
                    headline_table = build_author_headline_table(author_story_rows, author_name, limit=5)
                    example_items = build_story_examples_html(
                        headline_table,
                        show_outlet=True,
                        show_media_type=True,
                        show_mentions=show_mentions,
                        show_impressions=show_impressions,
                        show_effective_reach=show_effective_reach,
                    )
                    if example_items:
                        examples_html = (
                            '<div style="margin-top:0.72rem;">'
                            '<div style="font-size:0.98rem; font-weight:600; opacity:0.78; margin-bottom:0.45rem;">Representative examples</div>'
                            f"{example_items}"
                            "</div>"
                        )
                block = (
                    '<div style="margin-bottom:1.35rem;">'
                    f'<div style="font-size:1.08rem; font-weight:700; margin-bottom:0.2rem;">{header}</div>'
                    f'<div style="line-height:1.55; margin-bottom:0.28rem;">{body_html}</div>'
                    f"{metrics_html}"
                    f"{examples_html}"
                    "</div>"
                )
                blocks.append(block)
            if not blocks:
                return ""
            return f'<div style="display:block;">{"".join(blocks)}</div>'

        if mode == "selection":
            left_col, right_col = st.columns([0.9, 1.35], gap="large")
            with left_col:
                st.subheader("Author Inspector")
                st.selectbox(
                    "Inspect author",
                    options=valid_authors,
                    key="authors_insights_inspect_author_split",
                    on_change=sync_active_author,
                    args=("authors_insights_inspect_author_split",),
                )
                inspect_author = st.session_state["authors_insights_active_author"]
                inspect_row = ranked_df.loc[ranked_df["Author"] == inspect_author].iloc[0]
                headline_table = build_author_headline_table(author_story_rows, inspect_author, limit=5)
                st.caption(
                    f"Assigned outlet: {inspect_row.get('Assigned Outlet', '') or 'Unassigned'} | "
                    f"Mentions: {int(inspect_row.get('Mention_Total', 0)):,} | "
                    f"Impressions: {int(inspect_row.get('Impressions', 0)):,}"
                )
                examples_html = build_story_examples_html(headline_table)
                if examples_html:
                    st.markdown(examples_html, unsafe_allow_html=True)
            with right_col:
                render_candidate_selection_table(include_syndication=True, key_suffix="split")

            if selected_authors:
                st.divider()
                render_shortlist_editor("split")
            return

        if mode == "insights":
            st.divider()
            st.subheader("Chart Table")
            metric_label = st.radio(
                "Chart metric",
                ["Mentions", "Impressions", "Effective Reach"],
                horizontal=True,
                key="authors_insights_chart_metric",
            )

            chart_table = shortlist_output_df[[
                "Author",
                "Assigned Outlet",
                "Mention_Total",
                "Unique_Stories",
                "Impressions",
                "Effective_Reach",
            ]].copy()
            chart_table = chart_table.rename(
                columns={
                    "Mention_Total": "Mentions",
                    "Unique_Stories": "Unique Mentions",
                    "Effective_Reach": "Effective Reach",
                }
            )
            chart_table = chart_table.sort_values(metric_label, ascending=False).reset_index(drop=True)

            chart_tab, table_tab = st.tabs(["Chart", "Table"])

            with chart_tab:
                try:
                    alt = importlib.import_module("altair")
                except Exception:
                    alt = None

                if alt is None:
                    st.info("Author chart unavailable in this environment.")
                else:
                    plot_df = chart_table.copy()
                    plot_df["Author Label"] = plot_df["Author"].astype(str).str.upper()
                    plot_df["Outlet Label"] = plot_df["Assigned Outlet"].astype(str).str.strip().replace("", "Unassigned")
                    plot_df["Author Axis Label"] = plot_df["Author Label"]
                    plot_df["Metric Label"] = plot_df[metric_label].apply(format_compact_integer)
                    chart_height = max(260, len(plot_df) * 48)
                    sort_order = plot_df["Author Axis Label"].tolist()
                    max_metric = float(plot_df[metric_label].max()) if not plot_df.empty else 0.0
                    padded_max = max_metric * 1.18 if max_metric > 0 else 1.0
                    compact_axis_expr = """
                        datum.value >= 1e9 ? format(datum.value / 1e9, '.0f') + 'B' :
                        datum.value >= 1e6 ? format(datum.value / 1e6, '.0f') + 'M' :
                        datum.value >= 1e3 ? format(datum.value / 1e3, '.0f') + 'K' :
                        format(datum.value, ',')
                    """

                    label_base = alt.Chart(plot_df).encode(
                        y=alt.Y(
                            "Author Axis Label:N",
                            sort=sort_order,
                            title=None,
                            axis=None,
                        ),
                    )

                    author_labels = label_base.mark_text(
                        align="right",
                        baseline="bottom",
                        dx=-8,
                        dy=-2,
                        fontSize=12,
                        fontWeight="bold",
                        color="#f3f4f6",
                    ).encode(
                        x=alt.value(250),
                        text=alt.Text("Author Label:N"),
                    )

                    outlet_labels = label_base.mark_text(
                        align="right",
                        baseline="top",
                        dx=-8,
                        dy=2,
                        fontSize=11,
                        color="#b8bfcc",
                    ).encode(
                        x=alt.value(250),
                        text=alt.Text("Outlet Label:N"),
                    )

                    label_panel = (
                        (author_labels + outlet_labels)
                        .properties(width=260, height=chart_height)
                    )

                    bars = alt.Chart(plot_df).mark_bar(color="#37415f", cornerRadiusEnd=2).encode(
                        y=alt.Y("Author Axis Label:N", sort=sort_order, axis=None),
                        x=alt.X(
                            f"{metric_label}:Q",
                            title=None,
                            axis=alt.Axis(labelExpr=compact_axis_expr, grid=True),
                            scale=alt.Scale(domain=[0, padded_max], nice=False),
                        ),
                        tooltip=[
                            alt.Tooltip("Author:N", title="Author"),
                            alt.Tooltip("Assigned Outlet:N", title="Outlet"),
                            alt.Tooltip(f"{metric_label}:Q", title=metric_label, format=","),
                        ],
                    )

                    value_labels = alt.Chart(plot_df).mark_text(
                        align="left",
                        baseline="middle",
                        dx=6,
                        color="#f3f4f6",
                        fontSize=11,
                    ).encode(
                        y=alt.Y("Author Axis Label:N", sort=sort_order),
                        x=alt.X(f"{metric_label}:Q"),
                        text=alt.Text("Metric Label:N"),
                    )

                    bar_panel = (
                        (bars + value_labels)
                        .properties(height=chart_height, width=440)
                    )

                    chart = (
                        alt.hconcat(label_panel, bar_panel, spacing=6)
                        .resolve_scale(y="shared")
                        .properties(title=f"Top Authors by {metric_label.lower()}")
                        .configure_view(strokeWidth=0)
                    )
                    st.altair_chart(chart, use_container_width=True)

            with table_tab:
                st.dataframe(
                    chart_table,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Mentions": st.column_config.NumberColumn("Mentions", width="small", format="%d"),
                        "Unique Mentions": st.column_config.NumberColumn("Unique Mentions", width="small", format="%d"),
                        "Impressions": st.column_config.NumberColumn("Impressions", width="small", format="%,d"),
                        "Effective Reach": st.column_config.NumberColumn("Effective Reach", width="small", format="%,d"),
                    },
                )

            st.divider()
            st.subheader("Report Copy")
            generate_col1, generate_col2 = st.columns([1.2, 3], gap="medium")
            with generate_col1:
                render_generate_button("insights")
            with generate_col2:
                st.caption("Uses shortlisted authors plus representative grouped stories from each author footprint to generate concise, report-ready coverage themes.")
            metrics_col1, metrics_col2, metrics_col3, metrics_col4, metrics_col5 = st.columns(5, gap="small")
            with metrics_col1:
                show_mentions = st.checkbox("Show mentions", value=True, key="authors_report_show_mentions")
            with metrics_col2:
                show_unique_mentions = st.checkbox("Show unique mentions", value=True, key="authors_report_show_unique_mentions")
            with metrics_col3:
                show_impressions = st.checkbox("Show impressions", value=True, key="authors_report_show_impressions")
            with metrics_col4:
                show_effective_reach = st.checkbox("Show effective reach", value=True, key="authors_report_show_effective_reach")
            with metrics_col5:
                show_headline_examples = st.checkbox("Show headline examples", value=True, key="authors_report_show_examples")

            report_html = build_report_html(
                show_mentions=show_mentions,
                show_unique_mentions=show_unique_mentions,
                show_impressions=show_impressions,
                show_effective_reach=show_effective_reach,
                show_headline_examples=show_headline_examples,
            )
            if report_html:
                st.markdown(report_html, unsafe_allow_html=True)
            else:
                st.info("Save authors to the shortlist to build the report block.")
            return

    st.markdown(
        """
        <style>
        .authors-step-note {
            margin: 0.15rem 0 1rem 0;
            color: rgba(250, 250, 250, 0.72);
            font-size: 0.95rem;
        }
        div[data-testid="stButton"] button[kind="secondary"] {
            min-height: 2.8rem;
            font-weight: 600;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    step1, step2, step3, step4 = st.columns(4, gap="small")
    with step1:
        if st.button("1. Missing", key="authors_nav_missing", use_container_width=True, type="primary" if st.session_state.authors_section == "Missing" else "secondary"):
            st.session_state.authors_section = "Missing"
            st.rerun()
    with step2:
        if st.button("2. Outlets", key="authors_nav_outlets", use_container_width=True, type="primary" if st.session_state.authors_section == "Outlets" else "secondary"):
            st.session_state.authors_section = "Outlets"
            st.rerun()
    with step3:
        if st.button("3. Selection", key="authors_nav_selection", use_container_width=True, type="primary" if st.session_state.authors_section == "Selection" else "secondary"):
            st.session_state.authors_section = "Selection"
            st.rerun()
    with step4:
        if st.button("4. Insights", key="authors_nav_insights", use_container_width=True, type="primary" if st.session_state.authors_section == "Insights" else "secondary"):
            st.session_state.authors_section = "Insights"
            st.rerun()

    st.markdown('<div class="authors-step-note">Work left to right: fill missing authors, assign outlets, curate key authors, then review output-ready insights.</div>', unsafe_allow_html=True)

    section = st.session_state.authors_section

    if section == "Missing":
        render_missing_authors_tab()
    elif section == "Outlets":
        render_author_outlets_tab()
    elif section == "Selection":
        render_author_insights_tab(mode="selection")
    else:
        render_author_insights_tab(mode="insights")
