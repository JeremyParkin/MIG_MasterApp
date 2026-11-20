def invalidate_clean_workbook(session_state) -> None:
    if session_state.pop("clean_excel_bytes", None) is not None:
        session_state["_rapid_clean_workbook_needs_rebuild"] = True
    session_state.pop("clean_excel_built_at", None)
    session_state.pop("_clean_excel_scope_key", None)
