from __future__ import annotations

import io
import json
import re
import zipfile

import pandas as pd


def build_notebooklm_zip(
    df_traditional: pd.DataFrame,
    df_social: pd.DataFrame,
    client_name: str,
    max_files: int = 50,
    max_rows_per_file: int = 450,
    max_words_per_file: int = 200_000,
    max_bytes_per_file: int = 50 * 1024 * 1024,
    text_truncate_len: int = 10_000,
):
    frames = []
    if df_traditional is not None and len(df_traditional) > 0:
        frames.append(df_traditional)
    if df_social is not None and len(df_social) > 0:
        frames.append(df_social)

    if not frames:
        raise ValueError("No coverage rows available for NotebookLM bundle.")

    df = pd.concat(frames, ignore_index=True)

    base_cols = [
        "Published Date", "Date", "Author", "Outlet", "Headline",
        "Coverage Snippet", "Snippet", "Summary", "Title", "Content",
        "Country", "Prov/State", "Type", "Media Type",
        "Impressions", "URL", "Sentiment", "Tags"
    ]
    prominence_cols = [c for c in df.columns if isinstance(c, str) and c.startswith("Prominence")]
    cols_to_keep = [c for c in base_cols if c in df.columns] + prominence_cols

    if not cols_to_keep:
        raise ValueError("No expected columns found to include in NotebookLM bundle.")

    df = df[cols_to_keep].copy()

    for date_col in ["Published Date", "Date"]:
        if date_col in df.columns:
            try:
                tmp = pd.to_datetime(df[date_col], errors="coerce")
                df[date_col] = tmp.dt.strftime("%Y-%m-%d")
            except Exception:
                pass

    text_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
    for col in text_cols:
        s = df[col].astype("string")
        s = s.str.slice(0, text_truncate_len)
        df[col] = s

    df = df.astype(object)
    df = df.where(df.notna(), None)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    n_total_rows = len(df)

    client_clean = re.sub(r"[^A-Za-z0-9]+", "", str(client_name))
    if not client_clean:
        client_clean = "Client"
    client_short = client_clean[:15]

    output_zip = io.BytesIO()
    total_rows_included = 0
    total_words_included = 0
    files_created = 0

    def row_word_count(row_dict: dict) -> int:
        parts = []
        for col in text_cols:
            if col in row_dict:
                val = row_dict.get(col)
                if val is not None:
                    parts.append(str(val))
        return len(" ".join(parts).split()) if parts else 0

    def make_json_safe(val):
        try:
            if pd.isna(val):
                return None
        except TypeError:
            pass
        return val

    with zipfile.ZipFile(output_zip, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        chunk_rows = []
        chunk_words = 0
        chunk_bytes = 0

        def flush_chunk(local_rows, index: int):
            nonlocal files_created
            if not local_rows:
                return
            json_str = json.dumps(local_rows, ensure_ascii=False)
            filename = f"json_{index}-{client_short}.txt"
            zf.writestr(filename, json_str)
            files_created += 1

        file_index = 1

        for _, row in df.iterrows():
            if file_index > max_files:
                break

            row_dict = {col: make_json_safe(row[col]) for col in cols_to_keep}

            w = row_word_count(row_dict)
            row_json = json.dumps(row_dict, ensure_ascii=False)
            b = len(row_json.encode("utf-8"))

            if chunk_rows and (
                len(chunk_rows) >= max_rows_per_file
                or chunk_words + w > max_words_per_file
                or chunk_bytes + b > max_bytes_per_file
            ):
                flush_chunk(chunk_rows, file_index)
                file_index += 1
                if file_index > max_files:
                    break
                chunk_rows = []
                chunk_words = 0
                chunk_bytes = 0

            if file_index <= max_files:
                chunk_rows.append(row_dict)
                chunk_words += w
                chunk_bytes += b
                total_rows_included += 1
                total_words_included += w

        if chunk_rows and file_index <= max_files:
            flush_chunk(chunk_rows, file_index)

    output_zip.seek(0)

    info = {
        "client_short": client_short,
        "total_rows": int(n_total_rows),
        "rows_included": int(total_rows_included),
        "files_created": int(files_created),
        "max_files": int(max_files),
        "max_rows_per_file": int(max_rows_per_file),
        "max_words_per_file": int(max_words_per_file),
        "max_bytes_per_file": int(max_bytes_per_file),
        "total_words_included": int(total_words_included),
    }
    return output_zip, info