"""Fresh-process Basic Cleaning timing, RSS, and exact output comparison.

python benchmarks/benchmark_bounded_tokens.py --file /path/to/export.csv \
    --repeats 6 --output /tmp/bounded-tokens.json

Uses the page's three stages and upload/category preparation. Retains the upload
buffer; excludes startup, upload, GC, and previews from the cleaning timer. Each
worker saves its outputs temporarily for exact dataframe comparison after RSS
measurement stops. No source rows are written to the results JSON.
"""
from __future__ import annotations

import argparse
import ctypes
import gc
import hashlib
import io
import json
import os
import platform
import resource
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import nullcontext
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]


def resident_memory_reader():
    if sys.platform == "darwin":
        lib = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
        lib.mach_task_self.restype = ctypes.c_uint
        task = lib.mach_task_self()

        def rss():
            buf = (ctypes.c_uint64 * 6)()
            count = ctypes.c_uint(12)
            if lib.task_info(task, 20, ctypes.byref(buf), ctypes.byref(count)):
                raise RuntimeError("Cannot read own resident memory")
            return buf[1] / 1024**2
    elif sys.platform.startswith("linux"):
        def rss():
            pages = int(Path("/proc/self/statm").read_text().split()[1])
            return pages * os.sysconf("SC_PAGE_SIZE") / 1024**2
    else:
        raise RuntimeError("RSS sampling supports macOS and Linux")
    return rss


def worker(args):
    import pandas as pd
    import sklearn
    from threadpoolctl import threadpool_info

    from bounded_token_support import original_token_extraction
    from processing.coverage_flags import add_coverage_flags, apply_story_family_press_release_flags
    from processing.data_quality import build_data_quality_warnings
    from processing.effective_reach import apply_effective_reach_social, apply_effective_reach_traditional
    from processing.standard_cleaning import run_standard_cleaning
    from processing.story_grouping import (
        NORMAL_TEXT_SIMILARITY_THRESHOLD, build_unique_story_table,
        cluster_by_media_type, mark_prime_examples,
    )
    from utils.io import build_upload_quality_report, normalize_uploaded_dataframe, read_uploaded_file

    rss = resident_memory_reader()
    import_rss = rss()
    active, records = [], []
    stop = threading.Event()

    def sample():
        while not stop.wait(0.005):
            value = rss()
            for record in active.copy():
                record["peak_rss_mib"] = max(record["peak_rss_mib"], value)

    def measure(label, fn, *a, **kw):
        record = {"stage": label, "start_rss_mib": rss()}
        record["peak_rss_mib"] = record["start_rss_mib"]
        active.append(record)
        start = time.perf_counter()
        try:
            return fn(*a, **kw)
        finally:
            record.update(seconds=time.perf_counter() - start, end_rss_mib=rss())
            record["peak_rss_mib"] = max(record["peak_rss_mib"], record["end_rss_mib"])
            active.remove(record)
            records.append(record)

    def baseline(raw):
        frame = normalize_uploaded_dataframe(raw)
        for column in ["Sentiment", "Continent", "Country", "Prov/State", "City", "Language"]:
            if column in frame:
                frame[column] = frame[column].astype("category")
        return frame

    class Upload(io.BytesIO):
        name = args.file.name
        file_id = "bounded-token-benchmark"

    monitor = threading.Thread(target=sample, daemon=True)
    monitor.start()
    upload = measure("upload_buffer", lambda: Upload(args.file.read_bytes()))
    state = {}
    raw = measure("csv_read", read_uploaded_file, upload, session_state=state)
    state["df_untouched"] = raw
    state["df_traditional"] = measure("upload_normalization", baseline, raw)
    state["upload_quality_report"] = measure(
        "upload_quality", build_upload_quality_report, raw, state["df_traditional"]
    )

    def summary():
        source = baseline(raw)
        return {"row_count": len(source), "memory_bytes": int(source.memory_usage(deep=True).sum()),
                "warnings": build_data_quality_warnings(source)}

    state["_basic_cleaning_source_summary"] = measure("page_summary", summary)

    def stage1():
        source = baseline(raw)
        state.update(run_standard_cleaning(source, merge_online=True, drop_dupes=True, add_flags=False))
        state["df_ai_grouped"] = pd.DataFrame()
        state["df_ai_unique"] = pd.DataFrame()

    def stage2():
        state["df_traditional"] = apply_effective_reach_traditional(state["df_traditional"])
        state["df_social"] = apply_effective_reach_social(state["df_social"])
        state["df_traditional"] = add_coverage_flags(state["df_traditional"])

    def stage3():
        grouped = measure("canonical_grouping", cluster_by_media_type, state["df_traditional"],
                          similarity_threshold=NORMAL_TEXT_SIMILARITY_THRESHOLD, max_batch_size=1800)
        grouped = apply_story_family_press_release_flags(grouped)
        grouped = mark_prime_examples(grouped)
        unique = build_unique_story_table(grouped)
        state["df_traditional"] = state["df_ai_grouped"] = grouped
        state["df_ai_unique"] = unique

    context = original_token_extraction() if args.worker == "original" else nullcontext()
    with context:
        for label, fn in [("stage1", stage1), ("stage2", stage2), ("stage3", stage3)]:
            measure(label, fn)
            measure(label + "_gc", gc.collect)
    end_rss = rss()
    os_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2 if sys.platform == "darwin" else 1024)
    stop.set()
    monitor.join()
    frames = {key: value for key, value in state.items() if isinstance(value, pd.DataFrame)}
    result = {
        "implementation": args.worker, "python": platform.python_version(), "pandas": pd.__version__,
        "sklearn": sklearn.__version__, "platform": platform.platform(), "logical_cpus": os.cpu_count(),
        "threadpools": threadpool_info(), "import_rss_mib": import_rss, "end_rss_mib": end_rss,
        "os_high_water_mib": os_peak, "peak_pipeline_rss_mib": max(r["peak_rss_mib"] for r in records),
        "cleaning_seconds": sum(r["seconds"] for r in records if r["stage"] in {"stage1", "stage2", "stage3"}),
        "stages": records, "row_counts": {key: len(frame) for key, frame in frames.items()},
        "grouped_aliases_traditional": state["df_traditional"] is state["df_ai_grouped"],
    }
    # Sampling/OS high-water capture finish before snapshot serialization.
    pd.to_pickle(frames, args.snapshot)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument("--worker", choices=["original", "bounded"], help=argparse.SUPPRESS)
    parser.add_argument("--snapshot", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker(args)
        return
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    import pandas as pd
    from pandas.testing import assert_frame_equal

    results = {
        "file": args.file.name, "file_bytes": args.file.stat().st_size,
        "file_sha256": hashlib.sha256(args.file.read_bytes()).hexdigest(),
        "reference_revision": "58bd526e8f8562d7232fecd1abfb4c008d970aff",
        "method": "Fresh sequential processes; alternating AB/BA order; 5ms own-process RSS sampling; OS high-water before snapshot; exact dataframe parity for every run; unchanged neighbors/thread settings; retained upload buffer.",
        "samples": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="mig-bounded-tokens-") as temporary:
        temp = Path(temporary)
        oracle = temp / "oracle.pkl"
        for repeat in range(args.repeats):
            order = ["original", "bounded"] if repeat % 2 == 0 else ["bounded", "original"]
            for implementation in order:
                snapshot = temp / "current.pkl"
                output = temp / "worker.json"
                completed = subprocess.run(
                    [sys.executable, str(Path(__file__).resolve()), "--worker", implementation,
                     "--file", str(args.file.resolve()), "--output", str(output), "--snapshot", str(snapshot)],
                    capture_output=True, text=True,
                )
                if completed.returncode:
                    raise RuntimeError(completed.stderr)
                sample = json.loads(output.read_text())
                sample["repeat"] = repeat + 1
                if not oracle.exists():
                    snapshot.rename(oracle)
                else:
                    expected, actual = pd.read_pickle(oracle), pd.read_pickle(snapshot)
                    assert actual.keys() == expected.keys()
                    for key in expected:
                        assert_frame_equal(actual[key], expected[key], check_exact=True, obj=key)
                    del expected, actual
                    snapshot.unlink()
                    gc.collect()
                assert sample["grouped_aliases_traditional"]
                sample["exact_output_equivalence"] = True
                results["samples"].append(sample)
                args.output.write_text(json.dumps(results, indent=2) + "\n")
                print(f"{repeat + 1} {implementation}: {sample['cleaning_seconds']:.3f}s, "
                      f"RSS peak {sample['peak_pipeline_rss_mib']:.1f} MiB, "
                      f"OS peak {sample['os_high_water_mib']:.1f} MiB; exact output match", flush=True)
    results["medians"] = {
        implementation: {key: statistics.median(s[key] for s in results["samples"]
                                                if s["implementation"] == implementation)
                         for key in ["cleaning_seconds", "peak_pipeline_rss_mib", "os_high_water_mib"]}
        for implementation in ["original", "bounded"]
    }
    args.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
