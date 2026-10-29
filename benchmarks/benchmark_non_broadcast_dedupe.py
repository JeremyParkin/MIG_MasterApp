"""Compare Stage 1 against the frozen original; timings are medians in seconds.

Run from the repository root with the project's Python environment:
    python benchmarks/benchmark_non_broadcast_dedupe.py --output /tmp/dedupe-benchmark.json
"""
from __future__ import annotations

import argparse
import gc
import json
import platform
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]

import pandas as pd
from dedupe_support import FIXTURES, assert_identical_results, make_dataset, reference
from processing.standard_cleaning import dedupe_non_broadcast_by_fields
from utils.io import normalize_uploaded_dataframe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")

    cases = []
    for name in ["agility_golden_corpus.csv", "agility_malformed_inputs.csv"]:
        frame = normalize_uploaded_dataframe(pd.read_csv(FIXTURES / "agility" / name))
        cases.append((name, frame.loc[~frame["Type"].isin(["TV", "RADIO"])].copy()))
    for distribution in ["unique", "pairs", "small_groups", "mixed"]:
        for size in [250, 1000, 5000]:
            cases.append((distribution, make_dataset(distribution, size)))
    for distribution in ["dense", "spread_dates", "undated"]:
        for size in [200, 800, 1200]:
            cases.append((distribution, make_dataset(distribution, size)))
    for size in [60, 180]:
        cases.append(("fuzzy", make_dataset("fuzzy", size)))

    functions = [reference.dedupe_non_broadcast_by_fields, dedupe_non_broadcast_by_fields]
    results = {
        "python": platform.python_version(), "pandas": pd.__version__, "platform": platform.platform(),
        "reference_revision": reference.__doc__.splitlines()[0], "repeats": args.repeats,
        "method": "Exact dataframe parity before timing; alternate run order; gc.collect before each run; medians of wall-clock seconds; no application startup or I/O in timed region.",
        "results": [],
    }
    for case, frame in cases:
        assert_identical_results(*functions, frame)
        samples = [[], []]
        for repeat in range(args.repeats):
            for index in ([0, 1] if repeat % 2 == 0 else [1, 0]):
                gc.collect()
                start = time.perf_counter()
                output = functions[index](frame)
                elapsed = time.perf_counter() - start
                samples[index].append(elapsed)
                del output
        old, new = map(statistics.median, samples)
        result = {"distribution": case, "rows": len(frame), "columns": len(frame.columns),
                  "original_seconds": old, "optimized_seconds": new, "speedup": old / new,
                  "original_samples": samples[0], "optimized_samples": samples[1]}
        results["results"].append(result)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2) + "\n")
        print(f"{case:32s} {len(frame):5d}  {old:9.4f}s -> {new:9.4f}s  {old/new:6.2f}x", flush=True)


if __name__ == "__main__":
    main()
