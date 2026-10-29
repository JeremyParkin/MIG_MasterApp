# Bounded token extraction: implementation and verification

October 29, 2026. Changes are uncommitted pending review. Baseline revision:
`58bd526e8f8562d7232fecd1abfb4c008d970aff` (completed Stage 1 lookup optimization).

## Scope

Only two production expressions change:

- `processing/standard_cleaning.py:_leading_token_tuple`: use whitespace
  `split(maxsplit=limit)` before the existing slice. For the normal twelve-word
  prefix, this creates at most thirteen pieces, with the unused remainder left
  unsplit. Empty-input handling and custom limits, including zero and negative
  limits, retain their previous results. This shared helper is also called by
  broadcast processing; its returned tokens are unchanged and broadcast rules
  are untouched.
- `processing/story_grouping.py:cluster_similar_stories`: cap the temporary
  whitespace split at nineteen splits. Its length is therefore capped at twenty,
  preserving exactly the existing `<20` weak-evidence predicate. No other use is
  made of the temporary counts.

Both operations retain the same built-in whitespace splitting semantics,
including Unicode whitespace. They still may copy a long unused remainder;
this bounds the number of split pieces, not all temporary bytes independently
of snippet length. Full stored snippets and TF-IDF input text are unchanged.
No changes to thresholds, comparison rules, neighbor-search settings, text
normalization/reuse, winner/representative selection, connection storage,
batching, or session-state ownership are included. Stage 2 is not implemented.

## Equivalence and regression verification

The two original functions are frozen in
`tests/fixtures/bounded_tokens_reference.py`. Their function ASTs were checked
against the baseline Git revision and match exactly. The original functions
are substituted only in tests/benchmark workers, without switching any other
algorithm or runtime setting.

`tests/test_bounded_tokens.py`: **714 passed**, 22 warnings, 4.55 seconds.
Coverage includes:

- Every Unicode code point recognized as whitespace by Python 3.13 (29
  characters), repeated/leading/trailing whitespace, and mixed whitespace.
- Empty/whitespace-only/None prefix inputs, Unicode tokens, emoji, zero-width
  space and BOM (which remain non-whitespace), and custom prefix limits.
- Token counts 0, 1, 11, 12, 13, 18, 19, 20, 21, 100, and 20,000.
- Actual weak-evidence edge filtering and connected components around the
  threshold, using a controlled 0.94 cosine edge. Both vectorizer inputs are
  compared exactly, including complete long-text tails; input frames remain
  unchanged.
- 1,800 randomized Unicode samples for prefix and weak-evidence predicates.
- End-to-end exact comparisons on golden/malformed fixtures and eight altered
  golden datasets containing missing, Unicode, long, and boundary-length snippets
  and missing dates. All returned frames are compared with
  `assert_frame_equal(check_exact=True)`, including values, dtypes, indexes,
  column/row ordering, grouping and representative markers.

Main headless regression suite: **1,183 passed, 102 subtests passed**, 46 warnings,
67.37 seconds. This includes the focused tests plus the existing non-broadcast
oracle comparisons, canonical/syndication grouping, coverage flags, reach,
Basic Cleaning page/session-state, upload, export, and other app regression tests.

Browser upload/cleaning regressions: **4 passed**, 14.59 seconds. The initial
Chromium launch was blocked by the execution sandbox's macOS IPC restriction;
rerunning with local browser/server permissions passed all four tests. No test
or app changes were needed for that retry.

## Real-export fresh-process benchmark

Input: the same local `INOD-Mentions - 2026-10-01_09-49.csv`, **70,103,980 bytes,
4,828 rows**. No coverage rows/text are stored in the benchmark JSON. Its input
checksum and environment metadata are recorded in `results/bounded_tokens.json`.

Python 3.13.5, pandas 2.3.3, scikit-learn 1.8.0, local macOS/Apple ARM, ten logical
CPUs. Six runs per implementation, each in a fresh sequential subprocess;
alternating original/bounded and bounded/original order. No benchmark workers
ran concurrently with each other or with regression tests. Neighbor/thread
settings remain the production defaults.

The harness reproduces the page's upload/baseline category preparation, summary,
three processing stages, session-state frames, and inter-stage GC. It retains
the upload buffer. The total processing timer follows the page's convention:
includes the baseline rebuild and all three stages; excludes imports, upload,
summary, GC, reruns, previews, and browser rendering. Memory sampling covers
upload through final GC; OS high-water is captured before output serialization.
Own-process RSS is sampled every 5 ms; OS high-water provides an independent
check. These are local measurements, not Community Cloud container measurements.

Every subsequent worker's six output frames were compared exactly against the
first original worker's reference snapshot: raw, traditional, social, duplicates,
grouped, and unique. All eleven comparisons passed. Snapshot serialization and
comparison occur after measurement; temporary snapshots are removed automatically.
The grouped/traditional session-state alias was also verified. Outputs remain:
3,799 traditional rows, zero social, 1,029 duplicates, and 1,130 unique stories.

| Metric (median of six runs) | Original | Bounded | Difference |
|---|---:|---:|---:|
| Total processing | 7.630 s | 7.216 s | 0.415 s less; **5.44% less time, 1.058× speedup** |
| Sampled process peak RSS | 945.8 MiB | 827.3 MiB | **118.5 MiB less (12.53%)** |
| OS process high-water | 955.7 MiB | 829.0 MiB | **126.7 MiB less (13.25%)** |

| Pair | Original seconds | Bounded seconds | Speedup | Original OS peak MiB | Bounded OS peak MiB |
|---|---:|---:|---:|---:|---:|
| 1 | 7.413 | 7.285 | 1.018× | 995.4 | 823.5 |
| 2 | 7.739 | 7.627 | 1.015× | 981.3 | 641.4 |
| 3 | 7.676 | 7.213 | 1.064× | 947.9 | 833.5 |
| 4 | 7.471 | 7.218 | 1.035× | 940.3 | 832.9 |
| 5 | 7.990 | 7.200 | 1.110× | 931.8 | 890.4 |
| 6 | 7.585 | 7.199 | 1.054× | 963.4 | 825.1 |

| Stage (independent medians) | Original seconds | Bounded seconds |
|---|---:|---:|
| stage1 | 2.828 | 2.602 |
| stage2 | 0.893 | 0.889 |
| stage3 | 3.812 | 3.717 |
| canonical_grouping | 3.041 | 2.976 |

Independent stage medians should not be summed to reconstruct the median total.
`canonical_grouping` is nested inside `stage3`, not an additional stage.

## Regressions, variability, and interpretation

All six paired bounded runs were faster overall. The range was 7.413–7.990 s
original versus 7.199–7.627 s bounded. Part 3 nevertheless regressed in two pairs:
3.714→3.819 s and 3.809→4.167 s. Canonical grouping alone similarly regressed in
those two pairs. The slice improves median CPU time modestly; it is not a
promise that every stage or every run gets faster.

OS peaks varied considerably: 931.8–995.4 MiB original versus 641.4–890.4 MiB
bounded. All six paired peaks were lower, and the highest bounded peak remained
below the lowest original peak. Report the median reduction rather than the
largest single-run saving. Allocation history, native buffers, and process
scheduling affect both time and RSS; the sampler can miss short spikes. Eleven
of twelve sampled pipeline maxima occurred during canonical grouping. The
remaining bounded run peaked during post-Part-3 GC, only 1.05 MiB above its
canonical-grouping peak; these are process-residency measurements rather than
precise live-allocation attribution.

This slice provides useful local memory headroom without altering coverage
methodology. Full-text normalization, TF-IDF, neighbor-search intermediates, and
worst-case quadratic connections still exist. No claim is made that this alone
prevents Community Cloud resource-limit events or reproduces the Cloud's
24.71-second total.

## Reproduce

From the repository root with the existing project environment:

```sh
python -m pytest tests/test_bounded_tokens.py -q --disable-warnings
python -m pytest tests -q --disable-warnings
python -m pytest e2e/test_upload_cleaning_spine.py -q --disable-warnings
python benchmarks/benchmark_bounded_tokens.py \
  --file '/path/to/INOD-Mentions - 2026-10-01_09-49.csv' \
  --repeats 6 --output /tmp/bounded-tokens.json
```

The benchmark supports macOS and Linux RSS sampling. Browser tests require the
existing Playwright Chromium installation and permission to launch a local
browser and Streamlit server.
