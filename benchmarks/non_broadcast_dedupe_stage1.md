# Non-broadcast deduplication: Stage 1 verification

October 29, 2026. Verification record for the approved Stage 1 optimization, captured before committing. No deployment was performed during verification.

## Change and scope

The only production change is inside `dedupe_non_broadcast_by_fields` in `processing/standard_cleaning.py`. For unique-index groups of at least eight rows, the four existing comparison fields are read once into a local lookup. Smaller groups reuse the existing pandas indexer to avoid cache construction overhead. Non-unique labels retain the original indexer behavior.

Eight is a cache-setup cutoff, not a matching threshold. Pair iteration and early breaks, all comparison rules and thresholds, exact-snippet links, adjacency sets, transitive traversal, winner selection, retained/deleted row assembly, and output ordering are unchanged. Broadcast processing, URL deduplication, social processing, and all other cleaning functions are unchanged. An AST comparison against the pre-optimization revision verified that all production code outside this one function is identical.

The frozen reference includes the original function and its normalization, comparison, and winner-selection helpers from commit `c1c3a5614878d842151590f9fd5aadf5c02fdd9f`. It lives in `tests/fixtures/non_broadcast_dedupe_reference.py` and is independent of the optimized helpers.

## Verification

- Focused suite: **199 tests passed**: 196 output-comparison tests and three preexisting-invalid-input error comparisons.
- Comparisons use `assert_frame_equal(..., check_exact=True)` on both returned dataframes, preserving values, dtypes, indexes, column order, row order, and categorical metadata. Inputs are checked for mutation after each implementation.
- Includes 160 seeded randomized datasets; the golden and malformed repository fixtures; groups of 250, 251, 800, and 1,200 rows; missing/invalid dates; 48-hour boundaries and transitive chains; missing values and columns; tied winner rankings; non-default shuffled indexes; timezone-aware dates across daylight saving; nullable/string/categorical dtypes; helper-name collisions; and cache boundaries at 7/8 rows.
- The three unchanged error cases cover duplicate index labels, a categorical input without the required blank category, and `pd.NA` snippet scalars. These existing behaviors were preserved, not repaired.
- Full regression suite: **469 tests passed, 102 subtests passed**, with 24 date-parsing warnings; 61.77 seconds.
- Browser regression suite: **4 passed**, covering golden CSV upload and cleaning, malformed upload warnings and date-row removal, worksheet selection, and cleaned-workbook re-entry; 13.84 seconds.
- Every benchmark dataset also passed exact output and input-mutation comparisons before timing.

## Benchmark method

Local environment: Python 3.13.5, pandas 2.3.3, macOS-15.7.4-arm64-arm-64bit-Mach-O.

Each number is the median of three wall-clock runs of the complete non-broadcast function, including its copies and output construction. Run order alternates, and garbage collection runs before each sample. Dataset generation, parity checks/warmup, imports, disk I/O, and app startup are outside the timed region. Benchmarks ran after the regression suite completed, with no concurrent test run.

Synthetic datasets have 40 columns, including additional coverage context, and fixed seed 123. The two repository fixtures are normalized with TV/RADIO rows excluded. These are fixture and synthetic workloads, not live team/customer exports. Raw samples and environment metadata are in `benchmarks/results/non_broadcast_dedupe_stage1.json`.

| Distribution | Rows | Original seconds | Optimized seconds | Speedup |
|---|---:|---:|---:|---:|
| agility_golden_corpus.csv | 84 | 0.003962 | 0.003664 | 1.08x |
| agility_malformed_inputs.csv | 13 | 0.003060 | 0.002996 | 1.02x |
| unique | 250 | 0.006593 | 0.007284 | 0.91x |
| unique | 1,000 | 0.014650 | 0.015183 | 0.96x |
| unique | 5,000 | 0.059483 | 0.058133 | 1.02x |
| pairs | 250 | 0.121325 | 0.120884 | 1.00x |
| pairs | 1,000 | 0.481014 | 0.468537 | 1.03x |
| pairs | 5,000 | 2.317745 | 2.364347 | 0.98x |
| small_groups | 250 | 0.052382 | 0.031533 | 1.66x |
| small_groups | 1,000 | 0.212072 | 0.123276 | 1.72x |
| small_groups | 5,000 | 1.013811 | 0.603251 | 1.68x |
| mixed | 250 | 0.035106 | 0.029558 | 1.19x |
| mixed | 1,000 | 0.135435 | 0.108274 | 1.25x |
| mixed | 5,000 | 0.659545 | 0.554581 | 1.19x |
| dense | 200 | 0.389505 | 0.023642 | 16.48x |
| dense | 800 | 6.027780 | 0.295101 | 20.43x |
| dense | 1,200 | 13.672326 | 0.660698 | 20.69x |
| spread_dates | 200 | 0.011992 | 0.004849 | 2.47x |
| spread_dates | 800 | 0.041760 | 0.011834 | 3.53x |
| spread_dates | 1,200 | 0.061387 | 0.016540 | 3.71x |
| undated | 200 | 1.301167 | 0.847666 | 1.53x |
| undated | 800 | 5.597425 | 0.134432 | 41.64x |
| undated | 1,200 | 12.402251 | 0.286713 | 43.26x |
| fuzzy | 60 | 0.125928 | 0.086269 | 1.46x |
| fuzzy | 180 | 1.380690 | 0.768067 | 1.80x |

Distribution definitions: `unique` has no repeated grouping keys; `pairs` uses two-row groups; `small_groups` uses groups of ten; `mixed` has 70% singletons, 20% rows in pairs, and 10% in groups of 25. `dense` is one same-date group with varying snippets. `spread_dates` uses one grouping key with dates 72 hours apart. `undated` has missing dates and distinct snippets; groups above 250 retain the existing fuzzy-comparison guard. `fuzzy` has missing dates and similar/repeated snippets within the existing comparison limit.

## Observed regressions

- unique, 250 rows: **10.48% slower**, an additional **0.691 ms** (0.006593s to 0.007284s).
- unique, 1,000 rows: **3.64% slower**, an additional **0.533 ms** (0.014650s to 0.015183s).
- pairs, 5,000 rows: **2.01% slower**, an additional **46.602 ms** (2.317745s to 2.364347s).

These slowdowns are reported as measured. Very short runs are sensitive to timing variation; a universal speedup is not claimed. The cache-setup cutoff avoids the larger pair-group overhead seen with unconditional cache creation in the preliminary prototype.

## Limits

These timings measure this function, not total upload/Basic Cleaning latency or Community Cloud performance. No production memory reduction is claimed: connection storage and worst-case quadratic comparisons remain unchanged, and the local lookup adds temporary per-group metadata. Stage 2 was not implemented. Exact agreement on tested inputs is evidence, not proof for every possible dataframe; representative real team exports would extend the coverage.

## Reproduce

Use the project Python environment (the measured run used its Python 3.13.5 virtualenv), from the repository root:

```sh
python -m pytest -q tests/test_non_broadcast_dedupe.py
python -m pytest -q tests
python -m pytest -q e2e/test_upload_cleaning_spine.py
python benchmarks/benchmark_non_broadcast_dedupe.py --output /tmp/non_broadcast_dedupe_stage1.json --repeats 3
```
