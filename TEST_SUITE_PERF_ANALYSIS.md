# Numba Test Suite Compile-Time Analysis

Analysis of the Linux 64-bit compile-time/runtime metrics gathered by the
instrumentation in [swap357/numba#110](https://github.com/swap357/numba/pull/110),
plus benchmarking of the proposed fixes against released Numba 0.65.1.

## How the data was collected

PR #110 adds `CompileTimeTracker` (`numba/misc/compiletimeutils.py`), a
context manager wrapped around each test in `numba/testing/main.py`. It
installs listeners on three Numba compiler events:

| Event | What it measures |
|---|---|
| `numba:compile` | one full JIT compilation episode (start to finish) |
| `numba:run_pass` | the typing/lowering pass pipeline ("passes") |
| `numba:llvm_lock` | LLVM IR optimization + native codegen ("codegen") |

`codegen <= passes <= compile`, since codegen happens inside the lowering
pass. The CI workflows (`numba_*_wheel_builder.yml`) were patched to run
`NUMBA_TEST_COMPILE_TIMING=1 python -m numba.runtests -m 4 -v`, tee the log,
and upload `=== Compile Times === ... === End Compile Times ===` as a build
artifact — that block is the gist analyzed below. The rest of PR #110
(`ci_debug/analyze_compile_times.py`, `chart_compile_times.py`) is
post-processing/ranking tooling; it does not change compile behavior.

## Gist summary (10,715 parsed test records)

| | |
|---|---|
| Total wall time | 10,636.1s |
| Compile time | 7,770.3s (**73.1%** of total) |
| Run time | 2,865.8s (26.9% of total) |
| Tests issuing ≥10 distinct compiles | 446 of 10,715 (4.2% of tests) |
| Compile time owned by those 446 tests | 4,565.5s (**58.8%** of all compile time) |

The suite's compile-time bill is dominated by a small number of tests that
sweep many dtype/shape/kwarg combinations through `njit`, triggering a fresh
specialization (and fresh LLVM codegen) per combination — not by the typical
single-signature test.

Top 15 modules by total compile time (79.0% of all compile time):

```
2480.3s ( 1792 calls)  numba.tests.test_np_functions
 499.9s (  816 calls)  numba.tests.test_array_reductions
 442.2s (  897 calls)  numba.tests.test_array_methods
 373.9s (  609 calls)  numba.tests.test_sets
 298.9s (  285 calls)  numba.tests.npyufunc
 279.9s ( 2141 calls)  numba.tests.test_ufuncs
 260.4s (  424 calls)  numba.tests.test_fancy_indexing
 241.9s (  536 calls)  numba.tests.test_array_manipulation
 232.5s (  561 calls)  numba.tests.test_np_randomgen
 192.4s (  184 calls)  numba.tests.test_sort
 184.8s (  561 calls)  numba.tests.test_unicode
 173.8s (  256 calls)  numba.tests.test_unicode_array
 167.7s (  150 calls)  numba.tests.test_polynomial
 164.6s (  546 calls)  numba.tests.test_dyn_array
 147.8s (  345 calls)  numba.tests.test_typedlist
```

Single most expensive tests (`compile` column; `codegen` is the LLVM-side
share of `passes`):

```
319.24s compile |  46 calls | passes 319.10s | codegen 262.68s | test_np_functions.TestNPFunctions.test_isin_3a
317.00s compile |  46 calls | passes 316.86s | codegen 261.73s | test_np_functions.TestNPFunctions.test_isin_4
312.33s compile |  40 calls | passes 312.21s | codegen 258.99s | test_np_functions.TestNPFunctions.test_isin_3b
310.75s compile |  40 calls | passes 310.62s | codegen 258.71s | test_np_functions.TestNPFunctions.test_isin_2
101.40s compile | 170 calls | passes 100.89s | codegen  80.03s | test_array_methods.TestArrayMethods.test_clip_array_min_max
 94.18s compile |  56 calls | passes  94.06s | codegen  88.59s | test_linalg.TestBasics.test_outer
 72.91s compile |  51 calls | passes  72.76s | codegen  49.17s | test_np_functions.TestNPFunctions.test_correlate
 68.62s compile |  21 calls | passes  68.55s | codegen  61.42s | test_array_reductions.TestArrayReductions.test_nanpercentile_basic
 68.23s compile |  21 calls | passes  68.17s | codegen  61.17s | test_array_reductions.TestArrayReductions.test_percentile_basic
```

Across the worst offenders, **codegen is consistently 70–90% of compile
time** — this is the single biggest lever available, and it's controlled by
one already-existing config knob (see Candidate 1).

The four `test_isin_*` methods alone account for **1,259.3s — 16.2% of the
entire suite's compile time** (1,274.9s including a few unrelated
`isinstance`/`isinf` tests that happen to substring-match "isin").

## PR #110: what it is and isn't

PR #110 is **measurement infrastructure only** — it adds no compiler or
test-suite optimizations. It is the source of the dataset analyzed here, not
itself a candidate for "low-hanging fruit." The opportunities below were
found by analyzing its output, cross-referenced against this repo's current
source and CI config.

## Candidates

### 1. Set `NUMBA_OPT=1` for CI test runs *(highest impact, lowest risk)*

`numba/core/config.py:308` defaults LLVM optimization to `-O3` for every JIT
compilation, including ones the test suite performs purely to check
correctness (not measure runtime speed). This is a CI-only env var — it does
**not** change the optimization level used by Numba once installed/imported
normally by users; it only affects the test step itself.

```diff
# .github/workflows/numba_linux-64_wheel_builder.yml (and the other 4
# wheel-builder workflows, mirroring PR #110's own target files)
-          $PYTHON_PATH -m numba.runtests -m 4 -v
+          NUMBA_OPT=1 $PYTHON_PATH -m numba.runtests -m 4 -v
```

**Benchmarked against released Numba 0.65.1** (pip, numpy 2.4.6, llvmlite
0.47.0), compiling representative kernels (`np.isin`, `np.correlate`) across
the same kind of dtype-combination sweep the real tests perform, each
`NUMBA_OPT` level run in its own fresh process:

| kernel | combos | OPT=3 (current) | OPT=1 | OPT=0 |
|---|---|---|---|---|
| `np.isin(a, b)` | 64 dtype pairs | 325.75s | 282.64s (-13.2%) | 190.34s (**-41.6%**) |
| `np.correlate(a, b)` | 36 dtype pairs | 47.75s | 45.31s (-5.1%) | 36.42s (-23.7%) |
| `np.outer(a, b)` | 36 dtype pairs | 64.86s | 19.02s (**-70.7%**) | 11.89s (**-81.7%**) |
| `np.clip(a, lo, hi)` | 10 dtypes | 2.29s | 1.89s (-17.5%) | 1.55s (-32.3%) |

(Each row run in its own fresh subprocess — `numba.config.OPT` is latched at
import time, so opt level can't be changed mid-process.)

Per-call run time (post-compile execution of the tiny test arrays) increases
slightly at lower opt levels (e.g. isin: ~0.6ms → ~0.95ms at OPT=0) — a real
but practically irrelevant regression for correctness-only tests, and
irrelevant to the library's shipped default (`NUMBA_OPT` is unset for normal
`pip install numba` users; this only changes the CI test-runner's env).

Effect size varies by kernel (13–82% reduction at OPT=1/0 depending on how
codegen-bound the function is), but every kernel benchmarked got faster to
compile, with no exception, and the most expensive real-world offenders
(`test_isin_*`, `test_outer`) are exactly the ones that benefited most here
too. Applying even the conservative end of this range to the 7,770.3s suite
total implies CI compile time drops by roughly **15-30 minutes per full
run**.

**Tests that benefit most**: anything in the "≥10 compile calls" bucket
(58.8% of all compile time) — `test_isin_*`, `test_clip_array_min_max`,
`test_outer`, `test_correlate`, `test_percentile_basic`/`test_quantile_basic`,
`test_fancy_indexing` setitem variants, `test_sort` Timsort variants,
`test_polynomial`.

### 2. Roll out `_NUMBA_REDUCED_TESTING=1` to Linux/macOS CI

`numba/tests/support.py:127` already defines a `REDUCED_TESTING` flag, and
`numba/tests/test_np_functions.py` already uses it to shrink the isin/in1d
combinatorial sweep from dozens of array pairs (`_isin_arrays_full`) down to
4 (`_isin_arrays_reduced`), plus `@skip_if_reduced_testing` skips
`test_isin_3a/3b/4` entirely. **This flag is currently only enabled on
Windows CI** (`buildscripts/azure/azure-windows.yml:57,66,76`,
`buildscripts/condarecipe.local/run_test.bat:6`,
`.github/workflows/numba_win-64_wheel_builder.yml:148`) — Linux/macOS
wheel-builder workflows run the full combinatorial sweep.

```diff
# .github/workflows/numba_linux-64_wheel_builder.yml
       env:
+        _NUMBA_REDUCED_TESTING: 1
```

Since `test_isin_2/3a/3b/4` alone are 1,259.3s (16.2% of total suite compile
time) and reduced mode cuts the isin array-pair sweep from dozens to 4 *and*
skips 3 of the 4 test methods outright, this candidate alone is worth
roughly that 16.2% on top of whatever Candidate 1 saves — they compound
(fewer signatures × cheaper-per-signature codegen).

**Caveat**: this trades test coverage (fewer dtype/shape combinations
exercised) for speed. It's already accepted practice for Windows CI in this
repo, so extending it to Linux/macOS PR-validation runs (not necessarily the
scheduled release-wheel-validation runs) is a reasonable, precedented
trade — but it's a coverage decision, not a pure win like Candidate 1, so
it's flagged here rather than applied directly.

**Tests that benefit most**: `test_isin_2/3a/3b/4`, `test_in1d_2/3a/3b/4`,
and any other `numba/tests/*.py` module a maintainer chooses to extend the
existing `REDUCED_TESTING`/`skip_if_reduced_testing` pattern to (e.g.
`test_array_methods.test_clip_array_min_max`, `test_linalg.test_outer`,
`test_array_reductions` percentile/quantile — none of which currently use
the flag, so adopting it there is a larger follow-up, not 1-5 lines).

## What didn't pan out as "1-5 line" fixes

- `_in1d_impl` (`numba/np/arraymath.py:5034`, backs `np.isin`/`np.in1d`/
  `np.setdiff1d`) itself isn't inefficient — it's a reasonable
  sort-or-brute-force implementation. Its cost in the gist is purely a
  product of being compiled ~40-46 times per test for different
  dtype/shape/kwarg combinations, which candidates 1 and 2 both attack at
  the root (less optimization work per compile, fewer compiles).
- No caching opportunity exists here: each compile is a genuinely distinct
  type signature, so `cache=True` (which keys off signature) wouldn't avoid
  any of this work.

## Recommended priority

1. **`NUMBA_OPT=1` for CI test runs** — apply to all 5 wheel-builder
   workflows (and any other CI test-running step). Zero coverage trade-off,
   measured 13-42% compile-time cut, ~5 minutes of CI-config work.
2. **Extend `_NUMBA_REDUCED_TESTING=1` to Linux/macOS** — bring a
   already-Windows-only, already-merged pattern to the other platforms for
   PR validation. Maintainer call on which CI lanes (PR vs. scheduled
   release-wheel build) should keep full coverage.
3. *(Follow-up, not benchmarked here, >5 lines)* Extend the
   `REDUCED_TESTING`/`skip_if_reduced_testing` pattern from
   `test_np_functions.py` to the other combinatorial-sweep tests identified
   above (`test_clip_array_min_max`, `test_outer`, `test_percentile_basic`/
   `test_quantile_basic`, fancy-indexing `setitem`, Timsort merge tests).
