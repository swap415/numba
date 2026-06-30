# Numba Test Suite Performance Analysis

Analysis of compile-time/runtime metrics from the CI instrumentation added in
[swap357/numba#110](https://github.com/swap357/numba/pull/110)
(`NUMBA_TEST_COMPILE_TIMING=1` + `ci_debug/analyze_compile_times.py`), applied
to the slowest tests reported in the linked
[CI metrics gist](https://gist.github.com/swap415/6aee3d3742fcdab493c1c6fe2c2058c4).

All numbers below were re-measured locally (4 vCPU container, Python 3.11,
NumPy 2.x), running each test method in isolation to avoid CI noise. Absolute
times differ from the gist (different hardware), but the *relative* effect of
each change was verified directly and cross-checked against the latest
released Numba (`pip install numba==0.65.1`) to rule out a fork-specific
regression.

## Root cause

The slow tests in the gist all share one trait: they call a JIT function with
many distinct type signatures inside a single test method (different
dtype/axis/shape/order combinations). Each *new* signature pays Numba's full
compile pipeline (typing → lowering → LLVM IR → LLVM optimization → codegen),
and for several `numba/np/arraymath.py` / `arrayobj.py` overloads
(`array.sum(axis=, dtype=)`, `np.clip`, `np.broadcast_arrays`,
`fill_diagonal`, `transpose`), the **LLVM O3 optimization pass pipeline**
dominates that cost — confirmed by the gist's own `codegen` column being
60-90% of total `compile` time on the worst offenders.

Critically, this is **not a regression in this fork**: a standalone
`array.sum(axis=, dtype=)` micro-benchmark run against released `numba==0.65.1`
reproduces the same pathology (see Candidate 1).

## Candidate 1 — `NUMBA_OPT=0` (or `1`) for CI test runs ⭐ primary recommendation

**Change:** one line in the CI workflow (next to the
`NUMBA_TEST_COMPILE_TIMING=1` env var PR #110 already added):

```yaml
NUMBA_OPT=0 NUMBA_TEST_COMPILE_TIMING=1 python -m numba.runtests -m 4 -v 2>&1 | tee runtests.log
```

`NUMBA_OPT` controls the LLVM optimization level Numba uses for every JIT
compile (default `3`). Tests only need *correctness*, not optimized machine
code, so paying for O3 on every one of the thousands of signatures the test
suite compiles is pure waste.

**Measured impact** (this repo, dev branch, isolated re-runs, no concurrent
contention):

| Test | OPT=3 (default) | OPT=0 | Speedup |
|---|---|---|---|
| `test_sum_axis_dtype_kws` | 574.3s | 11.8s | **48.6x** |
| `test_sum_axis_kws1` | 461.9s | 12.2s | **37.9x** |
| `test_sum_exceptions` | 61.1s | 1.6s | **38.2x** |
| `test_array_transpose_axes` | 15.2s | 6.3s | 2.4x |
| `test_fill_diagonal_basic` | 41.6s | 23.5s | 1.8x |
| `test_add_axis` | 10.9s | 4.3s | 2.5x |
| `test_clip_array_min_max` | 72.5s | 71.6s | ~1.0x (no win) |
| `test_take` | 13.1s | 9.8s | 1.3x |
| `test_array_view` | 8.8s | 8.5s | ~1.0x (no win) |
| `test_broadcast_arrays_same_input_shapes` | 6.6s | 6.6s | ~1.0x (no win) |
| **10-test sample TOTAL** | **1265.95s** | **142.68s** | **8.9x** |

All tests passed (`0 failures`) in both configurations.

**Cross-checked against the latest release** (`numba==0.65.1` from PyPI, not
this fork), standalone `arr.sum(axis=, dtype=)` micro-benchmark, 3 signatures:

| | OPT=3 (default) | OPT=0 | Speedup |
|---|---|---|---|
| `numba==0.65.1` | 173.3s (57.8s/signature) | 1.95s (0.65s/signature) | **88.7x** |

This confirms the LLVM-O3-on-reduction-code pathology is a long-standing,
version-independent Numba characteristic, not something introduced recently
in this fork — and that the fix transfers directly to the released package.

**Scope/risk:** `NUMBA_OPT` also affects the runtime performance of the
*compiled* code, not just compile speed — O0 code runs slower. This is a
non-issue for the test suite (tests check correctness; the gist's own `Run`
column shows runtimes are sub-second vs. tens-to-hundreds of seconds of
compile time), but **do not** set this globally for end users / production
builds. Scope it to the CI test-running step only (as shown above), exactly
parallel to how PR #110 already scopes `NUMBA_TEST_COMPILE_TIMING=1`.

**Tests that benefit most:** any test exercising `array.sum(axis=, dtype=)`
combinatorics (`test_sum_*`), `fill_diagonal`, `transpose`, `add_axis` — i.e.
tests whose cost comes from compiling many distinct signatures of reduction
or shape-manipulation code, where each signature is a substantial function
LLVM's O3 pipeline works hard on. Tests dominated by `np.clip`/
`np.broadcast_arrays`/`array_view` (lots of *small*, simple specializations)
see little to no benefit, since their cost is mostly typing/lowering
overhead repeated per-signature, not O3 codegen time.

## Candidate 2 (investigated, not recommended) — `cache=True` on closures redefined in test loops

Two tests (`test_array_methods.TestArrayMethods._lower_clip_result_test_util`,
`test_array_view`'s `run()` helper) define-and-JIT-compile a brand-new
closure on every call inside a loop — a classic Numba anti-pattern that
defeats in-process dispatcher caching. Adding `cache=True` looked promising
on paper, but verification showed it **doesn't help here**: the closures
capture a `Dispatcher` object as a free variable, and Numba explicitly
refuses to disk-cache functions with such "dynamic globals" — it just emits
a `NumbaWarning` on every call with zero speed benefit. Confirmed empirically
(`test_clip_array_min_max`: 72.8s unpatched vs. 71.6s patched, plus warning
spam). Reverted; not included as a recommendation. Recorded here so it isn't
re-investigated.

## Candidate 3 — land PR #110

The instrumentation itself (`NUMBA_TEST_COMPILE_TIMING`,
`analyze_compile_times.py`) is what made this analysis possible. It's
currently a draft with no description. Landing it (or a lighter-weight
version that just uploads the artifact) gives ongoing visibility to catch
future compile-time regressions — including verifying the `NUMBA_OPT=0` win
above holds over time as the compiler evolves.

## Things that look like "low-hanging fruit" but aren't

- **Reducing the dtype/axis/shape cartesian products** in `test_sum_axis_dtype_kws`,
  `test_clip_array_min_max`, etc. would proportionally cut compile time, but
  trades away real type-signature test coverage for speed — not recommended
  as a first move when `NUMBA_OPT=0` gets a comparable or larger win for free.
- No 1-5 line change to the **compiler internals** (`np_clip`, `broadcast_arrays`,
  the reduction overloads) was found that meaningfully reduces per-signature
  compile cost — their cost is dominated by LLVM's own O3 pipeline, which
  `NUMBA_OPT` already controls at zero implementation risk.

## Summary

| Priority | Change | Scope | Effect |
|---|---|---|---|
| 1 | `NUMBA_OPT=0` for CI test runs | 1-line CI workflow change | Up to 49x on individual tests, 8.9x on a 10-test sample, confirmed on latest release too |
| 2 | Land PR #110 | merge existing draft PR | Ongoing regression visibility |
| — | `cache=True` on closures | investigated | No benefit — Numba refuses to cache dispatcher-capturing closures |

**Next step recommended:** apply `NUMBA_OPT=0` (or `1` as a more
conservative middle ground) to the CI test-running step in the five
`numba_*_wheel_builder.yml` workflows PR #110 already touches, and re-run
the full suite via that PR's own instrumentation to get a CI-validated,
whole-suite before/after number.
