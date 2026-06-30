# Numba test-suite compile time analysis (2026-06-30)

Analysis of the compile-time/runtime instrumentation data captured by the
monitoring approach in [swap357/numba#110](https://github.com/swap357/numba/pull/110)
(`numba/misc/compiletimeutils.py` + CI artifact capture + `analyze_compile_times.py`),
applied to the metrics gist `0.67-times-numba-linux-64.txt`
(https://gist.github.com/swap415/6aee3d3742fcdab493c1c6fe2c2058c4).

## Data caveat

`gist.githubusercontent.com` (the raw-content host) was blocked by this
session's egress policy (403). The data below comes from the GitHub Gist API
(`api.github.com/gists/...`) instead, which truncates large files: **5,400 of
the full set of test entries were retrieved (~51% of the 1.79MB file), covering
test modules alphabetically from `test_dummyarray` through `test_unpack_sequence`.**
Modules later in the alphabet (`test_url*` onward) were not analyzed. Findings
below are grounded in this partial-but-substantial sample; re-running the
analysis with full file access could surface additional candidates.

## Headline numbers (sampled sub-suite, 5,400 tests)

| Metric | Value |
|---|---|
| Total duration | 8,341.9s |
| Compile time | 7,440.3s (**89.2%**) |
| Run time | 901.6s (10.8%) |

JIT compilation, not execution, dominates the test suite's wall-clock time by
nearly 9:1. The single largest contributor is one file:

| Test file | Compile time | Run time | # tests |
|---|---:|---:|---:|
| `numba.tests.test_np_functions` | 2,480.3s | 125.6s | 206 |
| `numba.tests.test_array_reductions` | 499.9s | 65.7s | 447 |
| `numba.tests.test_array_methods` | 442.2s | 36.6s | 80 |
| `numba.tests.test_sets` | 373.9s | 52.9s | 237 |

`test_np_functions.py` alone accounts for **33.3% of all compile time** in the
sampled sub-suite.

## Finding 1 (validated, recommended): turn on `_NUMBA_REDUCED_TESTING` for main CI

`numba/tests/support.py` already defines a `REDUCED_TESTING` flag
(`_NUMBA_REDUCED_TESTING` env var) and a `skip_if_reduced_testing` decorator.
`test_np_functions.py` already gates its most expensive tests behind it
(`test_isin_3a/3b` are skipped outright; `test_isin_2/4`, `test_correlate`,
`test_convolve`, `test_vander_basic`, `test_ediff1d_basic`,
`test_argpartition_fuzz`, `test_extract_basic` use drastically smaller
dtype/size/value matrices). Today this flag is set in exactly one CI job,
`.github/workflows/numba_win-64_wheel_builder.yml`, for the Windows wheel
*validation* step — the main Linux/macOS test workflows
(`numba_linux-64_conda_builder.yml`, `numba_linux-64_wheel_builder.yml`) never
set it.

**Proposed change:** add `_NUMBA_REDUCED_TESTING: 1` to the env for the main
test-running CI jobs (or at least the PR-gating ones) — a one-line addition
per workflow, identical to the pattern already proven safe in the Windows job.

**Benchmark (released numba 0.65.1 from PyPI, same machine, same 6 tests):**

| Mode | `test_correlate`, `test_convolve`, `test_vander_basic`, `test_ediff1d_basic`, `test_argpartition_fuzz`, `test_extract_basic` |
|---|---:|
| Full (current CI default) | **244.0s** |
| `_NUMBA_REDUCED_TESTING=1` | **38.1s** |
| Speedup | **6.4x** (-84.4%) |

Confirmed the gating code is present and behaves identically in the latest
PyPI release (0.65.1), not just this dev branch — the win is available today.

Adding the 4 `test_isin_*` tests (gist baseline: 1,259.3s combined; 3 of 4 are
skipped entirely under reduced mode) would push total savings for this one
file well past 1,400s on top of the above.

**Risk:** none identified — reuses an existing, already-shipped mechanism;
full-coverage runs (release builds, nightly) simply omit the env var.

## Finding 2: extend the same pattern to currently-ungated heavy tests

These modules have no `REDUCED_TESTING` gating at all today, despite being
the next-largest compile-time contributors in the sample:

| Test | Compile time | Calls | Module |
|---|---:|---:|---|
| `test_clip_array_min_max` | 101.4s | 170 | `test_array_methods.py` |
| `test_outer` | 94.2s | 56 | `test_linalg.py` |
| `test_nanpercentile_basic` | 68.6s | 21 | `test_array_reductions.py` |
| `test_percentile_basic` | 68.2s | 21 | `test_array_reductions.py` |
| `test_nanquantile_basic` | 65.7s | 20 | `test_array_reductions.py` |
| `test_quantile_basic` | 64.2s | 20 | `test_array_reductions.py` |
| `test_fill_diagonal_basic` | 63.9s | 60 | `test_array_manipulation.py` |
| `test_setitem` (x2 classes) | 126.0s | 91+92 | `test_fancy_indexing.py` |
| `test_kron` | 40.6s | 49 | `test_linalg.py` |
| `test_poly_polydiv_basic` / `test_poly_polyval_basic` | 76.4s | 15+21 | `test_polynomial.py` |

Each is structured as a dtype/size/value cross-product loop (e.g. `test_outer`:
`product(self.sizes, self.sizes, self.dtypes)`; the 4 percentile/quantile
tests share a `check_percentile_and_quantile` helper with a fixed, ungated
input matrix). Applying the same 3-5 line `if REDUCED_TESTING: <small set>
else: <full set>` pattern already used in `test_np_functions.py` would trim
these similarly, without weakening full-mode coverage. Not yet benchmarked in
isolation — recommended as the next CI-time investigation, prioritized by the
table above.

## Finding 3 (lower confidence — flag for follow-up, not yet actionable)

`test_sort.TestTimsortArrays.test_merge_lo_hi` (61.4s compile, 128 calls)
recompiles repeatedly across 4 size pairs including (1000, 1100), despite
`merge_lo`/`merge_hi` being fixed-signature jitted functions that should only
need to compile once. Worth a follow-up to confirm whether this is genuine
recompilation (e.g. a fresh jit wrapper per test setup) or an instrumentation
artifact, before proposing a specific fix.

## What Finding 1+2 do *not* address

Run time was never the bottleneck here (10.8% of total), so none of the above
changes Numba's actual JIT-compiled code performance for end users — they cut
CI/test-suite wall-clock by skipping redundant *type-signature* recompiles in
the test matrix itself. No compiler-internals changes (e.g. LLVM optimization
pass behavior) are proposed here; that would need separate, far riskier
investigation and is out of scope for a "few-line, low-risk" change.
