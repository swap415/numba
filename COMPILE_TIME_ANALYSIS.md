# Numba test suite compile-time analysis

Source data: [compile-time/runtime metrics gist](https://gist.github.com/swap415/6aee3d3742fcdab493c1c6fe2c2058c4)
(5,400 test entries, Linux x86_64), captured with the instrumentation added in
[swap357/numba#110](https://github.com/swap357/numba/pull/110)
(`numba/misc/compiletimeutils.py`, wired to Numba's `numba:compile`,
`numba:run_pass`, and `numba:llvm_lock` events).

## Headline numbers

| | seconds | % of suite |
|---|---:|---:|
| Total suite time | 8,341.9 | 100% |
| **Compile time** | **7,440.3** | **89.2%** |
| Runtime | 901.6 | 10.8% |

JIT compilation, not runtime execution, dominates the test suite. Within
compile time, the codegen sub-phase (LLVM optimization + machine code
generation, `numba:llvm_lock`) consistently accounts for ~80-85% of the total
for the heaviest tests — confirmed both from the gist's per-test breakdown and
from the PR #110 instrumentation's own doc comment: "codegen <= passes <=
compile (codegen happens inside the lowering pass)".

Top 10 compile-time offenders (module `numba.tests.test_*`):

| Test | Compile | Calls | Codegen share |
|---|---:|---:|---:|
| test_np_functions.test_isin_3a | 319.2s | 46 | 82% |
| test_np_functions.test_isin_4 | 317.0s | 46 | 82% |
| test_np_functions.test_isin_3b | 312.3s | 40 | 83% |
| test_np_functions.test_isin_2 | 310.8s | 40 | 83% |
| test_array_methods.test_clip_array_min_max | 101.4s | 170 | 79% |
| test_linalg.test_outer | 94.2s | 56 | 94% |
| test_np_functions.test_correlate | 72.9s | 51 | 67% |
| test_array_reductions.test_nanpercentile_basic | 68.6s | 21 | 90% |
| test_array_reductions.test_percentile_basic | 68.2s | 21 | 90% |
| test_array_reductions.test_nanquantile_basic | 65.7s | 20 | 89% |

The four `test_isin_*` tests alone cost **1,259.3s (15.1% of the entire
suite)**. Each recompiles a numpy-overload-heavy call chain
(`np.asarray`/`ravel`/`unique`/`argsort`/`concatenate`/fancy-indexing) from
scratch for every one of ~40-46 dtype/list-type combinations, and the
compiler fully types+lowers+optimizes *both* branches of `_in1d_impl`
(brute-force and sort-based) regardless of which one executes.

## Candidate 1 (primary): drop LLVM optimization level during tests — `NUMBA_OPT=0`

Numba's codegen pipeline (`numba/core/codegen.py`) always runs the *full*
module optimization pass at `config.OPT` (default 3, i.e. `-O3` with loop
vectorization), on top of an internal "cheap" O0 pass used for ref-count
pruning. For test code — compiled once, run a handful of times, correctness
checked but not benchmarked — that `-O3` pass buys little and costs a lot.

**Change**: set `NUMBA_OPT=0` in the CI test job's environment (1-line
addition to the workflow YAML). Two test files assert on vectorized/SVML
output and read ambient env vars via `TestCase.run_test_in_subprocess`
(`numba/tests/test_vectorization.py`, `numba/tests/test_svml.py`); they need
an explicit `"NUMBA_OPT": "3"` added to their `envvars` dicts (~2-4 lines
total) so they keep testing real vectorization.
`test_optimisation_pipelines.py` already forces `OPT=0` itself via
`override_config`, so it's unaffected either way.

### Measured impact (this branch, real `unittest` runs, correctness verified)

| Test | OPT=3 (default) | OPT=0 | Change |
|---|---:|---:|---:|
| `test_np_functions.TestNPFunctions.test_isin_2` | 220.5s | 121.6s | **-44.9%** |
| `test_linalg.TestBasics.test_outer` | 94.1s | 16.7s | **-82.3%** |

Both runs passed all assertions (`OK`) under `OPT=0` — this is purely a
codegen-aggressiveness knob, it does not change compiled semantics.

Toggling `NUMBA_LOOP_VECTORIZE=0` alone (keeping `OPT=3`) only saved ~5% on
the isin microbenchmark, confirming the win comes from skipping the general
O3 pass (inlining, GVN, instcombine, etc. across the whole heavily-inlined
numpy-overload module), not specifically from loop vectorization.

### Confirmed against the latest released Numba (0.65.1 from PyPI)

Same effect reproduces on the officially released version, not just this
branch — i.e. this is a longstanding, general characteristic of Numba's
codegen pipeline, not a regression introduced by unmerged work:

| | OPT=3 (default) | OPT=0 | Change |
|---|---:|---:|---:|
| Numba 0.65.1, cold `np.isin` compile | 11.46s | 7.51s | **-34.5%** |

### Estimate for the full suite

Compile time is 89.2% of total suite time; the two measured cases show
45-82% compile-time reduction. Extrapolating conservatively (45% reduction
applied only to compile time) suggests **~30-35% lower total CI wall-clock**
for the Linux/macOS test jobs. This is an estimate — running the full
5,400-test suite end-to-end in both configurations was outside this
environment's time budget; the isin and outer measurements are real,
full-test-method timings, not extrapolated microbenchmarks.

**Risk: low.** Test-only knob; does not touch the shipped library's runtime
optimization level. Contained blast radius: 2 files need an explicit
override to preserve their vectorization assertions.

## Candidate 2: extend `_NUMBA_REDUCED_TESTING=1` to Linux/macOS CI

Numba already has a `@skip_if_reduced_testing` mechanism
(`numba/tests/support.py`) gated on the `_NUMBA_REDUCED_TESTING` env var, but
it's currently enabled **only for Windows** CI jobs
(`buildscripts/azure/azure-windows.yml`,
`buildscripts/condarecipe.local/run_test.bat`,
`.github/workflows/numba_win-64_wheel_builder.yml`). Only two test files use
the decorator today, on `test_isin_3a`, `test_isin_3b`, and `test_isin_4` —
three of the four biggest compile-time offenders in the gist, **948.6s
(11.4% of the total suite)**.

**Change**: add `_NUMBA_REDUCED_TESTING=1` to the Linux/macOS test job env
(1 line per workflow), mirroring what Windows already does.

**Risk: medium.** This is a real coverage trade-off, not free — it skips the
exhaustive dtype/list-type parametrization for `isin`, same as it already
does on Windows. Recommended only if paired with a full-coverage run
elsewhere (nightly/scheduled), and it should be considered a smaller,
narrower complement to Candidate 1 rather than a replacement — the two
overlap on the isin tests but Candidate 1 covers everything else too (clip,
outer, percentile, sort, convolve, etc.) that reduced-testing doesn't touch.

## What we didn't find

No single-function source bug in `numba/np/arraymath.py` (or the other hot
modules) stood out as a discrete, low-risk 1-5 line fix — `_in1d_impl` and
similar overloads are algorithmically reasonable; the cost is structural
(every branch of every numpy-overload call chain gets fully typed, lowered,
and O3-optimized per type specialization, and the isin/percentile/sort tests
exercise dozens of specializations by design). The two candidates above
target that structural cost from the CI-configuration side rather than
rewriting the implementations themselves.

## Reproducing these numbers

```
NUMBA_OPT=0 python -m unittest numba.tests.test_np_functions.TestNPFunctions.test_isin_2
NUMBA_OPT=0 python -m unittest numba.tests.test_linalg.TestBasics.test_outer
```
