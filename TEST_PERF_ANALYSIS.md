# Numba test suite performance analysis

Source data: full-suite compile/runtime metrics captured with the
`numba/misc/compiletimeutils.py` instrumentation from
[swap357/numba#110](https://github.com/swap357/numba/pull/110)
(`_NUMBA_TEST_COMPILE_TIMING=1`), 10,715 test entries, Linux x86_64.

Aggregate totals across the captured run: **10,636s wall time**, of which
**7,770s (73%) is JIT compile time** and 2,866s is actual test runtime.
21,329 individual compile episodes were recorded.

## How the instrumentation works (PR #110)

`CompileTimeTracker` wraps a test in `numba.core.event` listeners for the
`numba:compile`, `numba:run_pass`, and `numba:llvm_lock` events already
emitted by Numba's compiler pipeline. No new instrumentation points are
added to the compiler itself — it just records wall time for each event
and reports `compile`, `passes` (pass pipeline), and `codegen` (LLVM) time
per test, plus a top-level compile call count. `run` time is derived as
`duration - compile`. CI workflows set `NUMBA_TEST_COMPILE_TIMING=1`,
extract the per-test summary lines with `awk`, and upload them as
artifacts; `ci_debug/analyze_compile_times.py` and `chart_compile_times.py`
rank/plot the results. This is pure measurement — it does not itself
change any compile or runtime behavior.

## Finding #1 (implemented): enable `_NUMBA_REDUCED_TESTING` on Linux/macOS CI

Numba already has a test-breadth knob, `_NUMBA_REDUCED_TESTING`
(`numba/tests/support.py:127`), that trims combinatorial
dtype/shape/kwarg parametrization in the heaviest tests. It is currently
enabled **only** on Windows CI
(`.github/workflows/numba_win-64_wheel_builder.yml`,
`buildscripts/azure/azure-windows.yml`,
`buildscripts/condarecipe.local/run_test.bat`) — Linux and macOS wheel-builder
workflows run the full, unreduced matrix.

Today it only gates `numba/tests/test_np_functions.py`, but that one file
alone accounts for **~20.6% of total suite compile time** (1,599s of
7,770s) in the captured run, concentrated in:

| test | compile time | calls |
|---|---:|---:|
| `test_isin_3a` | 319.2s | 46 |
| `test_isin_4` | 317.0s | 46 |
| `test_isin_3b` | 312.3s | 40 |
| `test_isin_2` | 310.8s | 40 |
| `test_vander_basic` | 55.7s | 66 |
| `test_argpartition_fuzz` | 42.9s | 23 |
| `test_ediff1d_basic` | 49.3s | 48 |
| `test_extract_basic` | 38.2s | 75 |
| `test_partition_fuzz` | 32.4s | 23 |
| `test_argpartition_basic` | 28.6s | 10 |
| `test_repeat` | 26.1s compile + 15.4s run | 50 |
| `test_searchsorted` | 23.3s | 51 |
| (+ several smaller: `test_partition_basic`, `test_searchsorted_supplemental/complex`, `test_sinc`, `test_angle`, `test_histogram`) | | |

### Change

One line added to the `env:` block of the "Validate and test wheel" step
in each of the three Linux/macOS wheel-builder workflows (mirrors what
Windows already does):

```yaml
      - name: Validate and test wheel
        env:
          _NUMBA_REDUCED_TESTING: 1
```

Applied to `numba_linux-64_wheel_builder.yml`,
`numba_linux-aarch64_wheel_builder.yml`, and
`numba_osx-arm64_wheel_builder.yml`.

### Measured before/after (this sandbox, real test execution)

Built this branch locally (numpy 2.4.6, llvmlite 0.48.0) and ran the
gated tests directly with `python -m unittest`, toggling the env var:

**`test_isin_2/3a/3b/4`** (the single biggest offender):

| | wall time |
|---|---:|
| baseline (`_NUMBA_REDUCED_TESTING` unset, current Linux CI behavior) | 505.9s |
| with `_NUMBA_REDUCED_TESTING=1` | 16.5s |
| **speedup** | **~30x**, saves ~489s |

Note: with the flag set, `test_isin_2` runs a reduced 4-case matrix;
`test_isin_3a/3b/4` are skipped outright (`@skip_if_reduced_testing`) since
they exercise the `assume_unique`/`invert` kwargs on the same underlying
`_in1d_impl`, whose kwarg-handling is already covered by `test_in1d_*`.

**`test_vander_basic`, `test_argpartition_fuzz/basic`, `test_ediff1d_basic`,
`test_extract_basic`, `test_repeat`** (6 tests):

| | wall time |
|---|---:|
| baseline | 209.9s |
| with `_NUMBA_REDUCED_TESTING=1` | 58.1s |
| **speedup** | **~3.6x**, saves ~152s |

Combined measured savings for these 10 tests alone: **~641s** (baseline
715.8s → 74.6s). Since the full `test_np_functions.py` module accounts for
1,599s of the 7,770s total suite compile time, and the untested remainder
of the gated tests follow the same pattern (fixed setup cost + N
recompiles removed per fixture), extrapolated full-module savings are in
the same ~1,300–1,500s range — i.e. roughly **12–15% off total suite
duration** for a one-line-per-workflow change with zero source-code risk
(the mechanism is already battle-tested in production on Windows CI).

### Tradeoff (be upfront about this)

This is a coverage/speed tradeoff, not a free lunch: `_NUMBA_REDUCED_TESTING`
narrows the dtype/shape/kwarg matrix these tests exercise (and skips a few
tests entirely, e.g. `test_isin_3a/3b/4`). It reduces CI wall-clock and
memory pressure at the cost of fewer type combinations tested on Linux/macOS.
Given it is already the accepted tradeoff for Windows CI, extending it to
Linux/macOS is low-risk, but maintainers should confirm they're comfortable
with the same coverage reduction on the platforms that most users actually
run in production.

## Other candidates identified (not yet implemented — need more investigation)

These show the same "combinatorial parametrization inflates compile time"
pattern as above but aren't currently gated by any reduced-testing knob, so
turning them down needs a small (not yet written) code change per test
rather than a config flip:

| test(s) | compile time | calls | notes |
|---|---:|---:|---|
| `test_percentile_basic` / `test_nanpercentile_basic` / `test_quantile_basic` / `test_nanquantile_basic` | ~65–69s each (267s combined) | 20–21 each | `numba/tests/test_array_reductions.py`; shared `check_percentile_and_quantile`/`check_percentile_edge_cases` helpers — worth checking if edge-case coverage can reuse a single compiled signature instead of recompiling per q-type variant. |
| `test_setitem`/`test_getitem` (`TestFancyIndexing`, `TestFancyIndexingMultiDim`) | 62–64s / 32–34s each (~192s combined) | 91–92 each | `numba/tests/test_fancy_indexing.py`; large indexing-mode combinatorial matrix. |
| `test_comparisons` (`TestDatetimeArithmetic`, `TestDatetimeArithmeticNoPython`) | 32–35s each | **871 calls each** | `numba/tests/test_npdatetime.py`; by far the highest per-test compile-call count in the suite — comparison operator × datetime-unit matrix. Low per-call cost (~37–40ms) but sheer signature count adds up; candidate for trimming the unit/operator cross-product. |
| `test_clip_array_min_max` | 101.4s | 170 | `numba/tests/test_array_methods.py`; 6 clip variants × min/max combinations. |
| `test_outer` (`numba/np/linalg.py`) | 94.2s | 56 | Highest per-call average compile cost found (~1.68s/call) among frequently-called tests. |

These are flagged as **follow-up work**, not delivered fixes: each would
need someone familiar with the specific test's coverage intent to judge
which combinations are safe to drop or gate, the same way `test_isin_*`
already does.

## Not pursued

No single-line/few-line fix was found inside the Numba *implementation*
itself (e.g. in `arraymath.py`'s `_in1d_impl`, or the ufunc-loop codegen in
`npyimpl.py`) that would cut compile time without a deeper redesign — the
dominant cost is LLVM codegen/optimization time scaling with the number of
distinct type signatures compiled, which is inherent to the sort/searchsorted-based
algorithms these functions use and not fixable by a small patch. The
actionable lever, for now, is reducing how many signatures the test suite
asks the compiler to produce.
