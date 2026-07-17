# Numba test compile-time analysis

Source data: gist `0.67-times-numba-linux-64.txt`
(https://gist.github.com/swap415/6aee3d3742fcdab493c1c6fe2c2058c4), an
instrumented Linux-64 test run capturing per-test `Duration` / `Compile`
(with pass and codegen sub-splits) / `Run` timings. Instrumentation
approach reviewed from https://github.com/swap357/numba/pull/110
(`CompileTimeTracker` in `numba/misc/compiletimeutils.py` + CI upload +
`analyze_compile_times.py` / `chart_compile_times.py`). **That PR is
measurement-only** — it adds no compiler or test changes, so it does not
itself move any of the numbers below.

## Where the time goes

The 25 highest-compile-time tests in the gist account for ~509s (~8.5min)
of compile time by themselves, and they are all array-method/reduction
tests exercising many distinct type signatures against
`numba/tests/test_array_methods.py`, `test_array_manipulation.py`, and
`test_array_reductions.py`:

| Test | Compile | Calls | Notes |
|---|---|---|---|
| `test_clip_array_min_max` | 101.4s | 170 | 6 pyfuncs × ~15 (min,max) type combos, each a new signature |
| `test_fill_diagonal_basic` | 63.5s | 60 | shape × dtype × value-type sweep |
| `test_sum_axis_dtype_kws` | 51.2s | 46 | dtype × out_dtype × axis sweep |
| `test_array_transpose_axes` | 26.0s | 58 | axis-permutation sweep |
| `test_sum_exceptions` | 23.9s | 2 | only 2 signatures, ~12s/compile each (large generated body for axis-reduction over a 4D array) |
| `test_np_where_3_broadcast_x_or_y_scalar` | 30.2s | 40 | |
| `test_take` | 22.9s | 29 | |
| `test_argmax_axis_1d_2d_4d` | 19.0s | 14 | |

Common pattern: a single `pyfunc` is wrapped once with
`jit(nopython=True)(pyfunc)`, then invoked in nested loops over dtypes /
axes / kwarg combos. Each distinct argument-type combination forces a
fresh nopython specialization, so total test time is dominated by LLVM
compilation (`passes` + `codegen`), not by the actual `Run` time (which is
consistently sub-second).

## Candidates benchmarked (against numba 0.66.0, latest PyPI release, since
the local repo checkout requires an unreleased llvmlite that isn't
buildable in this sandbox)

### 1. Lower `NUMBA_OPT` for the test suite (highest impact, ~1 line)

Numba's default LLVM optimization level (`NUMBA_OPT=3`) runs a full -O3
pass pipeline on every compiled specialization. Test code only needs to be
*correct*, not fast, so the test runner (or CI workflow) can set a lower
level without touching the library's runtime default for end users.

`benchmarks/compile_time_analysis/bench_opt_level.py`, combined sum/clip/where/transpose sweep:

| NUMBA_OPT | Total | Δ vs default |
|---|---|---|
| 3 (default) | 47.49s | — |
| 1 | 33.86s | **-29%** |
| 0 | 30.09s | **-37%** |

Biggest single win is on `np.sum(axis=..., dtype=...)`-style code
(23.00s → 9.86s, **-57%**), which matches `test_sum_axis_dtype_kws`,
`test_sum_axis_kws1`, and `test_argmax_axis_1d_2d_4d` in the gist —
together already >89s of compile time in one run.

**Change**: set `NUMBA_OPT=1` (safer than 0, keeps mem2reg/inlining) in
the CI workflow env or in the test entry point
(`os.environ.setdefault("NUMBA_OPT", "1")`). No production code touched.

**Benefits most**: `test_sum_axis_dtype_kws`, `test_sum_axis_kws1`,
`test_argmax_axis_1d_2d_4d`, `test_clip_array_min_max`,
`test_np_where_*`, `test_array_transpose_axes`, `test_take`,
`test_fill_diagonal_basic` — i.e. essentially the whole top-25 list,
since LLVM pass time is the dominant cost for all of them.

### 2. Add `cache=True` to the hot test-module jit wrappers (~1 line per site)

None of the 98 `jit(nopython=True)(pyfunc)` call sites across
`test_array_methods.py` (44), `test_array_manipulation.py` (32), and
`test_array_reductions.py` (22) use `cache=True`.

`benchmarks/compile_time_analysis/bench_cache.py`, 12-signature sum sweep:

| Run | Time |
|---|---|
| Cold (writes disk cache) | 6.742s |
| Warm (reads disk cache) | 0.225s (**~30x faster**) |

This doesn't shrink a single from-scratch full-suite run, but it
directly addresses repeated-invocation cost: local dev iteration
(re-running the same test file while debugging), CI legs that persist
`NUMBA_CACHE_DIR` between runs (e.g. via `actions/cache`), and reruns of
flaky/retried tests.

**Change**: `cfunc = jit(nopython=True, cache=True)(pyfunc)` at the ~10
hottest call sites (start with the ones in the table above), plus a
`NUMBA_CACHE_DIR`-persisting `actions/cache` step in CI for the payoff to
extend across CI runs, not just local reruns.

**Benefits most**: any test rerun scenario — local iteration on
`test_array_methods.py` / `test_array_manipulation.py` /
`test_array_reductions.py`, and CI if paired with a persisted cache dir.

### Lower-confidence / not recommended as "low-hanging"

`test_sum_exceptions` (23.9s across only 2 calls, ~12s/compile) and the
combinatorial dtype/axis/out_dtype sweeps in `test_sum_axis_dtype_kws`
point at genuinely large generated-code bodies for N-D axis reductions —
fixing that is a compiler-level change, not a 1-5 line one, so it's out of
scope here. Trimming the tested type-combination matrix would also help
compile time but reduces type-coverage guarantees, so it's a
coverage/time tradeoff for maintainers to decide, not something to apply
unilaterally.

## Caveats

- The gist's full line count/total-suite sum could not be reliably
  extracted (network policy in this environment blocks fetching the raw
  gist file directly; only the rendered HTML page was fetchable, which an
  intermediate summarization step handles inconsistently on very large
  files). The top-25 by-compile-time figures above were cross-checked
  across two independent fetches and matched to the millisecond, so
  they're trustworthy; whole-suite totals are not quoted here for that
  reason.
- Benchmarks above run representative kernels mirroring the real tests'
  call patterns against numba 0.66.0 (PyPI), not the exact dev-branch
  checkout the gist was generated from (0.67-dev) — that checkout needs
  an unreleased llvmlite (`>=0.49.0dev0,<0.50`) with no prebuilt wheel
  available in this sandbox, so it can't be built here. Relative
  improvements should transfer directly since neither change depends on
  version-specific compiler internals.
