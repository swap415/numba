# Numba test-suite compile-time analysis

Source data: `0.67-times-numba-linux-64.txt` gist (Linux 64-bit compile/run
metrics captured via the per-test instrumentation format introduced in
[swap357/numba#110](https://github.com/swap357/numba/pull/110):
`Name: <id> | Duration: t | Compile: t (n calls; passes t, codegen t) | Run: t`).

Numbers below marked "measured locally" were reproduced independently in
this environment (not taken from the gist) using this repo's checkout
(`claude/beautiful-newton-mbfuyp`, ~1622 commits past the last tag) and,
for comparison, the latest PyPI release, **numba 0.66.0**.

## What the gist shows

Across nearly every slow-compiling test, the `codegen` component (LLVM
module-level optimization) dominates `compile` time, not Numba's own
typing/lowering (`passes`):

| test | compile_ms | codegen_ms | codegen share |
|---|---:|---:|---:|
| `test_clip_array_min_max` | 101,400 | 80,027 | 79% |
| `test_sum_axis_dtype_kws` | 51,220 | 40,320 | 79% |
| `test_fill_diagonal_basic` | 63,512 | 55,168 | 87% |
| `test_array_transpose_axes` | 25,959 | 21,400 | 82% |
| `test_take` | 22,886 | 16,658 | 73% |
| `test_sum_exceptions` | 23,898 | 23,041 | **96%** |

`test_sum_exceptions` stands out: only **2** distinct compiles account for
23.9s. That's not combinatorial parametrization (like `test_clip_*`, which
has 170 calls) — it's two individual functions taking ~12s each to
optimize.

## Root cause

`CPUCodeLibrary._optimize_final_module` in `numba/core/codegen.py` always
runs LLVM's full module optimization pass at `config.OPT` (env var
`NUMBA_OPT`, **default 3** — LLVM's `-O3`-equivalent pipeline: aggressive
inlining, GVN, loop rotation, vectorizer analysis, etc.) on top of a
preliminary "cheap" pass. For most short JIT-compiled kernels this is a
large fixed cost per compile, and for a handful of functions (large
generated reduction/exception code, e.g. `array.sum(axis=...)` on
higher-dimensional arrays) `-O3`'s pass costs scale badly with generated
IR size.

This is Numba's own documented lever
(`NUMBA_OPT`, see `numba/core/config.py:308`) for trading compile time
against generated-code speed — it isn't news to the project — but the
gist data shows the CI test suite is paying the full `-O3` cost on every
single compile, including for code paths that clearly don't need it.

## Candidate 1 (implemented): `NUMBA_OPT=1` for CI test runs

**Change:** one line — set `NUMBA_OPT: "1"` in the `env:` block of the
test job in `.github/workflows/numba_linux-64_wheel_builder.yml` (the job
that runs `python -m numba.runtests`). This only affects the CI test
run's JIT compilation; it does **not** change the shipped package's
default (`NUMBA_OPT=3`) for end users.

This directly targets the instrumentation approach from PR #110: those
workflow files are exactly what the PR modifies to capture compile-time
metrics, so this is a natural complementary change to land alongside it.

**Measured impact** (this repo's checkout, and independently confirmed
on the official `numba==0.66.0` PyPI release, same machine):

| workload | numba 0.66.0 (release), OPT=3 | numba 0.66.0, OPT=1 | speedup |
|---|---:|---:|---:|
| `array_sum(a, axis)` exception-path compile (mirrors `test_sum_exceptions`) | 44.67s | 1.87s | **23.9x** |
| same workload, this repo's dev checkout | 44.82s | 1.88s | **23.8x** |
| `test_fill_diagonal_basic` (full test, pytest) | 39.99s | 33.87s | 1.18x |
| `array.clip` 90-call combinatorial compile subset (mirrors `test_clip_array_min_max`) | 17.76s | 15.97s | 1.11x |

Takeaway: the win is dramatic (20x+) for tests dominated by one or two
expensive-to-optimize functions (reduction/exception code with large
generated IR), and modest (~10-20%) for tests whose cost is spread across
many small, cheap compiles (`test_clip_array_min_max`'s 170 calls,
`test_take`'s 29 calls). Tests most likely to benefit most:
`test_sum_exceptions`, `test_sum_axis_dtype_kws`, `test_sum_const*`,
`test_argmax_axis_1d_2d_4d`, and any other axis-reduction test over
arrays with ndim >= 3 — these are exactly the pattern that produced the
23x result above.

**Runtime-speed tradeoff, measured** (hot numeric loop,
`acc += x[i]*x[i] - 0.5*x[i]` over 2M elements, 30 calls averaged, this
repo's checkout):

| NUMBA_OPT | time/call |
|---|---:|
| 3 (default) | 2497 us |
| 1 | 2511 us (+0.6%, noise) |
| 0 | 30156 us (**12x slower**) |

`NUMBA_OPT=1` gives up essentially none of the default's runtime speed
(the loop-rotate/instruction-combine/jump-threading passes and loop
vectorization that matter for numeric hot loops are still applied at
`opt>=1`; only the most expensive whole-module inlining/analysis passes
are skipped). `NUMBA_OPT=0` is a much worse trade for a general-purpose
CI test run — it saves comparatively little extra compile time here but
craters runtime performance, which risks masking or triggering
performance-flavored test flakiness. **`NUMBA_OPT=1`, not `0`, is the
correct choice**, and is what was implemented.

Status: implemented in `numba_linux-64_wheel_builder.yml` only. The same
one-line change should be validated and rolled out to the other four
wheel/conda builder workflows (aarch64, win-64, osx-arm64, win-arm64)
once the Linux run confirms no correctness fallout; not done here to
avoid changing untested platforms' CI behavior in one shot.

## Candidate 2 (flagged, not implemented): runtime-dominant outliers

A few tests in the gist show the opposite signature — low compile time,
disproportionately high `run` time — which points at the *generated
code* being slow, not the compiler:

| test | compile_ms | run_ms | run/compile ratio |
|---|---:|---:|---:|
| `test_np_frombuffer_dtype` | 612 | 2,798 | 4.6x |
| `test_round_array` | 2,880 | 4,070 | 1.4x |
| `test_around_array` | 2,836 | 2,990 | 1.05x |
| `test_array_reshape` | 11,202 | 6,199 | 0.55x (highest absolute run time) |

These don't have an obvious 1-5 line fix identified yet — `run_ms` here
includes the test's own Python-side assertion/comparison overhead
(`np.testing.assert_equal` over multiple dtype/shape combinations), not
purely generated-code execution, so it's not yet clear how much is
"Numba runtime is slow" versus "the test does a lot of work per case."
Recommend profiling these four specifically (`python -m cProfile` around
just the `cfunc(...)` calls, isolated from the assertion harness) before
proposing a code change.

## Candidate 3 (not pursued): double module-optimization pass

`_optimize_final_module` builds and runs *two* module pass managers back
to back — a "cheap" `opt=0` pass (to maximize inlining scope for
ref-count pruning) followed by the full `opt=config.OPT` pass. When
`NUMBA_LLVM_REFPRUNE_PASS=1` (the default), the native LLVM refprune pass
is added to *both* pass managers, i.e. it runs twice per compile. Whether
one of those two refprune runs is redundant looks plausible from reading
the code, but changing it touches reference-counting correctness, not
just speed, so it needs sign-off from someone with more context on the
ref-op pruner's design intent (see comments at `numba/core/codegen.py`
lines 673-691) before treating it as low-risk. Flagged for follow-up, not
implemented.

## Note on the PR #110 instrumentation

The instrumentation approach (per-test `Name/Duration/Compile/Run`
logging via `NUMBA_TEST_COMPILE_TIMING=1`, `ci_debug/analyze_compile_times.py`
for ranking, `ci_debug/chart_compile_times.py` for charts, artifact
upload per platform) is exactly the right shape for catching regressions
like the ones above over time. It is not yet present on this branch —
only the CI env change above was applied here. Porting the instrumentation
itself (a `CompileTimeTracker` in `numba/misc/compiletimeutils.py` plus the
five workflow edits) is a separate, larger change than the "1-5 line"
scope of this analysis and is left to a follow-up.
