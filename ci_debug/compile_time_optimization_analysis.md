# Numba test-suite compile-time analysis

Source data: Linux 64-bit compile-time report (`0.67-times-numba-linux-64.txt`,
captured 2026-06-30 via the instrumentation from
[swap357/numba#110](https://github.com/swap357/numba/pull/110)), 5,400 parsed
test entries.

## Instrumentation approach (PR #110)

`numba/misc/compiletimeutils.CompileTimeTracker` wraps each test in a context
manager that installs three `numba.core.event` listeners:

- `numba:compile` — the whole top-level JIT compilation episode
- `numba:run_pass` — the compiler pass pipeline (typing, lowering, ...)
- `numba:llvm_lock` — LLVM codegen, held under the LLVM global lock

Containment is `codegen <= passes <= compile <= duration`. `numba/testing/main.py`
attaches a tracker per test in `SerialSuite`, warms up Numba once before timing
starts, and prints one `Name | Duration | Compile (...) | Run` line per test
under `NUMBA_TEST_COMPILE_TIMING=1`.

## Suite-wide totals (5,400 tests)

| Metric | Total | Share of duration |
|---|---|---|
| Duration | 8,341.9 s | 100% |
| Compile | 7,440.3 s | 89.2% |
| — of which passes | 7,422.7 s | 99.8% of compile |
| — of which codegen (LLVM lock) | 5,393.1 s | 72.5% of compile / 64.6% of duration |
| Run (post-compile execution) | 901.6 s | 10.8% |

Compilation dominates: **runtime is only ~11% of total suite time**, and LLVM
codegen alone accounts for **~65% of all wall-clock time in the suite**. This
means the highest-leverage optimizations target *compile* cost, not the
compiled code's execution speed — a very different tuning target than
end-user Numba performance work.

### Top compile-time offenders

| Test | Compile time | Calls (signatures) |
|---|---|---|
| `test_np_functions.test_isin_3a` | 319.2 s | 46 |
| `test_np_functions.test_isin_4` | 317.0 s | 46 |
| `test_np_functions.test_isin_3b` | 312.3 s | 40 |
| `test_np_functions.test_isin_2` | 310.8 s | 40 |
| `test_array_methods.test_clip_array_min_max` | 101.4 s | 170 |
| `test_linalg.test_outer` | 94.2 s | 56 |
| `test_np_functions.test_correlate` | 72.9 s | 51 |
| `test_array_reductions.test_nanpercentile_basic` | 68.6 s | 21 |
| `test_array_reductions.test_percentile_basic` | 68.2 s | 21 |
| `test_array_reductions.test_nanquantile_basic` | 65.7 s | 20 |

These tests each call `njit(fn)` once and then invoke it across dozens of
argument-type combinations (dtype × container-type matrices), so each entry's
compile time is really *N* independent full compiles of one fused,
numpy-heavy implementation (`np.isin`/`_in1d_impl` chains `argsort`,
`unique`, `concatenate`, `cumsum`...). That per-signature cost is inherent to
Numba's specialize-per-signature model and isn't fixable with a small patch
without cutting test coverage, so it is **not** one of the recommendations
below — it's noted here because it explains why these particular tests
dominate the ranking.

## Root cause of the codegen cost

`numba/core/codegen.py` (`CPUCodegen._init`, `_optimize_final_module`) runs
**two** LLVM module-level optimization passes per compiled function:

1. A cheap pass at `opt=0` (`self._opt_level`, unless `NUMBA_OPT=max`) to
   maximize inlining before NRT refcount-op pruning.
2. A **full** pass at `config.OPT`, which defaults to **`3`** — LLVM's O3
   pipeline (loop rotation/vectorization, SLP vectorization, aggressive
   inlining, etc.) — run on every single JIT compilation, in every test.

None of the CI workflows that produced this data (or any other workflow in
the repo) sets `NUMBA_OPT`, so every test compile pays full `-O3` codegen
cost even though the tests only check correctness, never runtime speed.

## Recommendation (validated)

### 1. Set `NUMBA_OPT=1` (or `0`) for CI test runs — 1-line change, no source patch

This is a pre-existing, documented, already-supported env var
(`numba/core/config.py:308`) — no numba source changes needed, only the test
invocation in `.github/workflows/numba_{linux-64,linux-aarch64,osx-arm64,
win-64,win-arm64}_wheel_builder.yml` (the same 5 files PR #110 instruments),
e.g.:

```yaml
- run: NUMBA_OPT=1 $PYTHON_PATH -m numba.runtests -m 4 -v
```

**Benchmarked** on this branch and independently against **Numba 0.66.0**
(latest PyPI release, installed in a clean venv), same machine, same
representative functions pulled from the offender list above (5 dtype
signatures per function, LLVM-lock/codegen timed via the actual
`numba:llvm_lock`/`numba:compile` events):

| Function | OPT=3 (default) compile | OPT=1 compile | Δ | OPT=1 codegen Δ |
|---|---|---|---|---|
| `np.outer` | 8.97 s (dev) / 8.72 s (0.66.0) | 2.49 s / 2.48 s | **−72%** | −81% |
| `np.correlate` | 8.07 s / 7.32 s | 6.46 s / 6.51 s | **−20%** | −27% |
| `np.clip` | 0.99 s / 1.39 s | 0.73 s / 1.18 s | **−16 to −26%** | −26 to −38% |
| `np.isin` (3-sig subset of `test_isin_3a`) | 25.0 s / 25.6 s | 22.1 s / 22.5 s | **−12%** | −18% |

At `NUMBA_OPT=0`: `np.outer` −80%, `np.correlate` −31%, `np.clip` −19 to
−37%, `np.isin` subset −33%. Gains scale with how much loop/SLP
vectorization the function's IR offers LLVM to chew on — dense array ops
(`outer`, `clip`) benefit most; branch/sort-heavy code (`isin`) benefits
least.

**Correctness check**: ran the full parametrized `test_isin_3a` and
`test_isin_2` (all 86 dtype/container signatures, not the 5-signature
subset) at `NUMBA_OPT=1` — both pass unchanged (242 s combined on this
machine).

**Suite-wide estimate**: applying the observed 20–70% compile-time range
conservatively (~30% blended average, weighted toward the lower end since
codegen is only 72.5% of compile time and not every op vectorizes as well as
`outer`) to the 7,440 s compile total projects total suite duration
8,342 s → roughly **6,100–6,700 s, a ~20–27% cut in total CI wall-clock time**,
entirely from a one-line env var.

**Trade-off to flag explicitly**: this changes what CI validates — an O3
pipeline exercises loop-vectorize/SLP-vectorize/unroll paths that O1/O0
never touch, so a small category of LLVM-miscompilation-under-vectorization
bugs would stop being caught by every job. Recommend keeping at least one
full-O3 job (e.g. a nightly/scheduled build) at the default, and only
lowering `NUMBA_OPT` on the fast per-PR gating jobs.

### 2. (Lower confidence, not benchmarked) Test-runner parallelism

The wheel-builder workflows already run `numba.runtests -m 4` (4 worker
processes). If CI runners have more than 4 cores available, raising `-m`
would cut wall-clock further — but the runner core counts aren't visible
from this environment, so this is a "worth checking," not a validated
recommendation the way #1 is.

## Ideas considered and rejected as "low-hanging fruit"

- **`_check_llvm_bugs()` locale-check re-run on every `finalize()`**
  (`codegen.py`): measured at 47.5 µs/call; over the suite's ~18.5k compile
  episodes that's <1 s total — not worth touching.
- **Reducing `isin`/`percentile`/`correlate` per-signature compiles**: the
  cost is the combinatorial dtype/container test matrix hitting a genuinely
  complex fused numpy-array implementation once per signature. Shrinking it
  either cuts test coverage or requires reworking the overloads themselves
  (a substantially larger change than "1–5 lines"), so it's out of scope
  here.
- **`NUMBA_LLVM_REFPRUNE_PASS`**: already enabled by default; no lever.

## Reproduction

Benchmark scripts and the parsed gist data used for this analysis are not
included in this commit (they were run from the session scratchpad); the
methodology is: install target Numba build, `njit` the offending function
once, call it across its test's dtype matrix, and read `compile`/`codegen`
directly off the `numba:compile`/`numba:llvm_lock` event listeners (the same
API `CompileTimeTracker` from PR #110 wraps).
