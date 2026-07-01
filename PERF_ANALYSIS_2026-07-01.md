# Numba test-suite performance analysis (2026-07-01)

Source data: [compile/runtime metrics gist](https://gist.github.com/swap415/6aee3d3742fcdab493c1c6fe2c2058c4)
(`0.67-times-numba-linux-64.txt`), cross-referenced against
[swap357/numba#110](https://github.com/swap357/numba/pull/110) (the
`CompileTimeTracker` instrumentation used to produce that data).

**Data caveat**: the gist is 1.79MB; the raw host (`gist.githubusercontent.com`)
is blocked by this environment's network policy and the GitHub API truncates
gist content at 900KB, so this analysis covers **5,400 of ~10,700 test
records** (the first alphabetical half of the suite, spanning `doc_examples`
through `test_unpack_sequence`). All modules that turned out to matter
(`test_np_functions`, `test_array_reductions`, `test_sort`, `test_linalg`,
etc.) are fully represented in the sample.

## What the data shows

| Metric | Total (5,400 tests) | Share |
|---|---|---|
| Wall duration | 8,341.9s | 100% |
| Compile time | 7,440.3s | 89.2% |
| — of which type inference/lowering ("passes") | 7,422.7s | — |
| — of which LLVM codegen/optimization | 5,393.1s | 64.6% of total, 72.5% of compile |
| Run time | 901.6s | 10.8% |

**Compilation, not execution, dominates the suite** — expected for a JIT
compiler test suite, but the split confirms LLVM codegen/optimization
specifically (not typing) is the largest single line item.

Biggest single contributors: `numba.tests.test_np_functions` (2,606s, 31% of
the whole sample), driven almost entirely by `test_isin_2/3a/3b/4`
(1,262s combined — `np.isin` compiled fresh 40-46 times per test, once per
distinct argument-type combination in the parametrized test data).
`test_array_reductions` (566s, percentile/quantile), `test_array_methods`
(479s), `test_sets` (427s), and `test_fancy_indexing` (264s) follow.

## PR #110 review

The referenced PR is a **draft instrumentation PR**, not a performance fix —
it adds `numba/misc/compiletimeutils.py` (a `CompileTimeTracker` that hooks
the compiler pipeline to record per-test compile/pass/codegen timing), CI
workflow changes to capture and upload these metrics, and analysis/charting
scripts (`analyze_compile_times.py`, `chart_compile_times.py`,
`download_compile_time_artifacts.py`). It's the tool that produced the gist,
not a candidate for optimization itself.

## Findings

### 1. [Implemented] Skip the redundant per-function O3 LLVM pass — `numba/core/codegen.py`

**The bug**: `CPUCodeLibrary._optimize_final_module` runs LLVM optimization
**three times** per compile:

1. `_optimize_functions()` — a per-function pass at `config.OPT` (default
   **O3**) run in a loop over every function in the module, *before* linking.
2. A module-level "cheap" pass at `self._codegen._opt_level` (**O0** by
   default) for NRT ref-op inlining/pruning.
3. `mpm_full` — a **full O3 module-level pass** over the whole linked module.

Step 1 and step 3 both run the expensive O3 pipeline, and step 3 fully
re-optimizes everything step 1 already touched (that's its job — it also
catches cross-function/inlining opportunities the per-function pass can't
see). Step 1's own docstring says its purpose is just "to reduce memory usage
and improve module-level optimization," not to fully optimize — so running
it at the same cheap opt level already used for the module-level pre-pass
(step 2) removes the duplicated O3 work while leaving the actual optimization
(step 3) untouched.

**The fix** (5 lines, `numba/core/codegen.py:659-666`):

```python
fpm, pb = self._codegen._function_pass_manager()
```
→
```python
fpm, pb = self._codegen._function_pass_manager(
    opt=self._codegen._opt_level)
```

**Correctness**: verified against `numba.tests.test_np_functions` (`test_isin_2`,
`test_correlate`), `numba.tests.test_linalg.TestBasics.test_outer`, and the
full `numba.tests.test_dispatcher` module (49 tests, including
`inspect_asm`/`inspect_llvm`/`inspect_cfg`, which read the post-optimization
LLVM IR/assembly directly) — all pass unchanged. Since the module still gets
a full O3 pass afterward, final generated code is unaffected; only the
now-redundant intermediate pass is cheapened.

**Benchmarked** (cold-compile wall time, single specialization, this
environment, mean of 4 runs; "dev" = this branch pre/post patch, "released"
= PyPI `numba==0.65.1`, the latest release):

| Function (test it dominates) | released 0.65.1 | dev baseline | dev + patch | patch vs baseline | patch vs released |
|---|---:|---:|---:|---:|---:|
| `np.isin` (`test_isin_*`, 1,262s of sample) | 11.908s | 11.365s | 10.997s | **-3.2%** | -7.6% |
| `np.percentile` (`test_array_reductions`) | 5.448s | 5.016s | 4.771s | **-4.9%** | -12.4% |
| `np.correlate` (`test_np_functions`) | 1.617s | 1.426s | 1.396s | **-2.1%** | -13.7% |
| `np.outer` (`test_linalg.test_outer`, 94s) | 2.263s | 2.024s | 1.575s | **-22.2%** | -30.4% |
| `ndarray.sort` (`test_sort`, 201s) | 1.251s | 1.189s | 1.138s | **-4.3%** | -9.0% |

`np.outer` benefits most (fewest functions get inlined into it before
linking, so the now-cheapened per-function pass previously did
proportionally more "wasted" O3 work relative to its total compile time).
`np.isin`'s absolute win is smaller in percentage but largest in absolute
seconds given it's the single biggest line item in the suite (46 calls ×
~0.3-0.4s saved ≈ 15-18s per test, ×4 isin tests ≈ 60-70s off the sampled
suite alone).

**Tests most likely to benefit**: anything with many distinct
type-specializations compiled in one test — `test_isin_2/3a/3b/4`,
`test_array_reductions` (percentile/quantile/nanpercentile/nanquantile family,
20-21 calls each), `test_fancy_indexing` (91-92 calls), `test_np_randomgen`,
and any downstream user code that JIT-compiles many overloads (e.g.
`@overload`-heavy user libraries), not just this test suite.

### 2. Investigated, not low-hanging: `_in1d_impl` (backs `isin`/`in1d`/`setdiff1d`)

`test_isin_2/3a/3b/4`'s 40-46 "compile calls" per test are **not** a caching
bug — I verified empirically that identical Numba argument types reuse the
existing compiled specialization (3 calls with different list *values* but
the same type → 1 signature, ~11s to compile once, 0s after). The 40-46
calls correspond to 40-46 genuinely distinct type signatures in the test's
`_isin_arrays()` parametrization (mixing scalars, 0-d/1-d/2-d/3-d arrays,
typed lists, and 8 dtype combinations). Each compile is legitimately
expensive because `_in1d_impl` inlines *two full algorithm paths* (a linear
mask-scan and a sort-based unique/argsort/concatenate path) selected by a
runtime-only condition (`len(ar2) < 10*len(ar1)**0.145`), so both get fully
compiled and optimized every time regardless of which one executes. There's
no small change that fixes this without either changing the algorithm or
accepting a runtime regression — flagging it so a future pass doesn't waste
time chasing what looks like a bug but isn't.

### 3. Checked, already optimal: NRT ref-count pruning

`NUMBA_LLVM_REFPRUNE_PASS` defaults to `1` (native LLVM pass), so the slow
pure-Python `remove_redundant_nrt_refct` fallback in
`numba/core/runtime/nrtopt.py` is already dead code on the default
configuration used to produce this data — not an opportunity.

## Next steps

- The `percentile`/`quantile`/`nanpercentile`/`nanquantile` family
  (`test_array_reductions`, 68s × 4 variants ≈ 272s of the sample) and
  `test_fancy_indexing.test_setitem` (91-92 calls, 64s each) look like they
  share the same "compile many genuinely distinct specializations" pattern
  as `isin` — worth a follow-up pass to check whether their shared helpers
  also inline multiple full algorithm branches unconditionally the way
  `_in1d_impl` does.
- Re-run this analysis against the **full** gist once raw-gist network access
  is available, to confirm the second half of the suite (alphabetically past
  `test_unpack_sequence`) doesn't surface a bigger opportunity than `isin`.
- Land [PR #110](https://github.com/swap357/numba/pull/110)'s instrumentation
  in CI so before/after compile-time deltas (like the table above) are
  tracked automatically per-PR instead of one-off gists.
