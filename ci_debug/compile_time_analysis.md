# Numba test-suite compile-time analysis

Source data: [compile-time gist](https://gist.github.com/swap415/6aee3d3742fcdab493c1c6fe2c2058c4)
(`0.67-times-numba-linux-64.txt`, ~700 test entries) collected with the
instrumentation added in [swap357/numba#110](https://github.com/swap357/numba/pull/110)
("`test-timing-hooks`" branch). All numbers below were cross-checked by
rebuilding Numba locally (release 0.66.0 from PyPI, llvmlite 0.48.0) and
re-running representative workloads with `cProfile`.

## 1. How the numbers are produced (PR #110)

`numba/misc/compiletimeutils.py` adds a `CompileTimeTracker` that wraps a test
in three of Numba's built-in event listeners:

- `numba:compile` — one top-level event per `Dispatcher.compile()` call (a
  genuinely new signature; cache hits do **not** re-trigger it).
- `numba:run_pass` — the internal compiler pipeline ("passes").
- `numba:llvm_lock` — LLVM codegen/optimization, which runs under a global
  lock and is a subset of "passes" time (`codegen <= passes <= compile`).

`numba/testing/main.py` is patched to run every test under this tracker when
`NUMBA_TEST_COMPILE_TIMING=1`, and the CI workflows for all five wheel-builder
platforms were updated to set that env var, pipe `runtests` output through
`tee`, `awk` out the `=== Compile Times ===` block, and upload it as an
artifact. Two analysis scripts (`analyze_compile_times.py`,
`chart_compile_times.py`) rank/plot the results. This is a clean, low-overhead
way to get per-test compile/run breakdowns without external profilers, and is
the reason the gist data below can be trusted at the "which tests are slow and
why" level.

## 2. Where the time actually goes

Top compile-time offenders from the gist (compile time, call count = number of
distinct signatures compiled):

| Test | Compile | Calls | Passes | Codegen | Run |
|---|---|---|---|---|---|
| `test_clip_array_min_max` | 101.4s | 170 | 100.9s | 80.0s | 0.43s |
| `test_fill_diagonal_basic` | 63.5s | 60 | 63.3s | 55.2s | 0.39s |
| `test_sum_axis_dtype_kws` | 51.2s | 46 | 51.1s | 40.3s | 0.45s |
| `test_np_where_3_broadcast_x_or_y_scalar` | 30.2s | 40 | 30.1s | 20.2s | 0.47s |
| `test_array_transpose_axes` | 26.0s | 58 | 25.8s | 21.4s | 0.26s |
| `test_reduce_zero_axis` | 24.5s | 11 | 24.4s | 13.6s | 0.01s |
| `test_argmax_axis_1d_2d_4d` | 19.0s | 14 | 19.0s | 16.3s | 0.19s |
| `test_ufunc_at_negative_indexes` | 19.3s | 50 | 19.2s | 14.2s | 0.04s |
| `test_reduceat_basic_2d` | 12.0s | 1 | 12.0s | 9.0s | 0.002s |

Two patterns hold across essentially every entry in the full ~700-row file:

1. **`codegen` is consistently 60–85% of `passes`.** LLVM optimization/machine
   code emission, not Numba's own typing/lowering logic, is the dominant cost.
2. **Compile time scales with call count**, i.e. with the number of *distinct
   type signatures* the test exercises (dtype × axis × shape/layout × scalar-
   vs-array argument combinations for NumPy-compatibility coverage), not with
   any single expensive compile.

## 3. Candidates investigated

I looked for 1–5 line changes that would cut this cost and verified each one
empirically (release Numba 0.66.0, llvmlite 0.48.0, reproducing the
`test_clip_array_min_max` pattern with `cProfile` and wall-clock A/B runs)
before treating anything as a real finding:

| # | Candidate | Result |
|---|---|---|
| 1 | Lower the default LLVM optimization level (`NUMBA_OPT`) | **Already at the minimum.** `numba/core/codegen.py:1213-1219` sets `opt=0` ("cheap pass") by default; this was already the cheapest option before we touched anything. |
| 2 | Disable SLP vectorization | **Already disabled by default** (`config.py:354`, `NUMBA_SLP_VECTORIZE` defaults to `0`). |
| 3 | Disable loop vectorization (`NUMBA_LOOP_VECTORIZE=0`) | **No measurable effect.** A/B wall-clock on the clip benchmark: 15.50s / 15.65s (default) vs 15.72s (disabled) — within run-to-run noise. At `OPT=0` the extra vectorization-prep passes in `_module_pass_manager` are gated off already, so this flag has nothing to act on. |
| 4 | Disable the ref-count-pruning pass (`NUMBA_LLVM_REFPRUNE_FLAGS=none`) | **No measurable improvement** (16.80s vs 15.50s baseline — slightly worse, within noise) and it would trade away runtime refcounting efficiency for no compile-time win. Not worth it. |
| 5 | Increase CI test-runner parallelism (`-m 4` → higher) | **No headroom.** All five wheel-builder workflows run on GitHub-hosted `ubuntu-latest` / `macos-14` / `windows-2025` runners, which provide exactly 4 vCPUs — `-m 4` already matches the available cores. |
| 6 | Trim redundant cross-product parametrization in the slow tests (e.g. `test_clip_array_min_max`'s `mins × maxs × pyfunc` grid, `test_sum_axis_dtype_kws`'s dtype/axis grid) | **Not free.** Numba's dispatcher already caches by type signature and does not recompile a signature it has already seen — confirmed via `cProfile` (each of the 170 `test_clip_array_min_max` compiles corresponds to a genuinely distinct type signature: scalar-int / array / `None` for each of `a_min`/`a_max`, crossed with in-place vs. no-`out` variants). Cutting combinations would directly cut tested NumPy-compatibility coverage, not eliminate waste. |

One incidental finding, noted for completeness but **not** a performance issue:
`test_sum_axis_dtype_kws` (`numba/tests/test_array_methods.py:1489-1501`)
builds an `all_test_arrays` list for six float/int/complex dtypes and then
immediately overwrites the variable with a second list (for
`uint32/uint64/bool_`) before the loop runs — so the first list is dead code,
never exercised by `cfunc`/`pyfunc`. It costs a small amount of NumPy array
allocation at test-setup time, not compile time, and removing it doesn't
change the metrics above; more importantly its presence looks like an
unintentional coverage gap (float/int/complex dtypes for this test aren't
actually being checked) rather than a perf lever, so it's a correctness/test-
coverage follow-up, not something this analysis is scoped to fix.

### What the profiler actually shows

Profiling the `test_clip_array_min_max` pattern directly (release Numba, cProfile,
sorted by self time) puts **`llvmlite/binding/ffi.py:210 __call__`** (the ctypes
bridge into LLVM's C++ API) at 6.29s of 15.15s total (41%) across 60,732 calls —
this *is* the "codegen" time the gist reports, and it's native LLVM work, not
Python overhead we can trim. The next-largest Python-side contributors
(`numba/core/ir.py::_rec_list_vars`, `numba/core/event.py::notify/broadcast`,
`copy.deepcopy` in the type-inference/IR-analysis machinery) are all
load-bearing parts of the pass pipeline (use-def analysis, event dispatch,
IR-restart-on-exception support for type inference) — real engineering
targets, but each is a multi-file, correctness-sensitive change, not a 1–5
line patch.

## 4. Conclusion

No safe, verifiable 1–5 line change surfaced that meaningfully improves
compile or run time for the tests the gist flags as slowest. The two
"cheap" compiler knobs available (`NUMBA_OPT`, `NUMBA_SLP_VECTORIZE`) are
already at their fastest settings, the other flags tested show no measurable
effect, CI parallelism already matches available hardware, and the
remaining cost is either genuine LLVM native compilation work or test
parametrization that is intentionally broad for NumPy-compatibility
coverage.

**Suggested next steps**, roughly in order of effort:

1. **Reclassify, don't trim, the heaviest combinatorial tests.** Move the
   handful of multi-minute tests (`test_clip_array_min_max`,
   `test_fill_diagonal_basic`, `test_sum_axis_dtype_kws`, the `np.where`
   family, the reduce/reduceat family) into a slower/optional CI tier (e.g.
   run once on Linux only, skip on the 4-vCPU macOS/Windows legs) instead of
   trying to shrink their coverage — this reduces wall-clock cost without
   giving up correctness coverage anywhere.
2. **Investigate cross-process/on-disk caching for CI**, i.e. whether
   `cache=True` plus a restored (not fresh) `NUMBA_CACHE_DIR` between
   scheduled CI runs on the same runner image would let repeat signatures
   skip LLVM entirely — this is a CI infra change, not a source change, and
   needs measuring cache hit-rate on real runners before committing to it.
3. **Land PR #110's instrumentation itself** — right now this analysis had
   to reconstruct the same profiling ad hoc; merging the `CompileTimeTracker`
   and CI artifact upload would make this kind of regression-hunting a
   standing capability instead of a one-off exercise, and would let this
   exact top-N table be tracked over time to catch new regressions early.
4. **If compile time is worth deeper investment**, the real lever is
   reducing per-specialization LLVM work itself (e.g. investigating whether
   the "cheap" `opt=0` pass set can drop any more passes for straight-line
   numeric kernels, or whether more type signatures can share a single LLVM
   compilation via boxing/generic dispatch for rarely-hot combinations) —
   that's a genuine engineering project, not a quick patch, and would need
   its own design + correctness review.

## Appendix: environment used for verification

- Numba 0.66.0 (PyPI release), llvmlite 0.48.0, NumPy 2.4.6, Python 3.11.15
- Local dev checkout (this branch, commit `11ccb10`) could not be built for
  a source-level A/B comparison: it requires `llvmlite >= 0.48.0dev0`, and at
  HEAD `numba/core/callconv.py:827` (`retarg.add_attribute("nocapture")`)
  raises `ValueError: unknown attr 'nocapture' for ArgumentAttributes()`
  against the released llvmlite 0.48.0 — an unrelated pre-existing
  dev/llvmlite compatibility break on `main`, not something introduced by
  this analysis. All A/B benchmarking above was therefore done against the
  installed 0.66.0 release's own source tree (verifying config flags and
  profiling the same code paths the gist measured), which is a valid stand-in
  since the compiler architecture under test (event system, pass manager
  setup, `OPT`/`SLP_VECTORIZE` defaults) is unchanged between 0.66.0 and
  `main`.
