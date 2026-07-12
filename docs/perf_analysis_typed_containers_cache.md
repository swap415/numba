# Compile-time optimization: cache typed-container trampolines

## Source data

- Test-suite compile/runtime instrumentation: gist
  `swap415/6aee3d3742fcdab493c1c6fe2c2058c4` (Linux x86_64 run).
- Instrumentation approach reviewed: `swap357/numba#110` ("Add compile time
  monitoring utilities and update test workflows") — a draft PR adding
  `CompileTimeTracker`, CI artifact upload, and cross-platform analysis/chart
  scripts. It only adds *measurement* infrastructure; it does not change
  compiler behavior.

## Observation

Across the instrumented suite, JIT **compile time** dominates **run time** by
2-3 orders of magnitude for any test that touches Numba's typed containers,
jitclass, or ufunc-at machinery, e.g.:

| Test | Total | Compile | Compiles | Run |
|---|---|---|---|---|
| `test_ex_typed_dict_from_cpython` | 5.106s | 5.101s | 4 | 4.98ms |
| `test_ex_typed_dict_njit` | 4.978s | 4.975s | 3 | 2.24ms |
| `test_ex_initial_value_dict_compile_time_consts` | 5.544s | 5.542s | 5 | 2.37ms |
| `test_ex_jitclass_type_hints` | 1.336s | 1.330s | 6 | 5.45ms |
| `test_ex_inferred_dict_njit` | 1.194s | 1.191s | 7 | 2.96ms |
| `test_ex_typed_set_from_cpython` | 887.1ms | 884.8ms | 4 | 2.27ms |
| `test_numpy_ufunc_at_basic` | 3.615s | 3.607s | 6 | 7.57ms |

We traced these against the source (both via an independent read-through and
via `cProfile` against a locally built dev tree) and confirmed: the "N
compiles" figure is **N distinct signatures being compiled once each**, not
redundant recompilation of the same signature within a process — Numba's
in-process dispatcher cache already avoids that. The cost is a **one-time,
per-process compilation tax** for `Dict`/`List`/`Set`/`jitclass` construction,
paid independently by every fresh Python process: every `pytest-xdist`
worker, every CI job, every local `python -m numba.runtests` invocation, every
notebook restart.

`numba/typed/typeddict.py`, `typedlist.py`, and `typedset.py` implement the
Python-facing `Dict`/`List`/`Set` wrapper classes as a set of small
module-level `@njit` "trampoline" functions (`_make_dict`, `_setitem`,
`_getitem`, `_additem`, `_append`, ...). None of them set `cache=True`, so
each one's LLVM compilation is discarded when the process exits, and the next
process pays full price again for the exact same `(key_type, value_type)`
specialization.

## Candidate: `cache=True` on typed-container trampolines

**Change:** `@njit` → `@njit(cache=True)` on the ~11/27/7 trampoline
functions in `numba/typed/typeddict.py`, `typedlist.py`, `typedset.py`
respectively — a mechanical decorator-argument change, no logic touched.

**Mechanism:** `cache=True` persists the compiled artifact under each
module's `__pycache__` (`.nbi`/`.nbc` files), keyed by function + argument
types + Numba/source version. The first process to hit a given
`(container, key_type, value_type)` combination still pays full compile
cost; every subsequent process loads the artifact from disk instead of
re-running type inference + LLVM codegen. Numba already treats
uncacheable specializations gracefully (falls back to normal in-memory
compilation with a `NumbaWarning`), so this is not an all-or-nothing bet —
see caveat below.

### Benchmark: fresh-process compile time, before vs after

Measured by running an isolated construction snippet (`Dict.empty()` +
2 inserts / `List()` + 5 appends / `Set()` + 5 adds / `jitclass` construct +
1 method call) in a **brand-new Python process** each time, timing only the
post-import construction — i.e. simulating what a fresh CI/xdist worker pays.

Baseline = latest PyPI release, **numba 0.66.0**. "Patched dev" = this
repo's `main` tree (commit `63b75658`) with the `cache=True` change, built
locally and installed editable in an isolated venv.

| Container | Released 0.66.0 (every worker) | Patched dev, worker 1 (cold) | Patched dev, worker 2+ (warm cache) | Speedup (worker 2+) |
|---|---|---|---|---|
| `Dict` | 2839–3225 ms | 3230 ms | **136–143 ms** | **~22x** |
| `List` | 292–299 ms | 298 ms | **115–118 ms** | **~2.5x** |
| `Set` | 621–632 ms | 646 ms | **134–144 ms** | **~4.5x** |
| `jitclass` (control, untouched) | 548–568 ms | 545 ms | 565–567 ms | ~1x (expected — not patched) |

The `jitclass` row is an internal control: it was *not* patched (jitclass
constructors are wrapped once per class decoration, not per instantiation —
we found no redundant recompilation there), and correctly shows no change,
which is evidence the measured wins are attributable to the cache and not to
noise/environment drift.

**Which tests benefit most:** anything exercising `numba.typed.Dict`/`List`/
`Set` from a cold process — `test_ex_typed_dict_*`, `test_ex_inferred_dict_*`,
`test_ex_typed_set_from_cpython`, `test_ex_inferred_list*`,
`test_ex_nested_list`, and (via `numba/tests/test_dictobject.py`,
`test_setobject.py`, `test_typedlist.py`) any test module that constructs a
typed container for the first time in its worker process. In the observed
data these single tests each account for 1-5+ seconds of pure compile-time
tax; multiplied across a CI test-matrix with many parallel workers this is
one of the largest single line-count-to-impact ratios in the suite.

**Caveats (read before merging):**
1. This is a **cross-process** win only. A single long-lived process that
   already constructed one `Dict[unicode_type, int64]` gets the second one
   for free either way (in-memory dispatcher cache), so this will not show
   up as a within-process speedup — only across fresh workers/CI runs/local
   re-invocations.
2. Not every specialization is cacheable: value/key types that embed
   "dynamic globals" (e.g. ctypes pointers, ffi handles) cannot be cached;
   Numba falls back safely but emits a `NumbaWarning: Cannot cache compiled
   function ... it uses dynamic globals`. We observed this exactly once
   across the full `test_dictobject.py` run (a jitclass-valued dict test).
   It's a warning, not a failure, but it does add test-log noise worth
   knowing about.
3. First-run cost is unchanged (someone still pays the ~3s dict / ~600ms
   set / ~300ms list tax once) — this only pays off from the second process
   onward, so the win scales with number of workers/re-runs, not with a
   single cold run.

**Validation performed:**
- Full `numba.tests.test_dictobject` (131 tests), `test_setobject` (8 tests),
  `test_typedlist` (90 tests) run against the patched tree — all pass
  (`OK`, no new failures) with only the expected caching-related warnings
  noted above.
- Correctness of `Dict`/`List` mutation, indexing, membership, deletion
  spot-checked manually against a cold-cache process.

## Other candidates investigated and not pursued as low-hanging fruit

- **`dufunc.py: ol_at`** — the `.at()` typing overload re-declares two
  `@intrinsic` closures on every type-inference `propagate()` pass instead
  of once. Structurally real (verified via `typeinfer.py`'s constraint
  propagation loop), but the dominant cost for `TestDUFuncAt` tests is the
  actual elementwise-kernel LLVM compile, not the (cheap, dict-append-level)
  intrinsic re-registration — we could not confirm this explains a
  meaningful fraction of the 1-3.6s compile times observed, so we're not
  recommending it as a "significant impact" low-hanging-fruit item without
  further instrumentation.
- **LLVM optimization level** — `numba/core/codegen.py` already runs a cheap
  pass at `NUMBA_OPT`-controlled level with a separate, deliberate full-`-O3`
  final pass; this is existing tunable design, not an unexploited gap, and
  changing the default is a global behavior change, not a small/low-risk one.
- **jitclass constructor** — compiled once per class decoration, not per
  instance; no redundant work found.
- **Typed container "duplicate compilation"** — the high "compile count"
  metric in the gist reflects genuinely distinct signatures, not the same
  signature recompiled; ruled out as a bug.

## Reproduction

Benchmark scripts used for the above (isolated per-container timing across
fresh subprocesses, plus a combined smoke-test harness) are not included in
this repo; the methodology was: build this tree editable in a venv with
`llvmlite==0.48.0` (temporarily relaxing the `>=0.49.0dev0` floor in
`numba/__init__.py`/`setup.py` for local build purposes only — not part of
this change), install `numba==0.66.0` from PyPI in a second venv, and time
`Dict.empty()+setitem`, `List()+append`, `Set()+add`, and a trivial
`@jitclass` construct+call in fresh subprocesses before and after clearing
`__pycache__`.
