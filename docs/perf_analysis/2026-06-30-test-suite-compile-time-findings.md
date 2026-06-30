# Numba test-suite performance analysis (2026-06-30)

Source data: [compile/run-time metrics gist](https://gist.github.com/swap415/6aee3d3742fcdab493c1c6fe2c2058c4)
(`0.67-times-numba-linux-64.txt`, 5,400 parsed `Name: ... | Duration: ... | Compile: ...`
lines), produced by the instrumentation added in
[swap357/numba#110](https://github.com/swap357/numba/pull/110).

**Coverage caveat:** the gist's raw file is 1.79&nbsp;MB; GitHub's Gist API truncates
embedded content at 1&nbsp;MB, so only ~51% of the file (alphabetically,
`test_dummyarray` through `test_unpack_sequence`) could be retrieved from this
sandboxed environment (`gist.githubusercontent.com` is blocked by egress
policy here). The captured top offenders are 3-100x larger than every other
entry seen, so it's unlikely the missing back half (`test_v*`-`test_z*`,
~40 modules) hides an undiscovered outlier of similar magnitude, but this
is a real gap — re-run the analysis with full gist access to confirm.

## What PR #110 actually does

It is pure **CI instrumentation**, not a performance fix: `CompileTimeTracker`
(`numba/misc/compiletimeutils.py`) wraps each test in `numba/testing/main.py`
with `numba.core.event` listeners on `numba:compile`, `numba:run_pass`, and
`numba:llvm_lock`, prints a `Name | Duration | Compile (passes, codegen) | Run`
line per test under `NUMBA_TEST_COMPILE_TIMING=1`, and the wheel-builder
workflows upload these as CI artifacts. `ci_debug/analyze_compile_times.py`
ranks tests by compile time across platforms. This is the tool that produced
the gist's data; it makes no code-path changes itself.

## Top compile-time offenders (from the captured data)

| Test | Compile | Calls | Compile/call |
|---|---|---|---|
| `test_np_functions.test_isin_3a` | 319.2s | 46 | 6.9s |
| `test_np_functions.test_isin_4` | 317.0s | 46 | 6.9s |
| `test_np_functions.test_isin_3b` | 312.3s | 40 | 7.8s |
| `test_np_functions.test_isin_2` | 310.8s | 40 | 7.8s |
| `test_array_methods.test_clip_array_min_max` | 101.4s | 170 | 0.6s |
| `test_linalg.test_outer` | 94.2s | 56 | 1.7s |
| `test_fancy_indexing.TestFancyIndexing.test_setitem` | 63.5s | 91 | 0.7s |
| `test_array_manipulation.test_fill_diagonal_basic` | 63.5s | 60 | 1.1s |
| `test_fancy_indexing.TestFancyIndexingMultiDim.test_setitem` | 62.5s | 92 | 0.7s |
| `test_sort.TestTimsortArrays.test_merge_lo_hi` | 61.4s | 128 | 0.5s |

The four `test_isin_*` tests alone account for **~1,260s of compile time**,
more than the next 15 entries combined.

## Findings, prioritized

All three are **test-suite-only** changes (zero risk to library/runtime code),
verified by reproducing the relevant logic against **Numba 0.65.1 (latest
PyPI release) + llvmlite 0.47.0**, since the dev branch here requires an
unreleased llvmlite (≥0.48) that isn't installable in this sandbox.

### 1. `test_isin_2` is missing `@skip_if_reduced_testing` (1-line fix)

**File:** `numba/tests/test_np_functions.py:6953`

`test_isin_2`, `test_isin_3a`, `test_isin_3b`, and `test_isin_4` all iterate
the same `_isin_arrays()` generator (~40-46 distinct dtype/container
combinations — `int8`×`int16`, `uint8`×`uint16`, typed `List`, reflected
list, 0-d array, etc.). `test_isin_3a/3b/4` are decorated
`@skip_if_reduced_testing` (skipped when `_NUMBA_REDUCED_TESTING=1`, which
this repo's own `analyze_compile_times.py` notes is how Windows CI keeps
heavy cases off that platform). `test_isin_2` has no such decorator, so it
unconditionally pays the full ~40-signature compile sweep even in reduced
mode — the same sweep its siblings were explicitly exempted from.

```diff
+    @skip_if_reduced_testing
     def test_isin_2(self):
         np_pyfunc = np_isin_2
         np_nbfunc = njit(np_pyfunc)
```

**Verified cost of the sweep:** reproducing `np.isin`'s underlying
`_in1d_impl` (copied from `numba/np/arraymath.py`) against released Numba
and compiling it across 10 representative dtype pairs took **48.3s total,
4.8s/compile average** — consistent with the gist's 6.9-7.8s/call. Isolating
the two internal branches shows why: the small-array linear-scan branch
compiles in ~0.5s, but the generic sort-based branch (`argsort`, `unique`,
`concatenate`, `argsort(kind='mergesort')`) compiles in **~4.9s — 10x
more** — and because the branch choice is a runtime length check, Numba
must type+lower both branches on every signature regardless of which one
actually executes.

**Impact:** removes test_isin_2's ~310s from every reduced-testing CI run
(currently this is Windows). No change to full Linux/macOS coverage.

### 2. `check_setitem_indices` recompiles a closure on every call (TestFancyIndexingMultiDim)

**File:** `numba/tests/test_fancy_indexing.py:479-482`

```python
def check_setitem_indices(self, arr_shape, index):
    @njit
    def set_item(array, idx, item):
        array[idx] = item
    ...
```

`test_setitem` (line 510) calls this once per index pattern (~92 times).
`set_item` closes over nothing — `idx` is already a parameter — so there is
no reason to redefine and recompile it on every call; doing so means a new
`Dispatcher` object is created each time and Numba's signature cache (keyed
on the function object) can never hit, even when the same `(array dtype,
index-pattern type)` signature recurs.

```diff
-    def check_setitem_indices(self, arr_shape, index):
-        @njit
-        def set_item(array, idx, item):
-            array[idx] = item
+    @staticmethod
+    @njit
+    def _set_item(array, idx, item):
+        array[idx] = item
+
+    def check_setitem_indices(self, arr_shape, index):
+        set_item = self._set_item
```

**Verified cost:** reproducing the same `array[idx] = item` njit closure,
called 20x with an identical signature: redefining it fresh each call costs
**537ms/call**; hoisting it to a module-level dispatcher and reusing it
costs **20ms/call after the first compile — a 27x speedup**. Even assuming
modest signature reuse across the ~92 index patterns actually exercised,
this is the highest-leverage fix of the three by speedup ratio.

**Impact:** the bulk of `TestFancyIndexingMultiDim.test_setitem`'s 62.5s
compile time; check whether `TestFancyIndexing.check_setitem_indices`
(line 138, which already reuses one `cfunc` — the correct pattern) should
be the template instead.

### 3. `_lower_clip_result_test_util` recompiles a closure on every call

**File:** `numba/tests/test_array_methods.py:1722-1730`

```python
def _lower_clip_result_test_util(self, func, a, a_min, a_max):
    def lower_clip_result(a):
        return np.expm1(func(a, a_min, a_max))
    np.testing.assert_almost_equal(
        lower_clip_result(a),
        jit(nopython=True)(lower_clip_result)(a))
```

`test_clip_array_min_max` calls this inside a double loop over
`mins x maxs` (up to 15 combinations) for each of 6 `pyfunc` variants — up
to 90 calls, each defining and JIT-compiling a brand-new closure, even
though only ~3 distinct types are exercised per side (`int`, `array`,
`NoneType`). Converting `a_min`/`a_max` to real parameters lets a single
cached dispatcher per `func` serve all 15 combinations:

```diff
 def _lower_clip_result_test_util(self, func, a, a_min, a_max):
-    def lower_clip_result(a):
+    def lower_clip_result(a, a_min, a_max):
         return np.expm1(func(a, a_min, a_max))
+    cfunc = self._clip_cache.setdefault(
+        func, jit(nopython=True)(lower_clip_result))
     np.testing.assert_almost_equal(
-        lower_clip_result(a),
-        jit(nopython=True)(lower_clip_result)(a))
+        lower_clip_result(a, a_min, a_max),
+        cfunc(a, a_min, a_max))
```

(`self._clip_cache = {}` would also need adding in `setUp`.)

**Verified cost:** reproducing the same `np.expm1(a.clip(a_min, a_max))`
closure pattern, called 9x: fresh-compile-per-call costs **219ms/call**;
a cached dispatcher costs **19ms/call — 11x speedup**.

**Impact:** the majority of `test_clip_array_min_max`'s 101.4s compile time
(170 reported compile calls — consistent with ~90 fresh closures plus the
outer per-`pyfunc` `cfunc` compiles).

## Not pursued

- `test_outer`, `test_fill_diagonal_basic`, `test_kron`, `test_merge_lo_hi`:
  inspected the same way; these already reuse a single `cfunc`/dispatcher
  across their loops. Their cost is genuine dtype/shape coverage (`test_outer`,
  `test_fill_diagonal_basic`) or Numba's internal Timsort having many small
  compiled subroutines deliberately recompiled per test for isolation
  (`test_merge_lo_hi`) — not a bug, and riskier to touch.
- Percentile/quantile family (`test_*percentile*`, `test_*quantile*`,
  ~65-69s each): the type/shape diversity driving their compile count
  (3-d arrays, flat lists, tuples, scalar vs. array `q`) looks like
  deliberate coverage, not redundant recompilation — no clear 1-5 line fix
  found.

## Summary

| # | Fix | Lines | Verified speedup | Tests benefiting |
|---|---|---|---|---|
| 1 | Add `@skip_if_reduced_testing` to `test_isin_2` | 1 | removes ~310s from reduced-testing runs | `test_isin_2` (Windows CI only) |
| 2 | Hoist `set_item` out of `check_setitem_indices` | ~6 | 27x per repeated signature | `TestFancyIndexingMultiDim.test_setitem` |
| 3 | Cache `lower_clip_result` dispatcher per `func` | ~6 | 11x per repeated signature | `test_clip_array_min_max` |

All three are test-only, mechanically simple, and independently verified
against the latest released Numba (0.65.1). None touch `numba/np/arraymath.py`
or other library code, so there's no runtime-behavior risk to ship.
