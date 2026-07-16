# Numba test-suite compile-time optimization analysis

Source data: [compile-time gist](https://gist.github.com/swap415/6aee3d3742fcdab493c1c6fe2c2058c4)
(Linux-64 CI run) and instrumentation from draft PR
[swap357/numba#110](https://github.com/swap357/numba/pull/110) (`CompileTimeTracker`,
which hooks the `numba:compile` / `numba:run_pass` / `numba:llvm_lock` events to
split each test's wall time into `compile` (further split into `passes` and
`codegen`) and `run`).

## Top compile-time offenders (from the gist)

| Test | Compile time | Compile calls | Run time |
|---|---|---|---|
| `test_clip_array_min_max` | 101.4s | 170 | 433ms |
| `test_fill_diagonal_basic` | 63.5s | 60 | 385ms |
| `test_sum_axis_dtype_kws` | 51.2s | 46 | 451ms |
| `test_array_transpose_axes` | 26.0s | 58 | 257ms |
| `test_sum_exceptions` | 23.9s | 2 | 132ms |
| `test_take` | 22.9s | 29 | 165ms |
| `test_sum_axis_kws1` | 20.4s | 14 | 537ms |
| `test_argmax_axis_1d_2d_4d` | 19.0s | 14 | 189ms |

Compile time dominates total duration for nearly every test in this list —
run time is a rounding error by comparison. That means the highest-leverage
fixes are ones that reduce **compile call count**, not runtime performance.

## Candidate #1 — fix redundant recompilation in `_lower_clip_result_test_util` (BENCHMARKED)

**File:** `numba/tests/test_array_methods.py`

`_lower_clip_result_test_util` is called up to ~90 times inside
`test_clip_array_min_max`'s `pyfunc x a_min x a_max` loop. Each call defined a
**brand new closure** and JIT-compiled it from scratch:

```python
def _lower_clip_result_test_util(self, func, a, a_min, a_max):
    def lower_clip_result(a):
        return np.expm1(func(a, a_min, a_max))
    np.testing.assert_almost_equal(
        lower_clip_result(a),
        jit(nopython=True)(lower_clip_result)(a))   # fresh Dispatcher every call
```

Because `jit()` is called on a newly-defined function object each time, Numba
gets a brand-new `Dispatcher` with an empty compilation cache on every
invocation — none of the ~90 calls can reuse a previous compile, even though
many of them share the same argument types. This single helper accounts for
roughly half of `test_clip_array_min_max`'s 170 compile calls.

**Fix (5 lines):** hoist the wrapper to module scope, compile it once, and pass
the varying values as arguments instead of closing over them:

```python
@jit(nopython=True)
def _lower_clip_result_jit(a, func, a_min, a_max):
    return np.expm1(func(a, a_min, a_max))

def _lower_clip_result_test_util(self, func, a, a_min, a_max):
    np.testing.assert_almost_equal(
        np.expm1(func(a, a_min, a_max)),
        _lower_clip_result_jit(a, func, a_min, a_max))
```

Now the same `Dispatcher` is reused across every call; Numba only recompiles
when it sees a genuinely new type signature (distinct `func` dispatcher x
`a_min`/`a_max` type), instead of on every call.

This exact code (byte-for-byte) is also present in the **latest released
Numba, 0.66.0**, so it isn't dev-branch-specific debt — it's a real, shippable
fix.

**Measured impact** — ran `TestArrayMethods.test_clip_array_min_max` under
released `numba==0.66.0` / `llvmlite==0.48.0` / `numpy==2.4.6` in a clean venv,
patching only `numba/tests/test_array_methods.py`:

| Variant | Wall time (2 runs) | Mean |
|---|---|---|
| Unpatched (baseline) | 68.44s, 63.71s | 66.1s |
| Patched | 36.35s, 37.52s | 36.9s |

**~44% reduction / 1.79x speedup**, real numbers, same test, same machine,
same released Numba version. This change is pure test-code refactoring with
no behavior change — the test still asserts the exact same thing.

**Applied to this branch:** `numba/tests/test_array_methods.py` (see diff).

## Candidate #2 — reduce LLVM optimization level during test runs (BENCHMARKED)

Numba exposes `NUMBA_OPT` (default `3`, i.e. `-O3`) to control the LLVM
function-pass-manager optimization level used during codegen. Test suites
care about *correctness*, not the speed of the JIT'd code, so running the
suite at a lower optimization level trades away runtime speed of the
generated code (irrelevant for a correctness test) for less time spent in
LLVM's optimizer (`codegen` time, per the PR's own breakdown).

**Fix (effectively 1 line):** set `NUMBA_OPT=0` (or `1`) in the CI workflow
env for test jobs.

**Measured impact** (same venv/version, `test_clip_array_min_max`):

| Variant | Wall time |
|---|---|
| Unpatched, default `NUMBA_OPT` | 66.1s (mean of 2) |
| Unpatched, `NUMBA_OPT=0` | 53.72s (~19% faster) |
| **Patched (Candidate #1) + `NUMBA_OPT=0` combined** | **28.83s (~56% faster than baseline, 2.3x speedup)** |

This lever is global — every compile in every test benefits, not just this
one — so its aggregate effect across the full suite is likely much larger
than the single-test number above suggests. It's also low-risk: `NUMBA_OPT`
only affects LLVM optimization passes, not program semantics, and is an
already-supported, documented Numba env var (no code changes, no new
dependencies). Recommend validating on a full CI run before adopting
suite-wide, since some numerical edge-case tests could conceivably be
sensitive to codegen differences (unlikely, but untested here beyond one
module).

## Candidate #3 — trim exhaustive permutation matrices where coverage is redundant (ESTIMATED, not yet applied)

`test_array_transpose_axes` (26.0s compile, 58 calls) iterates over **every**
`itertools.permutations()` of each test array's axes (up to 24 permutations
for a 4-D array), for 4 pyfunc variants, via `numba/tests/test_array_manipulation.py:307-352`.
Each distinct axes-tuple length/shape is a distinct type signature Numba must
compile. Sampling a representative subset of permutations (e.g. first axis,
last axis, one reversal, a couple of interior swaps — say 6-8 permutations
per array instead of up to 24) rather than the full permutation set would cut
this test's compile count roughly in half while still exercising every axis
position at least once. This is a coverage/speed trade-off (unlike #1 and #2,
which are free), so it needs a maintainer call rather than being landed
unilaterally — flagging as a lead rather than applying it.

**Estimated impact:** compile-call reduction of ~40-50% for this test based on
permutation-count math; not empirically benchmarked (would require rewriting
the axis-sampling logic and re-measuring, left as follow-up).

## Not low-hanging fruit

- `test_sum_exceptions` (23.9s / 2 calls, ~12s per compile) and
  `test_readonly_after_ravel`/`test_readonly_after_flatten` (8.6-8.7s each,
  1 call) spend seconds compiling a *single* signature. This isn't redundant
  recompilation (the anti-pattern behind #1) — it's one expensive compile,
  which points at Numba's own compiler pipeline (parfors/exception lowering
  cost) rather than a test-code fix. Out of scope for a small diff.
- `test_fill_diagonal_basic`, `test_take`, `test_sum_axis_dtype_kws`, and the
  `test_broadcast_arrays_*` family all compile a dispatcher once and reuse it
  correctly — their high compile-call counts reflect genuinely distinct
  dtype/shape/kwarg-presence combinations being tested (real coverage, not
  waste). No 1-5 line fix identified for these; shrinking their matrices is
  the same coverage trade-off as Candidate #3.

## Benchmark environment note

Numba's own dev branch requires `llvmlite>=0.49.0dev0,<0.50`, which has no
public PyPI wheel and would require a from-source LLVM build to test in this
environment — not attempted given time constraints. All empirical
measurements above therefore used the **latest PyPI release, `numba==0.66.0`**
(`llvmlite==0.48.0`, `numpy==2.4.6`), run in an isolated venv, which
satisfies the "benchmark against latest released Numba" requirement directly
and is a stronger baseline for upstreaming than testing against an unreleased
dev build would have been. Candidate #1 has additionally been applied to this
branch's `numba/tests/test_array_methods.py` unmodified from what was
benchmarked.

## Summary (prioritized)

1. **Candidate #1 (applied to this branch):** hoist `_lower_clip_result_jit`
   out of the per-call closure — 44% faster, verified, zero coverage
   trade-off, upstreamable to numba/numba as-is.
2. **Candidate #2 (recommend for CI, not yet applied here since it's an
   environment/workflow change, not application code):** `NUMBA_OPT=0` for
   test jobs — 19% faster standalone, 56% combined with #1, global effect
   across the whole suite, needs a full-suite CI validation pass before
   adoption.
3. **Candidate #3 (lead only):** sample rather than exhaustively enumerate
   axis permutations in `test_array_transpose_axes` — estimated 40-50%
   compile-call reduction for that test, trades some coverage, needs
   maintainer sign-off.
