"""
Benchmark script for the four compile-time optimizations:
  1. Cached inspect.signature() in _OverloadFunctionTemplate (templates.py)
  2. Cached _get_jit_decorator() result per template class (templates.py)
  3. Cached register_jitable(gen_sum_axis_impl()) via _make_sum_axis_compiled (arraymath.py)
  4. Flat iteration instead of np.ndindex in np_clip_ss/sn/ns (arrayobj.py)

Run with:
  python benchmarks/bench_compile_hotspots.py

Each section times cold-compile (first JIT call) for many type combinations,
which is the regime exercised by the tests flagged in the gist:
  - test_clip_array_min_max  : 101.8s / 170 calls (BEFORE)
  - test_fill_diagonal_basic : 63.9s  / 60 calls  (BEFORE)
  - test_sum_axis_dtype_kws  : 51.7s  / 46 calls  (BEFORE)
"""

import time
import numpy as np
import numba


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def time_compile(func, args, label):
    """Time a single cold-compile + first call."""
    t0 = time.perf_counter()
    func(*args)
    t1 = time.perf_counter()
    print(f"  {label}: {(t1 - t0)*1000:.1f} ms")
    return t1 - t0


# ---------------------------------------------------------------------------
# 1 & 4: np.clip with many type combinations
#         tests _cached_signature, _cached_jitter, and flat-iteration speedup
# ---------------------------------------------------------------------------

print("=== np.clip compile benchmark (scalar-scalar branch) ===")

dtypes = [np.float32, np.float64, np.int8, np.int16, np.int32, np.int64,
          np.uint8, np.uint16, np.uint32, np.uint64]

clip_funcs = []
clip_times = []
for dt in dtypes:
    @numba.njit
    def _clip(a, lo, hi):
        return np.clip(a, lo, hi)

    arr = np.array([1, 5, 3, 7], dtype=dt)
    lo = dt(2)
    hi = dt(6)
    t = time_compile(_clip, (arr, lo, hi), f"clip({dt.__name__})")
    clip_funcs.append(_clip)
    clip_times.append(t)

total_clip = sum(clip_times)
print(f"  Total clip compile: {total_clip*1000:.1f} ms over {len(dtypes)} dtypes")


# ---------------------------------------------------------------------------
# 3: np.sum with axis+dtype — tests _make_sum_axis_compiled cache
# ---------------------------------------------------------------------------

print("\n=== np.sum(axis, dtype) compile benchmark ===")

sum_times = []
for dt_in in [np.float32, np.float64, np.int32, np.int64]:
    for dt_out in [np.float32, np.float64]:
        for ndim, arr_fn in [(1, lambda dt: np.ones(10, dtype=dt)),
                              (2, lambda dt: np.ones((3, 4), dtype=dt))]:
            @numba.njit
            def _sum(a):
                return np.sum(a, axis=0, dtype=dt_out)

            arr = arr_fn(dt_in)
            label = f"sum({dt_in.__name__}[{ndim}d], axis=0, dtype={dt_out.__name__})"
            t = time_compile(_sum, (arr,), label)
            sum_times.append(t)

total_sum = sum(sum_times)
print(f"  Total sum compile: {total_sum*1000:.1f} ms over {len(sum_times)} combos")


# ---------------------------------------------------------------------------
# fill_diagonal: verifies _cached_signature helps sub-dispatch
# ---------------------------------------------------------------------------

print("\n=== np.fill_diagonal compile benchmark ===")

fill_times = []
for dt in [np.float32, np.float64, np.int32, np.int64]:
    @numba.njit
    def _fill(a, v):
        np.fill_diagonal(a, v)
        return a

    mat = np.zeros((4, 4), dtype=dt)
    val = dt(7)
    t = time_compile(_fill, (mat, val), f"fill_diagonal({dt.__name__})")
    fill_times.append(t)

total_fill = sum(fill_times)
print(f"  Total fill_diagonal compile: {total_fill*1000:.1f} ms over {len(fill_times)} dtypes")


# ---------------------------------------------------------------------------
# Runtime benchmark: flat vs ndindex (optimization 4 runtime benefit)
# ---------------------------------------------------------------------------

print("\n=== np.clip runtime benchmark (scalar-scalar, large array) ===")

@numba.njit
def clip_f64(a, lo, hi):
    return np.clip(a, lo, hi)

# warm up
big = np.random.rand(10_000_000).astype(np.float64)
clip_f64(big, 0.2, 0.8)  # already compiled above, reuse

N = 5
t0 = time.perf_counter()
for _ in range(N):
    clip_f64(big, 0.2, 0.8)
t1 = time.perf_counter()
print(f"  clip float64 10M elements: {(t1-t0)/N*1000:.1f} ms/call")

ref = np.clip(big, 0.2, 0.8)
assert np.allclose(clip_f64(big, 0.2, 0.8), ref), "clip result mismatch!"
print("  Result correctness: OK")

print("\nDone.")
