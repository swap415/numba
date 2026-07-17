"""
Benchmark: effect of `cache=True` on repeated compilation of the same
signature sweep, simulating a rerun of the same test module (local dev
iteration, or a CI leg with a persisted NUMBA_CACHE_DIR).

None of numba/tests/test_array_methods.py, test_array_manipulation.py or
test_array_reductions.py currently pass cache=True to any of their
`jit(nopython=True)(pyfunc)` call sites (98 call sites combined).

Run twice in a row from the same directory:

    python3 bench_cache.py   # cold: writes cache
    python3 bench_cache.py   # warm: reads cache

Measured against numba==0.66.0:

    cold (writes cache): 6.742s
    warm (reads cache):  0.225s   (~30x faster)
"""
import time
import numpy as np
from numba import njit


def array_sum_axis_dtype_kws(arr, axis, dtype):
    return np.sum(arr, axis=axis, dtype=dtype)


cfunc = njit(cache=True)(array_sum_axis_dtype_kws)

if __name__ == "__main__":
    a = np.linspace(-10, 10, 120).reshape(4, 5, 6)
    dtypes = [np.float64, np.float32, np.int64, np.int32]

    t0 = time.perf_counter()
    for dt in dtypes:
        arr = a.astype(dt)
        for axis in (0, 1, 2):
            cfunc(arr, axis, np.float64)
    t1 = time.perf_counter()
    print(f"run time: {t1 - t0:.3f}s")
