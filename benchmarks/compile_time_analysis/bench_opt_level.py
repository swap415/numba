"""
Benchmark: effect of NUMBA_OPT (LLVM optimization level) on compile time
for the array-method/reduction call patterns that dominate compile time in
numba/tests/test_array_methods.py, test_array_manipulation.py and
test_array_reductions.py (see gist of instrumented test run,
0.67-times-numba-linux-64.txt).

Run three times with different NUMBA_OPT values, e.g.:

    NUMBA_OPT=3 python3 bench_opt_level.py   # current default
    NUMBA_OPT=1 python3 bench_opt_level.py
    NUMBA_OPT=0 python3 bench_opt_level.py

Measured against numba==0.66.0 (latest PyPI release at analysis time):

    OPT=3 (default): sum=23.00s clip=7.91s where=11.06s transpose=5.52s TOTAL=47.49s
    OPT=1:           sum=13.53s clip=6.96s where=9.36s  transpose=4.01s TOTAL=33.86s (-29%)
    OPT=0:           sum=9.86s  clip=6.51s where=10.10s transpose=3.63s TOTAL=30.09s (-37%)
"""
import time
import numpy as np
import numba
from numba import njit

print("NUMBA", numba.__version__, "OPT=", numba.config.OPT)


def array_sum_axis_dtype_kws(arr, axis, dtype):
    return np.sum(arr, axis=axis, dtype=dtype)


def np_clip(a, a_min, a_max):
    return np.clip(a, a_min, a_max)


def np_where_3(cond, x, y):
    return np.where(cond, x, y)


def numpy_transpose_axes(arr, axes):
    return np.transpose(arr, axes)


def run_sum_combo():
    cfunc = njit(array_sum_axis_dtype_kws)
    a = np.linspace(-10, 10, 120).reshape(4, 5, 6)
    signed_dtypes = [np.float64, np.float32, np.int64, np.int32]
    unsigned_dtypes = [np.uint32, np.uint64]
    out_dtypes = {
        np.dtype('float64'): [np.float64],
        np.dtype('float32'): [np.float64, np.float32],
        np.dtype('int64'): [np.float64, np.int64, np.float32],
        np.dtype('int32'): [np.float64, np.int64, np.float32, np.int32],
        np.dtype('uint32'): [np.float64, np.int64, np.float32],
        np.dtype('uint64'): [np.float64, np.uint64],
    }
    for dt in signed_dtypes + unsigned_dtypes:
        arr = a.astype(dt)
        for out_dtype in out_dtypes[arr.dtype]:
            for axis in (0, 1, 2):
                cfunc(arr, axis, out_dtype)


def run_clip_combo():
    cfunc = njit(np_clip)
    a = np.linspace(-10, 10, 40).reshape(5, 2, 4)
    a_min_arr = np.arange(-8, 0).astype(a.dtype).reshape(2, 4)
    a_max_arr = np.arange(0, 8).astype(a.dtype).reshape(2, 4)
    mins = [0, -5, a_min_arr, None]
    maxs = [0, 5, a_max_arr, None]
    for a_min in mins:
        for a_max in maxs:
            if a_min is None and a_max is None:
                continue
            try:
                cfunc(a, a_min, a_max)
            except Exception:
                pass


def run_where_combo():
    cfunc = njit(np_where_3)
    dtypes = [np.float64, np.float32, np.int64, np.int32, np.bool_]
    cond = np.array([True, False, True, False])
    for dt in dtypes:
        x = np.arange(4).astype(dt)
        y = (np.arange(4) * 2).astype(dt)
        cfunc(cond, x, y)
        cfunc(cond, x, dt(1))
        cfunc(cond, dt(1), y)


def run_transpose_combo():
    cfunc = njit(numpy_transpose_axes)
    a = np.arange(24).reshape(2, 3, 4)
    import itertools
    for axes in itertools.permutations(range(3)):
        cfunc(a, axes)


if __name__ == "__main__":
    t0 = time.perf_counter()
    run_sum_combo()
    t1 = time.perf_counter()
    run_clip_combo()
    t2 = time.perf_counter()
    run_where_combo()
    t3 = time.perf_counter()
    run_transpose_combo()
    t4 = time.perf_counter()

    print(f"sum_combo:       {t1 - t0:.3f}s")
    print(f"clip_combo:      {t2 - t1:.3f}s")
    print(f"where_combo:     {t3 - t2:.3f}s")
    print(f"transpose_combo: {t4 - t3:.3f}s")
    print(f"TOTAL:           {t4 - t0:.3f}s")
