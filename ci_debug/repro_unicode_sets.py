"""Reproducer for TestUnicodeSets comparator timeout on win-arm64 CI.

Replicates TestSets._test_comparator with TestUnicodeSets data (seeded,
identical to the unittest) with per-combination timing. If any step hangs,
faulthandler dumps all thread stacks after DUMP_AFTER seconds and exits,
so the CI log shows where it is stuck (compile vs execute).
"""
import faulthandler
import itertools
import os
import sys
import time

# re-armed before every step: dump + exit if any single step stalls
DUMP_AFTER = 300
faulthandler.dump_traceback_later(DUMP_AFTER, exit=True)

import numba
import llvmlite
from numba import jit
from numba.tests.test_sets import (
    TestUnicodeSets,
    isdisjoint_usecase,
    issubset_usecase,
    issuperset_usecase,
)

print(f"numba {numba.__version__} llvmlite {llvmlite.__version__} "
      f"NUMBA_CPU_NAME={os.environ.get('NUMBA_CPU_NAME')}", flush=True)

inst = TestUnicodeSets('test_clear')
inst.setUp()

for pyfunc in (isdisjoint_usecase, issubset_usecase, issuperset_usecase):
    t_func = time.perf_counter()
    cfunc = jit(nopython=True)(pyfunc)
    a, b = map(set, [inst.sparse_array(10), inst.sparse_array(15)])
    args = [a & b, a - b, a | b, a ^ b]
    args = [tuple(x) for x in args]
    for i, (x, y) in enumerate(itertools.product(args, args)):
        faulthandler.dump_traceback_later(DUMP_AFTER, exit=True)
        t_it = time.perf_counter()
        expect = pyfunc(x, y)
        t_py = time.perf_counter()
        got = cfunc(x, y)
        t_jit = time.perf_counter()
        assert expect == got, (pyfunc.__name__, i, expect, got)
        print(f"{pyfunc.__name__}[{i:2d}] lens=({len(x)},{len(y)}) "
              f"py={t_py - t_it:6.2f}s jit(compile+run)={t_jit - t_py:7.2f}s",
              flush=True)
    print(f"{pyfunc.__name__} TOTAL {time.perf_counter() - t_func:.2f}s",
          flush=True)

faulthandler.cancel_dump_traceback_later()
print("DONE", flush=True)
