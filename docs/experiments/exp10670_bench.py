"""Indicative wall-clock benchmark for the #10670 unroll-metadata experiment.

Hardware perf counters are unavailable in the sandbox this experiment was
developed in, so this is a secondary, NOISY signal only - the primary metric
is the static IR branch analysis in exp10670_worker.py.  Run each mode in a
fresh process:

    python exp10670_bench.py baseline
    python exp10670_bench.py tagged      # llvm.loop.unroll.runtime.disable
                                         # on every loop latch
"""
import sys
import time

import numpy as np

import exp10670_worker as W


def main():
    mode = sys.argv[1]
    add_for, add_while_uint, add_for_const = W.make_funcs()
    if mode == 'tagged':
        with W.tagging('runtime', 'all'):
            add_for.compile(W.SIG2)
            add_while_uint.compile(W.SIG2)
            add_for_const.compile(W.SIG1)
    elif mode == 'baseline':
        add_for.compile(W.SIG2)
        add_while_uint.compile(W.SIG2)
        add_for_const.compile(W.SIG1)
    else:
        raise SystemExit(f'unknown mode {mode!r}')

    arr = np.zeros(20_000_000)
    x = 5  # issue-shaped workload: many outer iterations, tiny inner loop
    for name, f, args in [('add_for', add_for, (arr, x)),
                          ('add_while_uint', add_while_uint, (arr, x)),
                          ('add_for_const', add_for_const, (arr,))]:
        f(*args)  # warm up / JIT
        times = []
        for _ in range(7):
            t0 = time.perf_counter()
            f(*args)
            times.append(time.perf_counter() - t0)
        print(f'{mode:9} {name:16} best={min(times) * 1e3:8.2f} ms  '
              f'median={sorted(times)[len(times) // 2] * 1e3:8.2f} ms')


if __name__ == '__main__':
    main()
