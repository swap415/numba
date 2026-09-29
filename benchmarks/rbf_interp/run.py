"""Benchmark RBF interpolant evaluation across numba / jax / torch / pythran.

Replicates the CPU comparison from scipy PR #23447. Two layers:
  minimal  -- synthetic coeffs, isolates the eval kernel (compiler quality)
  faithful -- real scipy.interpolate.RBFInterpolator fit, the PR's code path

Every backend's output is checked against the numpy reference before timing.
Speedup is reported against pythran, the PR's AOT baseline.

Usage:
  python run.py                 # both layers, default sweep
  python run.py --layer minimal --kernel gaussian
"""
import os
os.environ.setdefault("JAX_ENABLE_X64", "1")        # jax must match float64
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # numba+torch OpenMP on mac

import argparse
import statistics
import time

import numpy as np

import backends
import problems

ORDER = ["pythran", "numpy", "numba", "jax", "torch"]


def time_run(run, repeats, warmup):
    for _ in range(warmup):
        run()
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        run()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts), statistics.pstdev(ts)


def bench_size(make, P, Q, kernel, degree, names, repeats, warmup, device):
    p = make(P, Q, kernel=kernel, degree=degree)
    ref = backends.build_numpy(p)()              # numpy/cpu reference for correctness
    out = {}
    for name in names:
        try:
            run = backends.BACKENDS[name](p, device)
            first = np.asarray(run())
            ok = np.allclose(first, ref, atol=1e-6, rtol=1e-6)
            med, std = time_run(run, repeats, warmup)
            out[name] = dict(ms=med * 1e3, std=std * 1e3, ok=ok)
        except Exception as e:           # a missing/failing backend must not sink the sweep
            out[name] = dict(ms=float("nan"), std=float("nan"),
                             ok=False, err=f"{type(e).__name__}: {e}")
    return out


def fmt_table(rows, names, baseline="pythran"):
    head = f"{'P':>6} {'Q':>7} | " + " | ".join(f"{n:>16}" for n in names)
    lines = [head, "-" * len(head)]
    for P, Q, res in rows:
        base = res.get(baseline, {}).get("ms", float("nan"))
        cells = []
        for n in names:
            r = res[n]
            if "err" in r:
                cells.append(f"{'ERR':>16}")
            else:
                sp = base / r["ms"] if r["ms"] == r["ms"] else float("nan")
                flag = "" if r["ok"] else "!"
                cells.append(f"{r['ms']:8.2f}ms {sp:4.1f}x{flag:>1}")
        lines.append(f"{P:>6} {Q:>7} | " + " | ".join(cells))
    errs = {n: res[n]["err"] for P, Q, res in rows for n in names if "err" in res[n]}
    for n, e in errs.items():
        lines.append(f"  {n} error: {e}")
    return "\n".join(lines)


def run_layer(layer, sizes, kernel, degree, names, repeats, warmup, device):
    make = problems.make_minimal if layer == "minimal" else problems.make_faithful
    rows = []
    for P, Q in sizes:
        print(f"  {layer} P={P} Q={Q} ...", flush=True)
        rows.append((P, Q, bench_size(make, P, Q, kernel, degree,
                                      names, repeats, warmup, device)))
    return rows


def main():
    import numba
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", choices=["minimal", "faithful", "both"], default="both")
    ap.add_argument("--kernel", default="thin_plate_spline")
    ap.add_argument("--degree", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=15)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--backends", default=",".join(ORDER))
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu",
                    help="device for jax/torch; numpy/numba/pythran are always CPU")
    ap.add_argument("--big", action="store_true",
                    help="larger sizes where the GPU is not launch-overhead-bound")
    args = ap.parse_args()

    # jax reads its platform from this env at import time (build_jax, below).
    os.environ["JAX_PLATFORMS"] = args.device if args.device == "cpu" else "cuda"

    names = [n for n in ORDER if n in args.backends.split(",")]
    sizes = ([(1000, 10000), (2000, 20000), (4000, 40000)] if args.big
             else [(100, 1000), (300, 3000), (1000, 10000)])
    layers = ["minimal", "faithful"] if args.layer == "both" else [args.layer]

    gpu = (f"  gpu={torch.cuda.get_device_name(0)}"
           if args.device == "cuda" and torch.cuda.is_available() else "")
    env = (f"device={args.device}{gpu}  numba threads={numba.get_num_threads()}  "
           f"torch threads={torch.get_num_threads()}")
    header = (f"RBF eval benchmark (scipy PR #23447) -- kernel={args.kernel} degree={args.degree}\n"
              f"{env}\njax/torch on {args.device}; numpy/numba/pythran always CPU; "
              f"speedup vs pythran(CPU); '!' = mismatch vs numpy\n")
    print(header)

    out = [header]
    for layer in layers:
        rows = run_layer(layer, sizes, args.kernel, args.degree,
                         names, args.repeats, args.warmup, args.device)
        block = f"\n[{layer}]\n" + fmt_table(rows, names)
        print(block)
        out.append(block)
    return "\n".join(out)


if __name__ == "__main__":
    main()
