# Results — RBF eval, numba vs jax vs torch vs pythran (CPU)

**Machine:** Apple M4 Pro (14 numba threads, 10 torch threads), macOS 26.2, Python 3.12.
**Stack:** numba 0.65.1 / jax 0.10.2 / torch 2.12.1 / pythran 0.18.1 / scipy 1.18.0 / numpy 2.4.6.
**Method:** median of 15 runs after 3 warmups; every backend's output asserted equal to numpy before timing. Speedup vs pythran (the PR's AOT baseline). No CUDA — CPU only.

## thin_plate_spline, degree 1

`P` data points, `Q` eval points; the eval matrix is `(Q, P+R)`.

### minimal layer (synthetic coeffs — isolates the eval kernel)

| P | Q | pythran | numpy | numba | jax | torch |
|---:|---:|---:|---:|---:|---:|---:|
| 100 | 1000 | 0.62 ms (1.0×) | 1.58 ms (0.4×) | 1.17 ms (0.5×) | 0.21 ms (3.0×) | 0.32 ms (1.9×) |
| 300 | 3000 | 5.44 ms (1.0×) | 12.59 ms (0.4×) | 7.76 ms (0.7×) | 0.96 ms (5.7×) | 0.79 ms (6.9×) |
| 1000 | 10000 | 56.5 ms (1.0×) | 143 ms (0.4×) | 71.0 ms (0.8×) | 9.71 ms (5.8×) | **6.05 ms (9.3×)** |

### faithful layer (real `RBFInterpolator` fit — the PR's code path)

| P | Q | pythran | numpy | numba | jax | torch |
|---:|---:|---:|---:|---:|---:|---:|
| 100 | 1000 | 0.62 ms (1.0×) | 1.49 ms (0.4×) | 1.15 ms (0.5×) | 0.20 ms (3.1×) | 0.32 ms (1.9×) |
| 300 | 3000 | 5.37 ms (1.0×) | 12.90 ms (0.4×) | 7.71 ms (0.7×) | 0.96 ms (5.6×) | 0.80 ms (6.7×) |
| 1000 | 10000 | 59.0 ms (1.0×) | 145 ms (0.4×) | 75.8 ms (0.8×) | 9.75 ms (6.1×) | **6.14 ms (9.6×)** |

The two layers agree — evaluation cost is the same; only the coeff *values* differ.

## gaussian, degree 1 (minimal) — pattern is kernel-independent

| P | Q | pythran | numpy | numba | jax | torch |
|---:|---:|---:|---:|---:|---:|---:|
| 1000 | 10000 | 28.2 ms (1.0×) | 134 ms (0.2×) | 75.3 ms (0.4×) | 5.96 ms (4.7×) | 4.48 ms (6.3×) |

## numba parallelism actually engages — but only reaches parity

`P=1000, Q=10000`, thin_plate_spline, varying `NUMBA_NUM_THREADS`:

| threads | numba | vs pythran |
|---:|---:|---:|
| 1 | 661 ms | 0.09× |
| 14 | 73 ms | 0.80× |

Single-threaded numba is **~11× slower than pythran**, and 14-way `prange` only
claws it back to ~0.8×. The imperative kernel calls `np.linalg.norm(x - y[i])`
`Q×P` times, each allocating a temp; pythran fuses that into a temp-free scalar
loop. So numba spends its parallelism compensating for per-iteration codegen,
not getting ahead.

## Takeaways

1. **Vectorized + JIT wins on CPU.** jax (`jax.jit`) and torch (`torch.compile`)
   run 3–10× over the pythran baseline by fusing the whole `(Q, P)` batch and
   multithreading it. torch's inductor backend is fastest at scale (9.6×).
2. **Eager numpy is the floor** (~0.4×): it materializes the `(Q, P, N)`
   broadcast and a `(Q, P)` distance matrix with no fusion.
3. **numba ≈ pythran, both AOT/loop-style**, and both lose to the vectorized
   JITs here. numba needs all cores to match pythran's single core — a codegen
   gap on `np.linalg.norm`-in-a-loop, not a parallelism gap.
4. This is the PR's thesis on CPU: *write the kernel vectorized once, let a
   tracing JIT compile it.* The big numbers in the PR are CUDA (5–40×) and are
   **not reproducible here** — no GPU on this machine.

Reproduce: `python run.py` (see `README.md`).
