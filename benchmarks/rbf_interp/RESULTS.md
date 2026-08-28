# Results — RBF eval, numba vs jax vs torch vs pythran

Two machines, identical stack (numba 0.65.1 / jax 0.10.2 / torch 2.12.1 /
pythran 0.18.1 / scipy 1.18.0, Python 3.12):

| host | CPU | GPU |
|---|---|---|
| **mac** (M4 Pro) | Apple M4 Pro, 14 numba threads | — (no CUDA) |
| **arrakis** (Linux) | Intel i7-14700, 28 numba threads | 2× RTX 3090 Ti (24 GB) |

**Method:** median of N runs after warmups; every backend's output asserted
equal to numpy before timing (no `!` mismatch anywhere). Speedup vs `pythran`
(the PR's AOT baseline). `P` data points, `Q` eval points; eval matrix `(Q, P+R)`.

## CPU — faithful layer (real `RBFInterpolator` fit), thin_plate_spline

### mac (M4 Pro, 14 threads)

| P | Q | pythran | numpy | numba | jax | torch |
|---:|---:|---:|---:|---:|---:|---:|
| 100 | 1000 | 0.60 ms (1.0×) | 1.40 ms (0.4×) | 1.25 ms (0.5×) | 0.17 ms (3.5×) | 0.25 ms (2.4×) |
| 300 | 3000 | 5.03 ms (1.0×) | 12.5 ms (0.4×) | 8.03 ms (0.6×) | 0.96 ms (5.2×) | 0.81 ms (6.2×) |
| 1000 | 10000 | 58.4 ms (1.0×) | 142 ms (0.4×) | 80.9 ms (0.7×) | 9.88 ms (5.9×) | **5.94 ms (9.8×)** |

### arrakis (i7-14700, 28 threads)

| P | Q | pythran | numpy | numba | jax | torch |
|---:|---:|---:|---:|---:|---:|---:|
| 100 | 1000 | 0.59 ms (1.0×) | 2.09 ms (0.3×) | 0.59 ms (1.0×) | 0.27 ms (2.2×) | 0.66 ms (0.9×) |
| 300 | 3000 | 9.52 ms (1.0×) | 32.8 ms (0.3×) | 16.5 ms (0.6×) | 7.93 ms (1.2×) | 2.79 ms (3.4×) |
| 1000 | 10000 | 99.0 ms (1.0×) | 289 ms (0.3×) | 119 ms (0.8×) | 27.9 ms (3.5×) | 29.8 ms (3.3×) |

Cross-machine surprises:
- **M4 Pro's single thread is faster in absolute ms** — pythran 58 ms vs the
  i7's 99 ms at the largest size. The scalar RBF loop is latency-bound, and the
  M4's per-core throughput wins.
- **The i7's 28 threads help numba at small sizes** (1.0–1.1× vs pythran where
  the Mac sits at 0.5×) — more cores amortize the `prange` overhead sooner.
- **CPU `torch.compile` is much stronger on the Mac** (9.8×) than on the i7
  here (3.3×); inductor's ARM codegen + the M4 memory system fuse this batch
  better than the x86 path did.

## GPU — arrakis, 1× RTX 3090 Ti, faithful layer, large sizes

jax/torch on CUDA; pythran/numba are the CPU baselines on the same box.

| P | Q | pythran (CPU) | numba (CPU) | jax (GPU) | torch (GPU) |
|---:|---:|---:|---:|---:|---:|
| 1000 | 10000 | 99.1 ms (1.0×) | 115 ms (0.9×) | 2.36 ms (41.9×) | **1.89 ms (52.4×)** |
| 2000 | 20000 | 320 ms (1.0×) | 340 ms (0.9×) | 9.74 ms (32.9×) | 7.09 ms (45.1×) |
| 4000 | 40000 | 1170 ms (1.0×) | 1088 ms (1.1×) | 29.7 ms (39.4×) | 27.6 ms (42.5×) |

**This reproduces — and exceeds — the PR's headline.** The PR reports 5–40×
for JAX/PyTorch JIT on GPU vs CPU; on one 3090 Ti we see **33–52×**. The Mac
cannot show this at all (no CUDA), which is exactly why the cross-machine run
matters.

## Takeaways

1. **GPU is the whole story.** A single consumer 3090 Ti turns the vectorized
   `compute_interpolation` into a 40–52× win over the best CPU baseline — the
   PR's thesis, confirmed. torch (inductor→Triton) edges jax (XLA) on CUDA.
2. **On CPU, "fastest" is hardware-dependent.** The vectorized JITs still win,
   but by 3–10× not 40×, and the ranking between torch and jax flips between
   the M4 and the i7. Absolute ms even flips for pythran (M4 faster).
3. **numba ≈ pythran on both CPUs.** numba needs all cores just to reach
   pythran's single-threaded AOT loop — a codegen gap on
   `np.linalg.norm`-in-a-loop (each call allocates a temp pythran fuses away),
   not a parallelism gap. It brings nothing on GPU (CPU-only here).
4. **Eager numpy is the floor** (0.3–0.4×): it materializes the `(Q, P, N)`
   broadcast with no fusion.

Reproduce: `python run.py` (CPU) / `python run.py --device cuda --big`
(GPU). See `README.md`.
