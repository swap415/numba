# RBF interpolation: numba vs jax vs torch vs pythran

A CPU replication of the cross-framework benchmark in
[scipy PR #23447](https://github.com/scipy/scipy/pull/23447) (Array API backends
for `RBFInterpolator`). It times **evaluation** of an RBF interpolant
(`compute_interpolation`: build the `(Q, P+R)` matrix of RBF distances +
polynomial monomials, then `@ coeffs`) across five backends.

## Backends

| backend | source | how |
|---|---|---|
| `pythran` | `rbf_pythran.py` | AOT-compiled scalar loops — scipy's default path, the PR baseline |
| `numpy`   | `kernel_xp.py`   | vectorized array-API source, eager |
| `numba`   | `kernel_numba.py`| imperative `@numba.jit(parallel=True)` + `prange` |
| `jax`     | `kernel_xp.py`   | `jax.jit` of the vectorized source |
| `torch`   | `kernel_xp.py`   | `torch.compile(fullgraph=True, dynamic=True)` of the vectorized source |

`kernel_xp.py` (vectorized) and `kernel_numba.py` (imperative) are faithful
ports of the PR's `_rbfinterp_xp.py` and `_rbfinterp_numba.py`
(`ev-br/scipy@5ccfca3`). The one source feeds numpy, jax, and torch unchanged —
only the compiler differs — via `array_api_compat.array_namespace`.

## Two layers

- **minimal** — synthetic coeffs, no linear solve. Isolates the eval kernel and
  thus compiler quality.
- **faithful** — a real `scipy.interpolate.RBFInterpolator` fit; evaluation runs
  on genuine solved coeffs/shift/scale/powers. The PR's actual code path.

## Run

```bash
unset PYTHONPATH                                  # avoid a dev-numba checkout leaking in
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install numba scipy jax torch pythran array-api-compat
pythran benchmarks/rbf_interp/rbf_pythran.py      # build the AOT baseline once
python benchmarks/rbf_interp/run.py               # both layers, default sweep
python run.py --layer minimal --kernel gaussian   # one layer, other kernel
```

Every backend's output is asserted equal to the numpy reference before timing;
`!` in a cell marks a mismatch. Speedup is reported against `pythran`.

## Caveat

CPU-only (Apple M4 Pro, no CUDA). The PR's headline 5–40× numbers are GPU
(CUDA jax/torch, CuPy) and **cannot** be reproduced here. This measures the CPU
story: AOT (pythran) vs JIT (numba/jax/torch) vs eager numpy. See `RESULTS.md`.
