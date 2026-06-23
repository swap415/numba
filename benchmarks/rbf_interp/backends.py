"""Backend builders for the RBF evaluation benchmark.

Each `build_*(problem)` returns a zero-arg callable that evaluates the
interpolant and returns a NumPy `(Q,)` array, so the harness can compare
outputs and time them uniformly. The vectorized backends (numpy/jax/torch)
share `kernel_xp.compute_interpolation`; numba uses its imperative twin;
pythran is the AOT baseline. The `xp` namespace comes from
`array_api_compat.array_namespace`, matching scipy's `_rbfinterp.py`.
"""
from dataclasses import dataclass

import numpy as np

import kernel_xp
import kernel_numba


@dataclass
class Problem:
    """A concrete RBF evaluation: eval points, data points, solved coeffs."""
    x: np.ndarray        # (Q, N) eval points
    y: np.ndarray        # (P, N) data points
    kernel: str
    epsilon: float
    powers: np.ndarray   # (R, N) int monomial exponents
    shift: np.ndarray    # (N,)
    scale: np.ndarray    # (N,)
    coeffs: np.ndarray   # (P + R,)


def build_numpy(p):
    from array_api_compat import array_namespace
    xp = array_namespace(p.x)

    def run():
        return np.asarray(kernel_xp.compute_interpolation(
            p.x, p.y, p.kernel, p.epsilon, p.powers, p.shift, p.scale,
            p.coeffs, xp))
    return run


def build_numba(p):
    def run():
        return np.asarray(kernel_numba.compute_interpolation(
            p.x, p.y, p.kernel, p.epsilon, p.powers, p.shift, p.scale,
            p.coeffs))
    return run


def build_pythran(p):
    import rbf_pythran
    def run():
        vec = rbf_pythran._build_evaluation_coefficients(
            p.x, p.y, p.kernel, p.epsilon, p.powers, p.shift, p.scale)
        return vec @ p.coeffs
    return run


def build_jax(p):
    import jax
    import jax.numpy as jnp
    from array_api_compat import array_namespace

    x, y = jnp.asarray(p.x), jnp.asarray(p.y)
    powers = jnp.asarray(p.powers)
    shift, scale = jnp.asarray(p.shift), jnp.asarray(p.scale)
    coeffs = jnp.asarray(p.coeffs)
    xp = array_namespace(x)
    fn = jax.jit(kernel_xp.compute_interpolation,
                 static_argnames=["kernel", "xp"])

    def run():
        out = fn(x, y, p.kernel, p.epsilon, powers, shift, scale, coeffs, xp)
        out.block_until_ready()
        return np.asarray(out)
    return run


def build_torch(p):
    import torch
    from array_api_compat import array_namespace

    f64 = torch.float64
    x = torch.tensor(p.x, dtype=f64)
    y = torch.tensor(p.y, dtype=f64)
    powers = torch.tensor(p.powers, dtype=torch.int64)
    shift = torch.tensor(p.shift, dtype=f64)
    scale = torch.tensor(p.scale, dtype=f64)
    coeffs = torch.tensor(p.coeffs, dtype=f64)
    xp = array_namespace(x)
    fn = torch.compile(kernel_xp.compute_interpolation,
                       fullgraph=True, dynamic=True)

    def run():
        with torch.no_grad():
            out = fn(x, y, p.kernel, p.epsilon, powers, shift, scale, coeffs, xp)
        return out.cpu().numpy()
    return run


BACKENDS = {
    "numpy": build_numpy,
    "pythran": build_pythran,
    "numba": build_numba,
    "jax": build_jax,
    "torch": build_torch,
}
