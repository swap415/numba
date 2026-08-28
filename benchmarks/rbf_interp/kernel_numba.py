"""Imperative, prange-parallel numba RBF evaluation kernel.

Faithful port of scipy PR #23447's `_rbfinterp_numba.py` (ev-br/scipy@5ccfca3),
the imperative twin of the vectorized `kernel_xp.py`. Scalar kernels are passed
as first-class jitted functions into one parallel driver. Kept as `@numba.jit`
(numba's preferred API, defaults to nopython) rather than the PR's `@njit`.
Keep the kernel set in sync with `kernel_xp.py`.
"""
import numba
import numpy as np


@numba.jit
def linear(r):
    return -r


@numba.jit
def thin_plate_spline(r):
    if r == 0:
        return 0.0
    return r**2 * np.log(r)


@numba.jit
def cubic(r):
    return r**3


@numba.jit
def quintic(r):
    return -r**5


@numba.jit
def multiquadric(r):
    return -np.sqrt(r**2 + 1)


@numba.jit
def inverse_multiquadric(r):
    return 1 / np.sqrt(r**2 + 1)


@numba.jit
def inverse_quadratic(r):
    return 1 / (r**2 + 1)


@numba.jit
def gaussian(r):
    return np.exp(-r**2)


NAME_TO_FUNC = {
    "linear": linear,
    "thin_plate_spline": thin_plate_spline,
    "cubic": cubic,
    "quintic": quintic,
    "multiquadric": multiquadric,
    "inverse_multiquadric": inverse_multiquadric,
    "inverse_quadratic": inverse_quadratic,
    "gaussian": gaussian,
}


@numba.jit
def kernel_vector(x, y, kernel_func, out):
    """Evaluate RBFs, with centers at `y`, at the point `x`."""
    for i in numba.prange(y.shape[0]):
        out[i] = kernel_func(np.linalg.norm(x - y[i]))


@numba.jit
def polynomial_vector(x, powers, out):
    """Evaluate monomials, with exponents from `powers`, at the point `x`."""
    for i in numba.prange(powers.shape[0]):
        out[i] = np.prod(x**powers[i])


@numba.jit(parallel=True)
def _build_evaluation_coefficients_impl(x, y, kernel_func, epsilon, powers, shift, scale):
    q = x.shape[0]
    p = y.shape[0]
    r = powers.shape[0]

    yeps = y * epsilon
    xeps = x * epsilon
    xhat = (x - shift) / scale

    vec = np.empty((q, p + r), dtype=np.float64)
    for i in numba.prange(q):
        kernel_vector(xeps[i], yeps, kernel_func, vec[i, :p])
        polynomial_vector(xhat[i], powers, vec[i, p:])
    return vec


@numba.jit
def _interp_impl(x, y, kernel_func, epsilon, powers, shift, scale, coeffs):
    vec = _build_evaluation_coefficients_impl(
        x, y, kernel_func, epsilon, powers, shift, scale
    )
    return vec @ coeffs


def compute_interpolation(x, y, kernel, epsilon, powers, shift, scale, coeffs, xp=None):
    """numpy-array entry point; signature mirrors `kernel_xp.compute_interpolation`."""
    return _interp_impl(
        x, y, NAME_TO_FUNC[kernel], epsilon, powers, shift, scale, coeffs
    )
