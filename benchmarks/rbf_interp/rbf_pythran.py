"""Pythran AOT baseline -- mirrors scipy's compiled `_rbfinterp_pythran`.

This is the PR's reference backend: scalar-loop kernels, AOT-compiled by
pythran. Build once with:

    pythran rbf_pythran.py

The `@ coeffs` matmul deliberately stays in NumPy/BLAS, matching scipy's
`_rbfinterp_np` call chain (pythran builds `vec`, numpy multiplies).
"""
import numpy as np


def linear(r):
    return -r


def thin_plate_spline(r):
    if r == 0:
        return 0.0
    return r**2 * np.log(r)


def cubic(r):
    return r**3


def quintic(r):
    return -r**5


def multiquadric(r):
    return -np.sqrt(r**2 + 1)


def inverse_multiquadric(r):
    return 1 / np.sqrt(r**2 + 1)


def inverse_quadratic(r):
    return 1 / (r**2 + 1)


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


def kernel_vector(x, y, kernel_func, out):
    for i in range(y.shape[0]):
        out[i] = kernel_func(np.linalg.norm(x - y[i]))


def polynomial_vector(x, powers, out):
    for i in range(powers.shape[0]):
        out[i] = np.prod(x**powers[i])


# pythran export _build_evaluation_coefficients(float[:, :], float[:, :], str, float, int64[:, :], float[:], float[:])
def _build_evaluation_coefficients(x, y, kernel, epsilon, powers, shift, scale):
    kernel_func = NAME_TO_FUNC[kernel]
    q = x.shape[0]
    p = y.shape[0]
    r = powers.shape[0]

    yeps = y * epsilon
    xeps = x * epsilon
    xhat = (x - shift) / scale

    vec = np.empty((q, p + r))
    for i in range(q):
        kernel_vector(xeps[i], yeps, kernel_func, vec[i, :p])
        polynomial_vector(xhat[i], powers, vec[i, p:])
    return vec
