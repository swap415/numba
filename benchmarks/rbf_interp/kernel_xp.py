"""Vectorized array-API RBF evaluation kernel.

Faithful port of scipy PR #23447's `_rbfinterp_xp.py` evaluation path
(ev-br/scipy@5ccfca3). A single source drives numpy / jax / torch via the
`xp` namespace argument -- the same body is fed to `jax.jit` and
`torch.compile`. Keep the kernel set in sync with `kernel_numba.py`, which
carries the imperative prange twin of this code.
"""


def linear(r, xp):
    return -r


def thin_plate_spline(r, xp):
    return xp.where(r == 0, 0.0, r**2 * xp.log(r))


def cubic(r, xp):
    return r**3


def quintic(r, xp):
    return -r**5


def multiquadric(r, xp):
    return -xp.sqrt(r**2 + 1.0)


def inverse_multiquadric(r, xp):
    return 1.0 / xp.sqrt(r**2 + 1.0)


def inverse_quadratic(r, xp):
    return 1.0 / (r**2 + 1.0)


def gaussian(r, xp):
    return xp.exp(-r**2)


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


def compute_interpolation(x, y, kernel, epsilon, powers, shift, scale, coeffs, xp):
    """Evaluate the RBF interpolant at `x`.

    x : (Q, N) eval points, y : (P, N) data points, powers : (R, N) monomials,
    coeffs : (P + R,) solved coefficients. Returns (Q,) interpolated values.
    """
    kernel_func = NAME_TO_FUNC[kernel]

    yeps = y * epsilon
    xeps = x * epsilon
    xhat = (x - shift) / scale

    vec = xp.concat(
        [
            kernel_func(
                xp.linalg.vector_norm(
                    xeps[:, None, :] - yeps[None, :, :], axis=-1
                ), xp
            ),
            xp.prod(xhat[:, None, :] ** powers, axis=-1),
        ], axis=-1
    )

    return vec @ coeffs
