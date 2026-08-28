"""Construct concrete RBF evaluation `Problem`s for the two benchmark layers.

minimal : synthetic coeffs, no linear solve -- isolates the eval kernel /
          compiler quality. The PR's "compute_interpolation only" microbench.
faithful: a real `scipy.interpolate.RBFInterpolator` fit; we time evaluation on
          genuine solved coeffs -- the PR's actual code path.
"""
import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.interpolate._rbfinterp_common import _monomial_powers_impl

from backends import Problem


def make_minimal(P, Q, N=2, kernel="thin_plate_spline", degree=1, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.random((P, N))
    x = rng.random((Q, N))
    powers = np.asarray(_monomial_powers_impl(N, degree), dtype=np.int64)
    if powers.shape[0] == 0:
        powers = powers.reshape(0, N)
    R = powers.shape[0]
    mins, maxs = y.min(0), y.max(0)
    shift = (maxs + mins) / 2
    scale = np.where((maxs - mins) / 2 == 0, 1.0, (maxs - mins) / 2)
    coeffs = rng.random(P + R)
    return Problem(x, y, kernel, 1.0, powers, shift, scale, coeffs)


def make_faithful(P, Q, N=2, kernel="thin_plate_spline", degree=1, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.random((P, N))
    d = rng.random(P)
    x = rng.random((Q, N))
    it = RBFInterpolator(y, d, kernel=kernel, epsilon=1.0, degree=degree)
    return Problem(x, y, kernel, float(it.epsilon), np.asarray(it.powers),
                   np.asarray(it._shift), np.asarray(it._scale),
                   np.asarray(it._coeffs)[:, 0])
