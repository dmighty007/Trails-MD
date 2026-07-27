"""Müller-Brown potential and basin predicates for committor validation."""

from __future__ import annotations

import numpy as np

A = np.array([-200.0, -100.0, -170.0, 15.0])
a = np.array([-1.0, -1.0, -6.5, 0.7])
b = np.array([0.0, 0.0, 11.0, 0.6])
c = np.array([-10.0, -10.0, -6.5, 0.7])
x0 = np.array([1.0, 0.0, -0.5, -1.0])
y0 = np.array([0.0, 0.5, 1.5, 1.0])

SCALE = 0.05
MIN_A = np.array([-0.558, 1.442])
MIN_B = np.array([0.623, 0.028])


def potential(xy: np.ndarray) -> np.ndarray:
    """Return the scaled Müller-Brown potential at ``xy``."""

    points = np.asarray(xy, dtype=float)
    x = points[..., 0]
    y = points[..., 1]
    value = np.zeros(x.shape, dtype=float)
    for idx in range(4):
        dx = x - x0[idx]
        dy = y - y0[idx]
        value += A[idx] * np.exp(
            a[idx] * dx**2 + b[idx] * dx * dy + c[idx] * dy**2
        )
    return SCALE * value


def grad(xy: np.ndarray) -> np.ndarray:
    """Return ``dV/d(x, y)`` for the scaled Müller-Brown potential."""

    points = np.asarray(xy, dtype=float)
    x = points[..., 0]
    y = points[..., 1]
    dvdx = np.zeros(x.shape, dtype=float)
    dvdy = np.zeros(x.shape, dtype=float)
    for idx in range(4):
        dx = x - x0[idx]
        dy = y - y0[idx]
        expo = A[idx] * np.exp(
            a[idx] * dx**2 + b[idx] * dx * dy + c[idx] * dy**2
        )
        dvdx += expo * (2.0 * a[idx] * dx + b[idx] * dy)
        dvdy += expo * (b[idx] * dx + 2.0 * c[idx] * dy)
    return SCALE * np.stack([dvdx, dvdy], axis=-1)


def basin_predicate(center: np.ndarray, radius: float):
    """Build a vectorized circular basin predicate."""

    center = np.asarray(center, dtype=float)
    radius2 = float(radius) ** 2

    def _predicate(xy: np.ndarray) -> np.ndarray:
        points = np.asarray(xy, dtype=float)
        delta = points - center
        return np.sum(delta * delta, axis=-1) < radius2

    return _predicate
