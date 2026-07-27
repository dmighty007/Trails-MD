"""Finite-difference reference committors for overdamped toy systems."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import spsolve

ArrayFn = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class PDECommittorResult:
    """Finite-difference committor on a rectangular grid."""

    xs: np.ndarray
    ys: np.ndarray
    q: np.ndarray
    is_a: np.ndarray
    is_b: np.ndarray


def solve_pde_committor(
    grad_fn: ArrayFn,
    in_basin_a: ArrayFn,
    in_basin_b: ArrayFn,
    *,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    nx: int = 120,
    ny: int = 120,
    beta: float = 1.0,
    diffusion: float = 1.0,
) -> PDECommittorResult:
    """Solve the backward-Kolmogorov boundary-value problem on a 2-D grid."""

    xs = np.linspace(*xlim, int(nx))
    ys = np.linspace(*ylim, int(ny))
    dx = xs[1] - xs[0]
    dy = ys[1] - ys[0]
    x_grid, y_grid = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([x_grid, y_grid], axis=-1)
    state_index = np.arange(nx * ny).reshape(nx, ny)

    is_a = _as_mask(in_basin_a(points), (nx, ny))
    is_b = _as_mask(in_basin_b(points), (nx, ny))
    gradients = np.asarray(grad_fn(points), dtype=float)
    if gradients.shape != points.shape:
        raise ValueError("grad_fn must return an array with shape (..., 2).")

    n_total = nx * ny
    operator = lil_matrix((n_total, n_total))
    rhs = np.zeros(n_total, dtype=float)

    diffusion_dx2 = diffusion / dx**2
    diffusion_dy2 = diffusion / dy**2
    for i in range(nx):
        for j in range(ny):
            row = state_index[i, j]
            if is_a[i, j]:
                operator[row, row] = 1.0
                continue
            if is_b[i, j]:
                operator[row, row] = 1.0
                rhs[row] = 1.0
                continue

            im1 = max(i - 1, 0)
            ip1 = min(i + 1, nx - 1)
            jm1 = max(j - 1, 0)
            jp1 = min(j + 1, ny - 1)
            gx = gradients[i, j, 0]
            gy = gradients[i, j, 1]

            operator[row, state_index[ip1, j]] += diffusion_dx2
            operator[row, state_index[im1, j]] += diffusion_dx2
            operator[row, state_index[i, jp1]] += diffusion_dy2
            operator[row, state_index[i, jm1]] += diffusion_dy2
            operator[row, row] += -2.0 * diffusion_dx2 - 2.0 * diffusion_dy2

            operator[row, state_index[ip1, j]] += -diffusion * beta * gx / (2.0 * dx)
            operator[row, state_index[im1, j]] += diffusion * beta * gx / (2.0 * dx)
            operator[row, state_index[i, jp1]] += -diffusion * beta * gy / (2.0 * dy)
            operator[row, state_index[i, jm1]] += diffusion * beta * gy / (2.0 * dy)

    q = spsolve(csr_matrix(operator), rhs)
    return PDECommittorResult(xs=xs, ys=ys, q=q.reshape(nx, ny), is_a=is_a, is_b=is_b)


def _as_mask(values: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    mask = np.asarray(values, dtype=bool)
    if mask.shape != shape:
        raise ValueError(
            f"basin predicate must return shape {shape}, got {mask.shape}."
        )
    return mask
