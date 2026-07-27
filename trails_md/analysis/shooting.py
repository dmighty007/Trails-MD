"""Endpoint-shooting estimators for overdamped committors."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.stats import beta as beta_dist

ArrayFn = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class ShootingResult:
    """Jeffreys-Beta endpoint-shooting committor estimate."""

    q_mean: np.ndarray
    q_lo: np.ndarray
    q_hi: np.ndarray
    n_b: np.ndarray
    n_a: np.ndarray
    n_censored: np.ndarray


def overdamped_step(
    x: np.ndarray,
    grad_fn: ArrayFn,
    dt: float,
    beta: float,
    diffusion: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """One Euler-Maruyama step for overdamped Langevin dynamics."""

    gradient = np.asarray(grad_fn(x), dtype=float)
    noise = rng.normal(size=x.shape)
    return x - diffusion * beta * gradient * dt + np.sqrt(2.0 * diffusion * dt) * noise


def shoot_point(
    x0: np.ndarray,
    grad_fn: ArrayFn,
    in_basin_a: ArrayFn,
    in_basin_b: ArrayFn,
    n_shots: int,
    rng: np.random.Generator,
    *,
    dt: float = 1.0e-4,
    max_steps: int = 200_000,
    beta: float = 1.0,
    diffusion: float = 1.0,
) -> tuple[int, int, int]:
    """Run endpoint shots and return ``(n_hit_b, n_hit_a, n_censored)``."""

    n_b = 0
    n_a = 0
    n_censored = 0
    for _ in range(int(n_shots)):
        x = np.asarray(x0, dtype=float).copy()
        for _ in range(int(max_steps)):
            x = overdamped_step(x, grad_fn, dt, beta, diffusion, rng)
            if bool(np.asarray(in_basin_a(x))):
                n_a += 1
                break
            if bool(np.asarray(in_basin_b(x))):
                n_b += 1
                break
        else:
            n_censored += 1
    return n_b, n_a, n_censored


def committor_shoot(
    points: np.ndarray,
    grad_fn: ArrayFn,
    in_basin_a: ArrayFn,
    in_basin_b: ArrayFn,
    *,
    n_shots: int,
    seed: int = 0,
    dt: float = 1.0e-4,
    max_steps: int = 200_000,
    beta: float = 1.0,
    diffusion: float = 1.0,
) -> ShootingResult:
    """Estimate committors at points by direct endpoint shooting."""

    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2:
        raise ValueError("points must have shape (n_points, n_dimensions).")

    rng = np.random.default_rng(seed)
    n_pts = pts.shape[0]
    q_mean = np.zeros(n_pts, dtype=float)
    q_lo = np.zeros(n_pts, dtype=float)
    q_hi = np.zeros(n_pts, dtype=float)
    n_b = np.zeros(n_pts, dtype=int)
    n_a = np.zeros(n_pts, dtype=int)
    n_censored = np.zeros(n_pts, dtype=int)

    for idx, point in enumerate(pts):
        hits_b, hits_a, censored = shoot_point(
            point,
            grad_fn,
            in_basin_a,
            in_basin_b,
            n_shots,
            rng,
            dt=dt,
            max_steps=max_steps,
            beta=beta,
            diffusion=diffusion,
        )
        n_b[idx] = hits_b
        n_a[idx] = hits_a
        n_censored[idx] = censored
        posterior = beta_dist(hits_b + 0.5, hits_a + 0.5)
        q_mean[idx] = posterior.mean()
        q_lo[idx], q_hi[idx] = posterior.ppf(0.025), posterior.ppf(0.975)

    return ShootingResult(
        q_mean=q_mean,
        q_lo=q_lo,
        q_hi=q_hi,
        n_b=n_b,
        n_a=n_a,
        n_censored=n_censored,
    )
