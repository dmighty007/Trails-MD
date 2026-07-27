"""Local first-exit committor stitching on a rectangular state grid."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

from .markov_check import MarkovCheckResult, compare_transition_subsets
from .shooting import ArrayFn, overdamped_step

InitialPointFn = Callable[[int, int, np.random.Generator], np.ndarray]
BasinDestination = Literal["A", "B", "cell", "censored"]


@dataclass(frozen=True)
class GridStates:
    """Regular 2-D grid used for local first-exit states."""

    xs: np.ndarray
    ys: np.ndarray
    dx: float
    dy: float
    centers: np.ndarray
    state_id: np.ndarray


@dataclass(frozen=True)
class LocalStitchResult:
    """Locally stitched committor and diagnostics."""

    xs: np.ndarray
    ys: np.ndarray
    q: np.ndarray
    state_id: np.ndarray
    transition_matrix: np.ndarray
    is_a: np.ndarray
    is_b: np.ndarray
    isolated: np.ndarray
    residual: np.ndarray
    counts: np.ndarray
    markov_check: MarkovCheckResult
    markov_total_variation: np.ndarray
    markov_jensen_shannon: np.ndarray


def build_grid_states(
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    nx: int,
    ny: int,
) -> GridStates:
    """Build regular-grid state centers and integer IDs."""

    xs = np.linspace(*xlim, int(nx))
    ys = np.linspace(*ylim, int(ny))
    dx = float(xs[1] - xs[0])
    dy = float(ys[1] - ys[0])
    centers = np.array([[xs[i], ys[j]] for i in range(nx) for j in range(ny)])
    state_id = np.arange(nx * ny, dtype=int).reshape(nx, ny)
    return GridStates(xs=xs, ys=ys, dx=dx, dy=dy, centers=centers, state_id=state_id)


def cell_of(
    xy: np.ndarray,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    dx: float,
    dy: float,
    nx: int,
    ny: int,
) -> tuple[int, int]:
    """Return the nearest grid-state index for a point."""

    i = int(np.clip(round((float(xy[0]) - xlim[0]) / dx), 0, nx - 1))
    j = int(np.clip(round((float(xy[1]) - ylim[0]) / dy), 0, ny - 1))
    return i, j


def first_exit_trajectory(
    x0: np.ndarray,
    grad_fn: ArrayFn,
    in_basin_a: ArrayFn,
    in_basin_b: ArrayFn,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    dx: float,
    dy: float,
    nx: int,
    ny: int,
    home_ij: tuple[int, int],
    rng: np.random.Generator,
    *,
    dt: float = 1.0e-4,
    max_steps: int = 20_000,
    beta: float = 1.0,
    diffusion: float = 1.0,
) -> tuple[str, int | None, int | None]:
    """Run until a trajectory leaves its home cell or reaches A/B."""

    x = np.asarray(x0, dtype=float).copy()
    for _ in range(int(max_steps)):
        x = overdamped_step(x, grad_fn, dt, beta, diffusion, rng)
        if bool(np.asarray(in_basin_a(x))):
            return ("A", None, None)
        if bool(np.asarray(in_basin_b(x))):
            return ("B", None, None)
        cell = cell_of(x, xlim, ylim, dx, dy, nx, ny)
        if cell != home_ij:
            return ("cell", cell[0], cell[1])
    return ("censored", None, None)


def committor_local_stitch(
    grad_fn: ArrayFn,
    in_basin_a: ArrayFn,
    in_basin_b: ArrayFn,
    *,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    nx: int,
    ny: int,
    n_shots_per_cell: int = 50,
    seed: int = 0,
    dt: float = 1.0e-4,
    max_steps: int = 20_000,
    beta: float = 1.0,
    diffusion: float = 1.0,
    initial_point_fn: InitialPointFn | None = None,
    n_markov_subsets: int = 2,
    markov_threshold_tv: float = 0.25,
    markov_threshold_js: float | None = None,
) -> LocalStitchResult:
    """Estimate local first-exit rows and solve a stitched committor."""

    grid = build_grid_states(xlim, ylim, nx, ny)
    n_states = int(nx * ny)
    a_col = n_states
    b_col = n_states + 1
    counts = np.zeros((n_states, n_states + 2), dtype=float)
    subset_counts = np.zeros((n_states, n_markov_subsets, n_states + 2), dtype=float)
    rng = np.random.default_rng(seed)

    is_a = _as_mask(in_basin_a(grid.centers), (nx, ny))
    is_b = _as_mask(in_basin_b(grid.centers), (nx, ny))

    for i in range(nx):
        for j in range(ny):
            state = int(grid.state_id[i, j])
            if is_a[i, j] or is_b[i, j]:
                continue
            for shot in range(int(n_shots_per_cell)):
                x0 = (
                    np.asarray(initial_point_fn(i, j, rng), dtype=float)
                    if initial_point_fn is not None
                    else np.array([grid.xs[i], grid.ys[j]], dtype=float)
                )
                destination = first_exit_trajectory(
                    x0,
                    grad_fn,
                    in_basin_a,
                    in_basin_b,
                    xlim,
                    ylim,
                    grid.dx,
                    grid.dy,
                    nx,
                    ny,
                    (i, j),
                    rng,
                    dt=dt,
                    max_steps=max_steps,
                    beta=beta,
                    diffusion=diffusion,
                )
                column = _destination_column(destination, grid.state_id, a_col, b_col)
                if column is not None:
                    counts[state, column] += 1.0
                    subset_counts[state, shot % n_markov_subsets, column] += 1.0

    return _solve_from_first_exit_counts(
        grid,
        counts,
        subset_counts,
        is_a,
        is_b,
        n_markov_subsets=n_markov_subsets,
        markov_threshold_tv=markov_threshold_tv,
        markov_threshold_js=markov_threshold_js,
    )


def committor_local_stitch_from_segments(
    segments: list[np.ndarray],
    in_basin_a: ArrayFn,
    in_basin_b: ArrayFn,
    *,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    nx: int,
    ny: int,
    n_markov_subsets: int = 2,
    markov_threshold_tv: float = 0.25,
    markov_threshold_js: float | None = None,
) -> LocalStitchResult:
    """Estimate stitched committor from observed trajectory first exits.

    Each contiguous residence in a non-basin grid cell contributes one local
    first-exit observation. The destination is the first later frame that either
    reaches basin A/B or leaves the home grid cell.
    """

    grid = build_grid_states(xlim, ylim, nx, ny)
    n_states = int(nx * ny)
    a_col = n_states
    b_col = n_states + 1
    counts = np.zeros((n_states, n_states + 2), dtype=float)
    subset_counts = np.zeros((n_states, n_markov_subsets, n_states + 2), dtype=float)

    is_a = _as_mask(in_basin_a(grid.centers), (nx, ny))
    is_b = _as_mask(in_basin_b(grid.centers), (nx, ny))

    for segment_index, segment in enumerate(segments):
        points = np.asarray(segment, dtype=float)
        if points.ndim != 2 or points.shape[1] < 2:
            raise ValueError("segments must contain arrays shaped (frames, >=2).")
        subset = segment_index % n_markov_subsets
        _count_segment_first_exits(
            points[:, :2],
            in_basin_a,
            in_basin_b,
            grid,
            xlim,
            ylim,
            counts,
            subset_counts,
            subset,
            a_col,
            b_col,
        )

    return _solve_from_first_exit_counts(
        grid,
        counts,
        subset_counts,
        is_a,
        is_b,
        n_markov_subsets=n_markov_subsets,
        markov_threshold_tv=markov_threshold_tv,
        markov_threshold_js=markov_threshold_js,
    )


def _count_segment_first_exits(
    points: np.ndarray,
    in_basin_a: ArrayFn,
    in_basin_b: ArrayFn,
    grid: GridStates,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    counts: np.ndarray,
    subset_counts: np.ndarray,
    subset: int,
    a_col: int,
    b_col: int,
) -> None:
    n_frames = points.shape[0]
    frame = 0
    while frame < n_frames - 1:
        x0 = points[frame]
        if bool(np.asarray(in_basin_a(x0))) or bool(np.asarray(in_basin_b(x0))):
            frame += 1
            continue
        home = cell_of(x0, xlim, ylim, grid.dx, grid.dy, len(grid.xs), len(grid.ys))
        state = int(grid.state_id[home])
        destination: tuple[BasinDestination, int | None, int | None] = (
            "censored",
            None,
            None,
        )
        exit_frame = frame + 1
        for probe in range(frame + 1, n_frames):
            x = points[probe]
            if bool(np.asarray(in_basin_a(x))):
                destination = ("A", None, None)
                exit_frame = probe
                break
            if bool(np.asarray(in_basin_b(x))):
                destination = ("B", None, None)
                exit_frame = probe
                break
            cell = cell_of(
                x,
                xlim,
                ylim,
                grid.dx,
                grid.dy,
                len(grid.xs),
                len(grid.ys),
            )
            if cell != home:
                destination = ("cell", cell[0], cell[1])
                exit_frame = probe
                break
        column = _destination_column(destination, grid.state_id, a_col, b_col)
        if column is not None:
            counts[state, column] += 1.0
            subset_counts[state, subset, column] += 1.0
        frame = max(exit_frame, frame + 1)


def _solve_from_first_exit_counts(
    grid: GridStates,
    counts: np.ndarray,
    subset_counts: np.ndarray,
    is_a: np.ndarray,
    is_b: np.ndarray,
    *,
    n_markov_subsets: int,
    markov_threshold_tv: float,
    markov_threshold_js: float | None,
) -> LocalStitchResult:
    nx = len(grid.xs)
    ny = len(grid.ys)
    n_states = int(nx * ny)
    b_col = n_states + 1
    transition_matrix = _row_normalize_observed_support(counts)
    a_states = np.flatnonzero(is_a.ravel())
    b_states = np.flatnonzero(is_b.ravel())
    boundary = np.zeros(n_states, dtype=bool)
    boundary[a_states] = True
    boundary[b_states] = True
    has_data = counts.sum(axis=1) > 0
    reaches_boundary = _states_reaching_observed_boundary(counts, boundary)
    interior = np.flatnonzero((~boundary) & has_data & reaches_boundary)
    isolated = np.flatnonzero((~boundary) & ((~has_data) | (~reaches_boundary)))

    q = np.full(n_states, np.nan, dtype=float)
    q[a_states] = 0.0
    q[b_states] = 1.0
    if interior.size:
        index_map = {int(state): idx for idx, state in enumerate(interior)}
        matrix = np.zeros((interior.size, interior.size), dtype=float)
        rhs = np.zeros(interior.size, dtype=float)
        for row, state in enumerate(interior):
            matrix[row, row] = 1.0
            rhs[row] += transition_matrix[state, b_col]
            rhs[row] += transition_matrix[state, b_states].sum()
            for target in interior:
                matrix[row, index_map[int(target)]] -= transition_matrix[state, target]
        q[interior] = spsolve(csr_matrix(matrix), rhs)

    residual = np.full(n_states, np.nan, dtype=float)
    for state in interior:
        prediction = transition_matrix[state, b_col]
        prediction += np.nansum(transition_matrix[state, :n_states] * q)
        residual[state] = q[state] - prediction

    markov = compare_transition_subsets(
        subset_counts,
        threshold_tv=markov_threshold_tv,
        threshold_js=markov_threshold_js,
    )
    return LocalStitchResult(
        xs=grid.xs,
        ys=grid.ys,
        q=q.reshape(nx, ny),
        state_id=grid.state_id,
        transition_matrix=transition_matrix,
        is_a=is_a,
        is_b=is_b,
        isolated=isolated,
        residual=residual.reshape(nx, ny),
        counts=counts,
        markov_check=markov,
        markov_total_variation=markov.total_variation.reshape(nx, ny),
        markov_jensen_shannon=markov.jensen_shannon.reshape(nx, ny),
    )


def _states_reaching_observed_boundary(
    counts: np.ndarray,
    boundary: np.ndarray,
) -> np.ndarray:
    n_states = boundary.shape[0]
    support = counts > 0
    reachable = boundary.copy()
    direct_boundary_exit = support[:, n_states:].any(axis=1)
    queue = list(np.flatnonzero(reachable | direct_boundary_exit))
    reachable[direct_boundary_exit] = True
    while queue:
        state = int(queue.pop())
        predecessors = np.flatnonzero(support[:, state] & (~reachable))
        reachable[predecessors] = True
        queue.extend(int(pred) for pred in predecessors)
    return reachable


def _destination_column(
    destination: tuple[str, int | None, int | None],
    state_id: np.ndarray,
    a_col: int,
    b_col: int,
) -> int | None:
    label, i, j = destination
    if label == "A":
        return a_col
    if label == "B":
        return b_col
    if label == "cell" and i is not None and j is not None:
        return int(state_id[i, j])
    return None


def _row_normalize_observed_support(
    counts: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    rows = np.asarray(counts, dtype=float)
    out = np.zeros_like(rows, dtype=float)
    row_sum = rows.sum(axis=1)
    for idx, total in enumerate(row_sum):
        if total <= 0:
            continue
        support = rows[idx] > 0
        smoothed = np.zeros(rows.shape[1], dtype=float)
        smoothed[support] = rows[idx, support] + alpha
        out[idx] = smoothed / smoothed.sum()
    return out


def _as_mask(values: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    mask = np.asarray(values, dtype=bool)
    if mask.shape == (shape[0] * shape[1],):
        return mask.reshape(shape)
    if mask.shape != shape:
        raise ValueError(
            f"basin predicate must return shape {shape}, got {mask.shape}."
        )
    return mask
