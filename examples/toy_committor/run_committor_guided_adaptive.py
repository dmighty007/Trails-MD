"""Closed-loop committor-guided local-stitch validation on Müller-Brown."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join(tempfile.gettempdir(), "trails-md-matplotlib"),
)

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.toy_committor.muller_brown import (  # noqa: E402
    MIN_A,
    MIN_B,
    basin_predicate,
    grad,
)
from trails_md.analysis.committor_acquisition import (  # noqa: E402
    CommittorAcquisitionWeights,
    score_local_stitch_cells,
)
from trails_md.analysis.local_stitch import (  # noqa: E402
    _destination_column,
    _solve_from_first_exit_counts,
    build_grid_states,
    first_exit_trajectory,
)
from trails_md.analysis.pde_reference import solve_pde_committor  # noqa: E402

LOGGER = logging.getLogger(__name__)

XLIM = (-1.6, 1.2)
YLIM = (-0.3, 2.1)
RADIUS_A = 0.12
RADIUS_B = 0.12


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
    parser.add_argument("--grid-size", type=int, default=18)
    parser.add_argument("--pde-grid", type=int, default=90)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--cells-per-iteration", type=int, default=24)
    parser.add_argument("--shots-per-cell", type=int, default=4)
    parser.add_argument(
        "--switch-mode",
        choices=("convergence", "fixed"),
        default="convergence",
        help=(
            "Use committor-guided allocation only after exploration coverage "
            "converges, or after a fixed warmup for controlled comparisons."
        ),
    )
    parser.add_argument("--warmup-iterations", type=int, default=10)
    parser.add_argument("--exploration-min-iterations", type=int, default=20)
    parser.add_argument("--exploration-patience", type=int, default=3)
    parser.add_argument("--exploration-max-new-observed", type=int, default=5)
    parser.add_argument("--exploration-max-new-finite", type=int, default=8)
    parser.add_argument("--exploration-max-isolated-drop", type=int, default=8)
    parser.add_argument("--dt", type=float, default=2.0e-4)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level, format="%(message)s")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    in_a = basin_predicate(MIN_A, RADIUS_A)
    in_b = basin_predicate(MIN_B, RADIUS_B)

    LOGGER.info("Solving PDE reference ...")
    pde = solve_pde_committor(
        grad,
        in_a,
        in_b,
        xlim=XLIM,
        ylim=YLIM,
        nx=args.pde_grid,
        ny=args.pde_grid,
    )

    weights = CommittorAcquisitionWeights()
    common = dict(
        in_a=in_a,
        in_b=in_b,
        pde=pde,
        grid_size=args.grid_size,
        iterations=args.iterations,
        cells_per_iteration=args.cells_per_iteration,
        shots_per_cell=args.shots_per_cell,
        switch_mode=args.switch_mode,
        warmup_iterations=args.warmup_iterations,
        exploration_min_iterations=args.exploration_min_iterations,
        exploration_patience=args.exploration_patience,
        exploration_max_new_observed=args.exploration_max_new_observed,
        exploration_max_new_finite=args.exploration_max_new_finite,
        exploration_max_isolated_drop=args.exploration_max_isolated_drop,
        dt=args.dt,
        max_steps=args.max_steps,
        weights=weights,
    )

    LOGGER.info("Running uniform baseline ...")
    uniform = run_closed_loop(strategy="uniform", seed=args.seed, **common)
    LOGGER.info("Running committor-guided allocation ...")
    adaptive = run_closed_loop(strategy="adaptive", seed=args.seed + 1, **common)

    report = {
        "grid_size": args.grid_size,
        "iterations": args.iterations,
        "cells_per_iteration": args.cells_per_iteration,
        "shots_per_cell": args.shots_per_cell,
        "switch_mode": args.switch_mode,
        "warmup_iterations": args.warmup_iterations,
        "exploration_convergence": {
            "min_iterations": args.exploration_min_iterations,
            "patience": args.exploration_patience,
            "max_new_observed": args.exploration_max_new_observed,
            "max_new_finite": args.exploration_max_new_finite,
            "max_isolated_drop": args.exploration_max_isolated_drop,
        },
        "max_steps": args.max_steps,
        "dt": args.dt,
        "weights": asdict(weights),
        "uniform": uniform["metrics"],
        "adaptive": adaptive["metrics"],
        "final": {
            "uniform_rmse_penalized": uniform["metrics"][-1]["rmse_penalized"],
            "adaptive_rmse_penalized": adaptive["metrics"][-1]["rmse_penalized"],
            "adaptive_minus_uniform_rmse_penalized": (
                adaptive["metrics"][-1]["rmse_penalized"]
                - uniform["metrics"][-1]["rmse_penalized"]
            ),
            "adaptive_rmse_ratio": (
                adaptive["metrics"][-1]["rmse_penalized"]
                / uniform["metrics"][-1]["rmse_penalized"]
            ),
            "uniform_isolated_states": uniform["metrics"][-1]["isolated_states"],
            "adaptive_isolated_states": adaptive["metrics"][-1]["isolated_states"],
            "uniform_transition_shot_fraction": uniform["metrics"][-1][
                "transition_shot_fraction"
            ],
            "adaptive_transition_shot_fraction": adaptive["metrics"][-1][
                "transition_shot_fraction"
            ],
            "adaptive_switch_iteration": adaptive["switch_iteration"],
            "adaptive_post_gate_iterations": (
                None
                if adaptive["switch_iteration"] is None
                else args.iterations - int(adaptive["switch_iteration"])
            ),
        },
    }

    report_path = args.output_dir / "committor_guided_adaptive_report.json"
    report_path.write_text(json.dumps(_json_ready(report), indent=2), encoding="utf-8")
    LOGGER.info(json.dumps(_json_ready(report["final"]), indent=2))

    _plot_convergence(
        args.output_dir / "committor_guided_adaptive_convergence.png",
        uniform["metrics"],
        adaptive["metrics"],
    )
    _plot_final_maps(
        args.output_dir / "committor_guided_adaptive_maps.png",
        pde,
        uniform["stitch"],
        adaptive["stitch"],
    )


def run_closed_loop(
    *,
    strategy: str,
    seed: int,
    in_a,
    in_b,
    pde,
    grid_size: int,
    iterations: int,
    cells_per_iteration: int,
    shots_per_cell: int,
    switch_mode: str,
    warmup_iterations: int,
    exploration_min_iterations: int,
    exploration_patience: int,
    exploration_max_new_observed: int,
    exploration_max_new_finite: int,
    exploration_max_isolated_drop: int,
    dt: float,
    max_steps: int,
    weights: CommittorAcquisitionWeights,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    grid = build_grid_states(XLIM, YLIM, grid_size, grid_size)
    n_states = grid_size * grid_size
    a_col = n_states
    b_col = n_states + 1
    counts = np.zeros((n_states, n_states + 2), dtype=float)
    subset_counts = np.zeros((n_states, 2, n_states + 2), dtype=float)
    is_a = np.asarray(in_a(grid.centers), dtype=bool).reshape(grid_size, grid_size)
    is_b = np.asarray(in_b(grid.centers), dtype=bool).reshape(grid_size, grid_size)
    candidates = np.flatnonzero(~(is_a | is_b).ravel())
    pde_on_grid = _reference_on_grid(pde, grid)
    transition_reference = (pde_on_grid > 0.2) & (pde_on_grid < 0.8)

    metrics: list[dict[str, float]] = []
    stitch = None
    shot_counter = 0
    adaptive_enabled = strategy == "adaptive" and switch_mode == "fixed" and warmup_iterations <= 0
    switch_iteration: int | None = 0 if adaptive_enabled else None
    for iteration in range(iterations):
        stitch = _solve_from_first_exit_counts(
            grid,
            counts,
            subset_counts,
            is_a,
            is_b,
            n_markov_subsets=2,
            markov_threshold_tv=0.25,
            markov_threshold_js=None,
        )
        use_adaptive = strategy == "adaptive" and adaptive_enabled
        selected = _select_cells(
            stitch,
            candidates,
            use_adaptive=use_adaptive,
            cells_per_iteration=cells_per_iteration,
            weights=weights,
            rng=rng,
        )
        for state in selected:
            i, j = divmod(int(state), grid_size)
            for _ in range(shots_per_cell):
                x0 = _jittered_cell_point(grid, i, j, rng)
                destination = first_exit_trajectory(
                    x0,
                    grad,
                    in_a,
                    in_b,
                    XLIM,
                    YLIM,
                    grid.dx,
                    grid.dy,
                    grid_size,
                    grid_size,
                    (i, j),
                    rng,
                    dt=dt,
                    max_steps=max_steps,
                )
                column = _destination_column(destination, grid.state_id, a_col, b_col)
                if column is not None:
                    counts[int(state), column] += 1.0
                    subset_counts[int(state), shot_counter % 2, column] += 1.0
                    shot_counter += 1

        stitch = _solve_from_first_exit_counts(
            grid,
            counts,
            subset_counts,
            is_a,
            is_b,
            n_markov_subsets=2,
            markov_threshold_tv=0.25,
            markov_threshold_js=None,
        )
        metrics.append(
            metric := _metrics(
                iteration,
                stitch,
                pde_on_grid,
                transition_reference,
                counts,
                is_a | is_b,
                phase="adaptive" if use_adaptive else "explore",
            )
        )
        if strategy == "adaptive" and not adaptive_enabled:
            adaptive_enabled = _exploration_converged(
                metrics,
                switch_mode=switch_mode,
                warmup_iterations=warmup_iterations,
                min_iterations=exploration_min_iterations,
                patience=exploration_patience,
                max_new_observed=exploration_max_new_observed,
                max_new_finite=exploration_max_new_finite,
                max_isolated_drop=exploration_max_isolated_drop,
            )
            if adaptive_enabled:
                switch_iteration = iteration + 1
                LOGGER.info(
                    "Switching %s run to committor-guided allocation at iteration %d",
                    strategy,
                    switch_iteration,
                )
        LOGGER.debug("%s iter %d: %s", strategy, iteration, metrics[-1])

    return {"metrics": metrics, "stitch": stitch, "switch_iteration": switch_iteration}


def _select_cells(
    stitch,
    candidates: np.ndarray,
    *,
    use_adaptive: bool,
    cells_per_iteration: int,
    weights: CommittorAcquisitionWeights,
    rng: np.random.Generator,
) -> np.ndarray:
    if not use_adaptive:
        return rng.choice(
            candidates,
            size=min(cells_per_iteration, len(candidates)),
            replace=False,
        )
    acquisition = score_local_stitch_cells(stitch, weights=weights)
    scores = acquisition.score.ravel()[candidates].astype(float)
    if scores.sum() <= 0 or not np.isfinite(scores).all():
        probabilities = None
        replace = False
    else:
        probabilities = scores / scores.sum()
        replace = int(np.count_nonzero(probabilities)) < min(
            cells_per_iteration, len(candidates)
        )
    return rng.choice(
        candidates,
        size=min(cells_per_iteration, len(candidates)),
        replace=replace,
        p=probabilities,
    )


def _jittered_cell_point(grid, i: int, j: int, rng: np.random.Generator) -> np.ndarray:
    jitter = np.array(
        [
            rng.uniform(-0.45 * grid.dx, 0.45 * grid.dx),
            rng.uniform(-0.45 * grid.dy, 0.45 * grid.dy),
        ]
    )
    point = np.array([grid.xs[i], grid.ys[j]], dtype=float) + jitter
    point[0] = np.clip(point[0], XLIM[0], XLIM[1])
    point[1] = np.clip(point[1], YLIM[0], YLIM[1])
    return point


def _metrics(
    iteration: int,
    stitch,
    reference: np.ndarray,
    transition_reference: np.ndarray,
    counts: np.ndarray,
    boundary: np.ndarray,
    phase: str,
) -> dict[str, float]:
    q = np.asarray(stitch.q, dtype=float)
    non_boundary = ~boundary
    finite = np.isfinite(q)
    q_penalized = np.where(finite, q, 0.5)
    total_counts = counts.sum(axis=1).reshape(q.shape)
    transition_counts = float(total_counts[transition_reference].sum())
    all_counts = float(total_counts.sum())
    return {
        "iteration": int(iteration),
        "phase": phase,
        "total_first_exits": int(all_counts),
        "rmse_penalized": _rmse(q_penalized[non_boundary], reference[non_boundary]),
        "rmse_finite": _rmse(q[finite & non_boundary], reference[finite & non_boundary])
        if np.any(finite & non_boundary)
        else None,
        "finite_states": int(finite.sum()),
        "isolated_states": int(len(stitch.isolated)),
        "observed_states": int((total_counts > 0).sum()),
        "transition_shot_fraction": transition_counts / all_counts
        if all_counts > 0
        else 0.0,
        "max_abs_residual": _nanmax_or_none(np.abs(stitch.residual)),
        "max_markov_total_variation": stitch.markov_check.max_total_variation,
        "n_markov_rejected": int(len(stitch.markov_check.rejected_states)),
    }


def _exploration_converged(
    metrics: list[dict[str, float]],
    *,
    switch_mode: str,
    warmup_iterations: int,
    min_iterations: int,
    patience: int,
    max_new_observed: int,
    max_new_finite: int,
    max_isolated_drop: int,
) -> bool:
    if switch_mode == "fixed":
        return len(metrics) >= warmup_iterations
    if switch_mode != "convergence":
        raise ValueError(f"unknown switch mode: {switch_mode}")
    if len(metrics) < max(min_iterations, patience + 1):
        return False

    recent = metrics[-(patience + 1) :]
    for before, after in zip(recent[:-1], recent[1:]):
        new_observed = int(after["observed_states"]) - int(before["observed_states"])
        new_finite = int(after["finite_states"]) - int(before["finite_states"])
        isolated_drop = int(before["isolated_states"]) - int(after["isolated_states"])
        if new_observed > max_new_observed:
            return False
        if new_finite > max_new_finite:
            return False
        if isolated_drop > max_isolated_drop:
            return False
    return True


def _reference_on_grid(pde, grid) -> np.ndarray:
    out = np.zeros(grid.state_id.shape, dtype=float)
    for i, x in enumerate(grid.xs):
        for j, y in enumerate(grid.ys):
            ix = int(np.argmin(np.abs(pde.xs - x)))
            iy = int(np.argmin(np.abs(pde.ys - y)))
            out[i, j] = pde.q[ix, iy]
    return out


def _rmse(values: np.ndarray, reference: np.ndarray) -> float:
    return float(np.sqrt(np.nanmean((values - reference) ** 2)))


def _nanmax_or_none(values: np.ndarray) -> float | None:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(np.max(finite)) if finite.size else None


def _plot_convergence(path: Path, uniform: list[dict], adaptive: list[dict]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for label, rows in [("uniform", uniform), ("adaptive", adaptive)]:
        x = [row["total_first_exits"] for row in rows]
        axes[0].plot(x, [row["rmse_penalized"] for row in rows], "o-", label=label)
        axes[1].plot(x, [row["isolated_states"] for row in rows], "o-", label=label)
        axes[2].plot(
            x, [row["transition_shot_fraction"] for row in rows], "o-", label=label
        )
    axes[0].set_ylabel("RMSE vs PDE (NaN -> 0.5)")
    axes[1].set_ylabel("isolated cells")
    axes[2].set_ylabel("transition-region shot fraction")
    for ax in axes:
        ax.set_xlabel("first-exit shots")
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_final_maps(path: Path, pde, uniform, adaptive) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    items = [
        ("PDE reference", pde.xs, pde.ys, pde.q),
        ("uniform local stitch", uniform.xs, uniform.ys, uniform.q),
        ("adaptive local stitch", adaptive.xs, adaptive.ys, adaptive.q),
    ]
    for ax, (title, xs, ys, q) in zip(axes, items):
        im = ax.contourf(xs, ys, q.T, levels=np.linspace(0, 1, 21), cmap="RdBu_r")
        ax.scatter(*MIN_A, c="k", marker="x")
        ax.scatter(*MIN_B, c="k", marker="x")
        ax.set_title(title)
        fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _json_ready(value):
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, np.floating | float):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer | int):
        return int(value)
    return value


if __name__ == "__main__":
    main()
