"""Validate committor estimators on the Müller-Brown toy potential."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from trails_md.analysis.local_stitch import committor_local_stitch  # noqa: E402
from trails_md.analysis.pde_reference import solve_pde_committor  # noqa: E402
from trails_md.analysis.shooting import committor_shoot  # noqa: E402

from examples.toy_committor.muller_brown import (  # noqa: E402
    MIN_A,
    MIN_B,
    basin_predicate,
    grad,
)

LOGGER = logging.getLogger(__name__)
XLIM = (-1.6, 1.2)
YLIM = (-0.3, 2.1)
RADIUS_A = 0.12
RADIUS_B = 0.12


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
    parser.add_argument("--resolutions", type=int, nargs="+", default=[18, 30, 40])
    parser.add_argument("--pde-grid", type=int, default=90)
    parser.add_argument("--shots", type=int, default=40)
    parser.add_argument("--direct-shots", type=int, default=60)
    parser.add_argument("--max-steps", type=int, default=8000)
    parser.add_argument("--baseline-max-steps", type=int, default=4000)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level, format="%(message)s")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    in_a = basin_predicate(MIN_A, RADIUS_A)
    in_b = basin_predicate(MIN_B, RADIUS_B)

    LOGGER.info("[1/5] Solving PDE reference committor ...")
    pde = solve_pde_committor(
        grad,
        in_a,
        in_b,
        xlim=XLIM,
        ylim=YLIM,
        nx=args.pde_grid,
        ny=args.pde_grid,
    )

    LOGGER.info("[2/5] Running direct-shooting committor at probe points ...")
    fractions = np.linspace(0.05, 0.95, 9)
    probes = (
        (1.0 - fractions)[:, None] * MIN_A[None, :]
        + fractions[:, None] * MIN_B[None, :]
    )
    shoot = committor_shoot(
        probes,
        grad,
        in_a,
        in_b,
        n_shots=args.direct_shots,
        seed=2,
        dt=2.0e-4,
        max_steps=20_000,
    )

    q_pde_probe = np.array([_grid_at(pde.xs, pde.ys, pde.q, point) for point in probes])
    rmse_shoot = _rmse(shoot.q_mean, q_pde_probe)

    LOGGER.info("[3/5] Running local-stitch resolution sweep ...")
    convergence = []
    stitch_results = {}
    for resolution in args.resolutions:
        max_steps = args.baseline_max_steps if resolution == 18 else args.max_steps
        LOGGER.info(
            "  grid=%sx%s shots=%s max_steps=%s",
            resolution,
            resolution,
            args.shots,
            max_steps,
        )
        stitch = committor_local_stitch(
            grad,
            in_a,
            in_b,
            xlim=XLIM,
            ylim=YLIM,
            nx=resolution,
            ny=resolution,
            n_shots_per_cell=args.shots,
            seed=1,
            dt=2.0e-4,
            max_steps=max_steps,
        )
        q_stitch_probe = np.array(
            [_grid_at(stitch.xs, stitch.ys, stitch.q, point) for point in probes]
        )
        entry = {
            "grid_size": resolution,
            "n_shots_per_cell": args.shots,
            "max_steps": max_steps,
            "rmse_local_stitch_vs_pde": _rmse(q_stitch_probe, q_pde_probe),
            "isolated_states": int(len(stitch.isolated)),
            "max_abs_residual": float(np.nanmax(np.abs(stitch.residual))),
            "max_markov_total_variation": stitch.markov_check.max_total_variation,
            "max_markov_jensen_shannon": stitch.markov_check.max_jensen_shannon,
            "n_markov_rejected": int(len(stitch.markov_check.rejected_states)),
        }
        convergence.append(entry)
        stitch_results[resolution] = (stitch, q_stitch_probe)

    baseline_resolution = args.resolutions[0]
    baseline_stitch, q_baseline = stitch_results[baseline_resolution]

    LOGGER.info("[4/5] Writing validation report ...")
    report = {
        "probe_fraction_along_AB": fractions.tolist(),
        "q_pde": q_pde_probe.tolist(),
        "q_direct_shoot_mean": shoot.q_mean.tolist(),
        "q_direct_shoot_95CI": list(zip(shoot.q_lo.tolist(), shoot.q_hi.tolist())),
        "q_local_stitch": q_baseline.tolist(),
        "rmse_direct_shoot_vs_pde": rmse_shoot,
        "rmse_local_stitch_vs_pde": convergence[0]["rmse_local_stitch_vs_pde"],
        "local_stitch_isolated_states": convergence[0]["isolated_states"],
        "local_stitch_max_abs_residual": convergence[0]["max_abs_residual"],
        "resolution_convergence": convergence,
    }
    report_path = args.output_dir / "validation_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    LOGGER.info(json.dumps(report, indent=2))

    LOGGER.info("[5/5] Writing plots ...")
    _plot_validation(
        args.output_dir / "committor_validation.png",
        pde,
        baseline_stitch,
        fractions,
        q_pde_probe,
        shoot,
        q_baseline,
        rmse_shoot,
        convergence[0]["rmse_local_stitch_vs_pde"],
    )
    _plot_convergence(
        args.output_dir / "committor_resolution_convergence.png",
        convergence,
    )


def _grid_at(
    xs: np.ndarray,
    ys: np.ndarray,
    values: np.ndarray,
    point: np.ndarray,
) -> float:
    i = int(np.argmin(np.abs(xs - point[0])))
    j = int(np.argmin(np.abs(ys - point[1])))
    return float(values[i, j])


def _rmse(values: np.ndarray, reference: np.ndarray) -> float:
    return float(np.sqrt(np.nanmean((values - reference) ** 2)))


def _plot_validation(
    path: Path,
    pde,
    stitch,
    fractions: np.ndarray,
    q_pde_probe: np.ndarray,
    shoot,
    q_stitch_probe: np.ndarray,
    rmse_shoot: float,
    rmse_stitch: float,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    im0 = axes[0].contourf(
        pde.xs,
        pde.ys,
        pde.q.T,
        levels=np.linspace(0, 1, 21),
        cmap="RdBu_r",
    )
    axes[0].set_title("PDE reference committor q(x,y)")
    axes[0].scatter(*MIN_A, c="k", marker="x")
    axes[0].annotate("A", MIN_A)
    axes[0].scatter(*MIN_B, c="k", marker="x")
    axes[0].annotate("B", MIN_B)
    fig.colorbar(im0, ax=axes[0])

    im1 = axes[1].contourf(
        stitch.xs,
        stitch.ys,
        stitch.q.T,
        levels=np.linspace(0, 1, 21),
        cmap="RdBu_r",
    )
    axes[1].set_title("Local-stitched committor")
    axes[1].scatter(*MIN_A, c="k", marker="x")
    axes[1].scatter(*MIN_B, c="k", marker="x")
    fig.colorbar(im1, ax=axes[1])

    axes[2].plot(fractions, q_pde_probe, "k-o", label="PDE reference")
    axes[2].errorbar(
        fractions,
        shoot.q_mean,
        yerr=[shoot.q_mean - shoot.q_lo, shoot.q_hi - shoot.q_mean],
        fmt="s",
        label="Direct shooting (95% CI)",
        color="tab:red",
    )
    axes[2].plot(
        fractions,
        q_stitch_probe,
        "^--",
        label="Local stitched",
        color="tab:blue",
    )
    axes[2].set_xlabel("fraction along A->B line")
    axes[2].set_ylabel("committor q")
    axes[2].legend()
    axes[2].set_title(f"RMSE shoot={rmse_shoot:.3f} stitch={rmse_stitch:.3f}")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_convergence(path: Path, convergence: list[dict[str, float]]) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    grids = [entry["grid_size"] for entry in convergence]
    rmses = [entry["rmse_local_stitch_vs_pde"] for entry in convergence]
    ax.plot(grids, rmses, "o-", color="tab:blue")
    ax.set_xlabel("grid size")
    ax.set_ylabel("RMSE vs PDE")
    ax.set_title("Local-stitch resolution convergence")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
