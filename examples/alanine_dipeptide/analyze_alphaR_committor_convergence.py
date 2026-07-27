"""Post-hoc target-basin committor convergence for alanine dipeptide."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
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

from trails_md.analysis.committor import solve_forward_committor  # noqa: E402
from trails_md.analysis.local_stitch import committor_local_stitch_from_segments  # noqa: E402
from trails_md.msm import MSMEstimator  # noqa: E402

PHI_PSI_LIMITS = (-np.pi, np.pi)
BASINS = {
    "alphaR": {
        "box": ((-2.4, -0.5), (-1.75, 0.35)),
        "label": "alphaR",
    },
    "alphaL": {
        "box": ((0.35, 1.75), (0.05, 1.75)),
        "label": "alphaL",
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("examples/alanine_dipeptide/runs/alanine_dipeptide_cuda_longer"),
    )
    parser.add_argument("--lagtime", type=int, default=1)
    parser.add_argument("--n-microstates", type=int, default=20)
    parser.add_argument("--source-half-width", type=float, default=0.35)
    parser.add_argument("--probe-count", type=int, default=21)
    parser.add_argument("--start-iteration", type=int, default=1)
    parser.add_argument("--step-iteration", type=int, default=1)
    parser.add_argument("--local-grid-size", type=int, default=24)
    parser.add_argument("--target", choices=sorted(BASINS), default="alphaR")
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    target_name = args.target
    target_box = BASINS[target_name]["box"]
    iteration_segments = load_iteration_segments(run_dir)
    if not iteration_segments:
        raise SystemExit(f"No CV segments found under {run_dir}")

    all_points = np.vstack([seg for segments in iteration_segments for seg in segments])
    source_center = iteration_segments[0][0][0].astype(float)
    target_center = box_center(target_box)
    target_mask = lambda pts: box_mask(pts, target_box)
    probes = make_probe_points(source_center, target_center, args.probe_count)

    rows: list[dict[str, object]] = []
    stitch_rows: list[dict[str, object]] = []
    previous_probe_q: np.ndarray | None = None
    previous_stitch_probe_q: np.ndarray | None = None
    latest = None
    latest_stitch = None
    max_iteration = len(iteration_segments) - 1
    for end_iteration in range(
        args.start_iteration,
        max_iteration + 1,
        args.step_iteration,
    ):
        segments = [
            segment
            for iteration in iteration_segments[: end_iteration + 1]
            for segment in iteration
        ]
        row, latest = fit_and_solve(
            segments,
            source_center,
            probes,
            args.source_half_width,
            args.lagtime,
            args.n_microstates,
            target_name,
            target_box,
            target_mask,
        )
        row["end_iteration"] = end_iteration
        if previous_probe_q is None or row["status"] != "ok":
            row["probe_rmse_from_previous"] = None
        else:
            probe_q = np.asarray(row["probe_q"], dtype=float)
            row["probe_rmse_from_previous"] = float(
                np.sqrt(np.nanmean((probe_q - previous_probe_q) ** 2))
            )
        if row["status"] == "ok":
            previous_probe_q = np.asarray(row["probe_q"], dtype=float)
        rows.append(row)

        stitch_row, latest_stitch = fit_and_solve_local_stitch(
            segments,
            source_center,
            probes,
            args.source_half_width,
            args.local_grid_size,
            target_name,
            target_box,
            target_mask,
        )
        stitch_row["end_iteration"] = end_iteration
        if previous_stitch_probe_q is None or stitch_row["status"] != "ok":
            stitch_row["probe_rmse_from_previous"] = None
        else:
            stitch_probe_q = np.asarray(stitch_row["probe_q"], dtype=float)
            stitch_row["probe_rmse_from_previous"] = float(
                np.sqrt(np.nanmean((stitch_probe_q - previous_stitch_probe_q) ** 2))
            )
        if stitch_row["status"] == "ok":
            previous_stitch_probe_q = np.asarray(stitch_row["probe_q"], dtype=float)
        stitch_rows.append(stitch_row)

    report = {
        "run_dir": str(run_dir),
        "target": target_name,
        "target_box": target_box,
        "source_center": source_center.tolist(),
        "source_half_width": args.source_half_width,
        "lagtime": args.lagtime,
        "n_microstates": args.n_microstates,
        "frames_total": int(all_points.shape[0]),
        "iterations": len(iteration_segments),
        "rows": rows,
        "caveat": (
            "Adaptive exploration data are sparse and biased; use this as an "
            "engineering convergence diagnostic, not a final kinetic committor."
        ),
    }
    stitch_report = {
        "run_dir": str(run_dir),
        "target": target_name,
        "method": "observed_cv_local_first_exit_stitch",
        "target_box": target_box,
        "source_center": source_center.tolist(),
        "source_half_width": args.source_half_width,
        "grid_size": args.local_grid_size,
        "frames_total": int(all_points.shape[0]),
        "iterations": len(iteration_segments),
        "rows": stitch_rows,
        "caveat": (
            "This stitches first exits observed in saved phi/psi CV trajectories. "
            "Sparse, adaptive, and discretized data can leave unsolved grid regions."
        ),
    }
    out_json = run_dir / f"{target_name}_committor_convergence.json"
    out_csv = run_dir / f"{target_name}_committor_convergence.csv"
    out_plot = run_dir / f"{target_name}_committor_convergence.png"
    out_map = run_dir / f"{target_name}_committor_latest.png"
    stitch_json = run_dir / f"{target_name}_local_stitch_convergence.json"
    stitch_csv = run_dir / f"{target_name}_local_stitch_convergence.csv"
    stitch_plot = run_dir / f"{target_name}_local_stitch_convergence.png"
    stitch_map = run_dir / f"{target_name}_local_stitch_latest.png"
    out_json.write_text(
        json.dumps(_json_ready(report), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    stitch_json.write_text(
        json.dumps(_json_ready(stitch_report), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    write_csv(out_csv, rows)
    write_local_stitch_csv(stitch_csv, stitch_rows)
    plot_convergence(
        out_plot,
        rows,
        title=f"{target_name}-target MSM committor convergence",
        target_name=target_name,
    )
    plot_convergence(
        stitch_plot,
        stitch_rows,
        title=f"{target_name}-target local-stitch committor convergence",
        target_name=target_name,
    )
    if latest is not None:
        plot_latest_map(
            out_map,
            all_points,
            source_center,
            latest,
            target_box,
            target_name=target_name,
            title=f"Latest {target_name}-target MSM committor",
        )
    if latest_stitch is not None:
        plot_latest_map(
            stitch_map,
            all_points,
            source_center,
            latest_stitch,
            target_box,
            target_name=target_name,
            title=f"Latest {target_name}-target local-stitched committor",
        )

    print(json.dumps(_json_ready(report), indent=2, allow_nan=False))
    print(f"Wrote {out_json}")
    print(f"Wrote {out_csv}")
    print(f"Wrote {out_plot}")
    if latest is not None:
        print(f"Wrote {out_map}")
    print(f"Wrote {stitch_json}")
    print(f"Wrote {stitch_csv}")
    print(f"Wrote {stitch_plot}")
    if latest_stitch is not None:
        print(f"Wrote {stitch_map}")


def load_iteration_segments(run_dir: Path) -> list[list[np.ndarray]]:
    iterations: list[list[np.ndarray]] = []
    for iter_dir in sorted_iter_dirs(run_dir):
        cv_file = iter_dir / "cvs.npz"
        if not cv_file.exists():
            continue
        with np.load(cv_file) as data:
            key = "cvs" if "cvs" in data.files else data.files[0]
            cvs = np.asarray(data[key], dtype=float)
        xtcs = sorted(iter_dir.glob(f"iteration_{iter_number(iter_dir)}_*.xtc"))
        n_walkers = len(xtcs)
        if n_walkers == 0:
            continue
        if cvs.shape[0] % n_walkers != 0:
            raise ValueError(
                f"{cv_file} has {cvs.shape[0]} frames, not divisible by "
                f"{n_walkers} walkers."
            )
        iterations.append([chunk.copy() for chunk in np.split(cvs, n_walkers)])
    return iterations


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


def sorted_iter_dirs(run_dir: Path) -> list[Path]:
    dirs = [
        path
        for path in run_dir.glob("iter_*")
        if path.is_dir() and path.name.removeprefix("iter_").isdigit()
    ]
    return sorted(dirs, key=iter_number)


def iter_number(path: Path) -> int:
    return int(path.name.removeprefix("iter_"))


def make_probe_points(
    source_center: np.ndarray,
    target_center: np.ndarray,
    count: int,
) -> np.ndarray:
    t = np.linspace(0.0, 1.0, int(count))
    return (1.0 - t)[:, None] * source_center[None, :] + t[:, None] * target_center


def fit_and_solve(
    segments: list[np.ndarray],
    source_center: np.ndarray,
    probes: np.ndarray,
    source_half_width: float,
    lagtime: int,
    n_microstates: int,
    target_name: str,
    target_box: tuple[tuple[float, float], tuple[float, float]],
    target_mask,
) -> tuple[dict[str, object], dict[str, np.ndarray] | None]:
    row: dict[str, object] = {
        "segments": len(segments),
        "frames": int(sum(len(segment) for segment in segments)),
    }
    points = np.vstack(segments)
    row.update(coverage_stats(points, target_name, target_mask))
    try:
        estimator = MSMEstimator(
            lagtime=lagtime,
            n_microstates=n_microstates,
            n_timescales=3,
            seed=42,
        )
        result = estimator.fit(segments)
        centers = np.asarray(result.cluster_centers, dtype=float)
        symbols = np.asarray(result.state_symbols, dtype=int)
        active_centers = centers[symbols]
        basin_a = source_clusters(centers, source_center, source_half_width)
        basin_b = target_clusters(centers, target_mask)
        row["active_states"] = int(result.n_states_active)
        row["vamp2"] = result.vamp2_score
        row["timescales"] = np.asarray(result.timescales, dtype=float).tolist()
        row["basin_a_clusters"] = basin_a.tolist()
        row["basin_b_clusters"] = basin_b.tolist()
        if basin_a.size == 0 or basin_b.size == 0:
            row["status"] = "missing_basin_clusters"
            return row, None
        committor = solve_forward_committor(
            result.transition_matrix,
            basin_a,
            basin_b,
            state_symbols=symbols,
            basin_id_space="full",
        )
        probe_q = nearest_committor(active_centers, committor.q, probes)
        row["status"] = "ok"
        row["q_min"] = float(np.nanmin(committor.q))
        row["q_max"] = float(np.nanmax(committor.q))
        row["max_abs_residual"] = float(np.nanmax(np.abs(committor.residual)))
        row["probe_q"] = probe_q.tolist()
        latest = {
            "active_centers": active_centers,
            "q": committor.q,
            "basin_a_centers": centers[basin_a],
            "basin_b_centers": centers[basin_b],
        }
        return row, latest
    except Exception as exc:  # noqa: BLE001 - report per-checkpoint failures
        row["status"] = f"failed: {exc}"
        return row, None


def fit_and_solve_local_stitch(
    segments: list[np.ndarray],
    source_center: np.ndarray,
    probes: np.ndarray,
    source_half_width: float,
    grid_size: int,
    target_name: str,
    target_box: tuple[tuple[float, float], tuple[float, float]],
    target_mask,
) -> tuple[dict[str, object], dict[str, np.ndarray] | None]:
    row: dict[str, object] = {
        "segments": len(segments),
        "frames": int(sum(len(segment) for segment in segments)),
    }
    points = np.vstack(segments)
    row.update(coverage_stats(points, target_name, target_mask))
    source = lambda pts: source_mask(pts, source_center, source_half_width)
    try:
        stitch = committor_local_stitch_from_segments(
            segments,
            source,
            target_mask,
            xlim=PHI_PSI_LIMITS,
            ylim=PHI_PSI_LIMITS,
            nx=grid_size,
            ny=grid_size,
        )
        centers = np.array(
            [[x, y] for x in stitch.xs for y in stitch.ys],
            dtype=float,
        )
        q = stitch.q.ravel()
        finite = np.isfinite(q)
        probe_q = nearest_grid_committor(stitch.xs, stitch.ys, stitch.q, probes)
        row["status"] = "ok"
        row["grid_size"] = int(grid_size)
        row["first_exit_counts"] = int(stitch.counts.sum())
        row["observed_interior_states"] = int((stitch.counts.sum(axis=1) > 0).sum())
        row["isolated_states"] = int(len(stitch.isolated))
        row["finite_q_states"] = int(finite.sum())
        row["q_min"] = float(np.nanmin(q)) if finite.any() else None
        row["q_max"] = float(np.nanmax(q)) if finite.any() else None
        row["max_abs_residual"] = float(np.nanmax(np.abs(stitch.residual)))
        row["max_markov_total_variation"] = stitch.markov_check.max_total_variation
        row["max_markov_jensen_shannon"] = stitch.markov_check.max_jensen_shannon
        row["n_markov_rejected"] = int(len(stitch.markov_check.rejected_states))
        row["probe_q"] = probe_q.tolist()
        latest = {
            "active_centers": centers[finite],
            "q": q[finite],
            "basin_a_centers": centers[stitch.is_a.ravel()],
            "basin_b_centers": centers[stitch.is_b.ravel()],
        }
        return row, latest
    except Exception as exc:  # noqa: BLE001 - report per-checkpoint failures
        row["status"] = f"failed: {exc}"
        return row, None


def coverage_stats(points: np.ndarray, target_name: str, target_mask) -> dict[str, object]:
    hist, _, _ = np.histogram2d(
        points[:, 0],
        points[:, 1],
        bins=[np.linspace(-np.pi, np.pi, 25), np.linspace(-np.pi, np.pi, 25)],
    )
    return {
        "phi_range": [float(points[:, 0].min()), float(points[:, 0].max())],
        "psi_range": [float(points[:, 1].min()), float(points[:, 1].max())],
        "occupied_24x24_bins": int((hist > 0).sum()),
        "target_frame_hits": int(target_mask(points).sum()),
        f"{target_name}_frame_hits": int(target_mask(points).sum()),
    }


def source_clusters(
    centers: np.ndarray,
    source_center: np.ndarray,
    half_width: float,
) -> np.ndarray:
    delta = np.abs(centers[:, :2] - source_center[:2])
    return np.flatnonzero((delta[:, 0] <= half_width) & (delta[:, 1] <= half_width))


def target_clusters(centers: np.ndarray, target_mask) -> np.ndarray:
    return np.flatnonzero(target_mask(centers))


def box_center(box: tuple[tuple[float, float], tuple[float, float]]) -> np.ndarray:
    return np.array(
        [
            0.5 * (box[0][0] + box[0][1]),
            0.5 * (box[1][0] + box[1][1]),
        ],
        dtype=float,
    )


def box_mask(
    points: np.ndarray,
    box: tuple[tuple[float, float], tuple[float, float]],
) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    single = pts.ndim == 1
    rows = pts.reshape(1, -1) if single else pts
    mask = (
        (rows[:, 0] >= box[0][0])
        & (rows[:, 0] <= box[0][1])
        & (rows[:, 1] >= box[1][0])
        & (rows[:, 1] <= box[1][1])
    )
    return mask[0] if single else mask


def source_mask(
    points: np.ndarray,
    source_center: np.ndarray,
    half_width: float,
) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    single = pts.ndim == 1
    rows = pts.reshape(1, -1) if single else pts
    delta = np.abs(rows[:, :2] - source_center[:2])
    mask = (delta[:, 0] <= half_width) & (delta[:, 1] <= half_width)
    return mask[0] if single else mask


def nearest_committor(
    centers: np.ndarray,
    q: np.ndarray,
    probes: np.ndarray,
) -> np.ndarray:
    values = np.zeros(len(probes), dtype=float)
    for idx, point in enumerate(probes):
        nearest = int(np.argmin(np.linalg.norm(centers[:, :2] - point[:2], axis=1)))
        values[idx] = q[nearest]
    return values


def nearest_grid_committor(
    xs: np.ndarray,
    ys: np.ndarray,
    q: np.ndarray,
    probes: np.ndarray,
) -> np.ndarray:
    values = np.full(len(probes), np.nan, dtype=float)
    for idx, point in enumerate(probes):
        i = int(np.argmin(np.abs(xs - point[0])))
        j = int(np.argmin(np.abs(ys - point[1])))
        values[idx] = q[i, j]
    return values


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = [
        "end_iteration",
        "frames",
        "segments",
        "status",
        "active_states",
        "occupied_24x24_bins",
        "target_frame_hits",
        "vamp2",
        "probe_rmse_from_previous",
        "max_abs_residual",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def write_local_stitch_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = [
        "end_iteration",
        "frames",
        "segments",
        "status",
        "grid_size",
        "occupied_24x24_bins",
        "target_frame_hits",
        "first_exit_counts",
        "observed_interior_states",
        "finite_q_states",
        "isolated_states",
        "probe_rmse_from_previous",
        "max_abs_residual",
        "max_markov_total_variation",
        "n_markov_rejected",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def plot_convergence(
    path: Path,
    rows: list[dict[str, object]],
    *,
    title: str,
    target_name: str,
) -> None:
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    if not ok_rows:
        return
    iterations = [int(row["end_iteration"]) for row in ok_rows]
    rmse = [
        np.nan if row.get("probe_rmse_from_previous") is None else row[
            "probe_rmse_from_previous"
        ]
        for row in ok_rows
    ]
    hits = [int(row["target_frame_hits"]) for row in ok_rows]
    occupied = [int(row["occupied_24x24_bins"]) for row in ok_rows]

    fig, axes = plt.subplots(3, 1, figsize=(7, 8), sharex=True)
    axes[0].plot(iterations, rmse, "o-", color="tab:blue")
    axes[0].set_ylabel("probe q RMSE\nvs previous")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(iterations, hits, "o-", color="tab:red")
    axes[1].set_ylabel(f"{target_name} frame hits")
    axes[1].grid(True, alpha=0.3)
    axes[2].plot(iterations, occupied, "o-", color="tab:green")
    axes[2].set_ylabel("occupied bins")
    axes[2].set_xlabel("cumulative end iteration")
    axes[2].grid(True, alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_latest_map(
    path: Path,
    points: np.ndarray,
    source_center: np.ndarray,
    latest: dict[str, np.ndarray],
    target_box: tuple[tuple[float, float], tuple[float, float]],
    *,
    target_name: str,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(
        points[:, 0],
        points[:, 1],
        s=10,
        c="#aeb8c2",
        alpha=0.55,
        label="saved CV frames",
    )
    scatter = ax.scatter(
        latest["active_centers"][:, 0],
        latest["active_centers"][:, 1],
        c=latest["q"],
        s=90,
        cmap="RdBu_r",
        vmin=0.0,
        vmax=1.0,
        edgecolor="black",
        linewidth=0.5,
        label="active centers",
    )
    ax.scatter(
        latest["basin_a_centers"][:, 0],
        latest["basin_a_centers"][:, 1],
        marker="x",
        s=140,
        c="tab:blue",
        linewidth=3,
        label="A source",
    )
    ax.scatter(
        latest["basin_b_centers"][:, 0],
        latest["basin_b_centers"][:, 1],
        marker="x",
        s=140,
        c="tab:red",
        linewidth=3,
        label=f"B {target_name}",
    )
    ax.add_patch(
        plt.Rectangle(
            (source_center[0] - 0.35, source_center[1] - 0.35),
            0.70,
            0.70,
            fill=False,
            edgecolor="tab:blue",
            linestyle="--",
            linewidth=1.4,
        )
    )
    ax.add_patch(
        plt.Rectangle(
            (target_box[0][0], target_box[1][0]),
            target_box[0][1] - target_box[0][0],
            target_box[1][1] - target_box[1][0],
            fill=False,
            edgecolor="tab:red",
            linestyle="--",
            linewidth=1.4,
        )
    )
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-np.pi, np.pi)
    ax.set_xlabel("phi (rad)")
    ax.set_ylabel("psi (rad)")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    fig.colorbar(scatter, ax=ax, label=f"q({target_name} before source)")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
