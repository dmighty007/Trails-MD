"""Diagnose when AlaD exploration should hand off to committor-guided spawning."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join(tempfile.gettempdir(), "trails-md-matplotlib"),
)

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from trails_md.analysis.committor_acquisition import (  # noqa: E402
    CommittorAcquisitionWeights,
    score_local_stitch_cells,
)
from trails_md.analysis.local_stitch import (  # noqa: E402
    committor_local_stitch_from_segments,
)

PHI_PSI_LIMITS = (-np.pi, np.pi)
ALPHAL_BOX = ((0.35, 1.75), (0.05, 1.75))
SOURCE_HALF_WIDTH = 0.35


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("examples/AlaD/runs/alad_alphaL_explore_36x36"),
    )
    parser.add_argument("--local-grid-size", type=int, default=36)
    parser.add_argument("--min-iteration", type=int, default=100)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min-target-hits", type=int, default=1000)
    parser.add_argument("--max-new-observed", type=int, default=2)
    parser.add_argument("--max-new-finite", type=int, default=2)
    parser.add_argument("--max-isolated-drop", type=int, default=2)
    parser.add_argument("--max-new-occupied", type=int, default=2)
    parser.add_argument("--top-cells", type=int, default=20)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    stitch_report = json.loads(
        (run_dir / "alphaL_local_stitch_convergence.json").read_text(
            encoding="utf-8"
        )
    )
    output_rows = load_output_log(run_dir / "output.log")
    output_by_iter = {row["iteration"]: row for row in output_rows}

    gate_rows = []
    for row in stitch_report["rows"]:
        end_iteration = int(row["end_iteration"])
        output_row = output_by_iter.get(end_iteration)
        gate_rows.append(
            {
                "end_iteration": end_iteration,
                "target_hits": int(row.get("target_frame_hits", 0)),
                "observed_interior_states": int(
                    row.get("observed_interior_states", 0)
                ),
                "finite_q_states": int(row.get("finite_q_states", 0)),
                "isolated_states": int(row.get("isolated_states", 0)),
                "occupied_bins": None
                if output_row is None
                else int(output_row["occupied_bins"]),
                "total_bins": None if output_row is None else int(output_row["total_bins"]),
                "frames": int(row.get("frames", 0)),
            }
        )

    switch_iteration = find_handoff_iteration(
        gate_rows,
        min_iteration=args.min_iteration,
        patience=args.patience,
        min_target_hits=args.min_target_hits,
        max_new_observed=args.max_new_observed,
        max_new_finite=args.max_new_finite,
        max_isolated_drop=args.max_isolated_drop,
        max_new_occupied=args.max_new_occupied,
    )

    segments_by_iter = load_iteration_segments(run_dir)
    source_center = np.asarray(segments_by_iter[0][0][0], dtype=float)
    segments = [segment for iteration in segments_by_iter for segment in iteration]
    acquisition_summary = final_acquisition_summary(
        segments,
        source_center,
        grid_size=args.local_grid_size,
        top_cells=args.top_cells,
    )

    report = {
        "run_dir": str(run_dir),
        "policy": (
            "Do not switch on first alphaL hit. Switch only after target support "
            "exists and exploration/local-stitch support growth has plateaued."
        ),
        "gate": {
            "min_iteration": args.min_iteration,
            "patience": args.patience,
            "min_target_hits": args.min_target_hits,
            "max_new_observed": args.max_new_observed,
            "max_new_finite": args.max_new_finite,
            "max_isolated_drop": args.max_isolated_drop,
            "max_new_occupied": args.max_new_occupied,
        },
        "switch_iteration": switch_iteration,
        "first_alphaL_iteration": next(
            (row["end_iteration"] for row in gate_rows if row["target_hits"] > 0),
            None,
        ),
        "final_gate_row": gate_rows[-1],
        "gate_rows": gate_rows,
        "final_acquisition": acquisition_summary,
    }

    output_json = args.output_json or (
        run_dir / "alphaL_committor_guided_handoff.json"
    )
    output_json.write_text(json.dumps(_json_ready(report), indent=2), encoding="utf-8")
    print(json.dumps(_json_ready({k: report[k] for k in (
        "first_alphaL_iteration",
        "switch_iteration",
        "final_gate_row",
        "gate",
    )}), indent=2))
    print(f"Wrote {output_json}")


def find_handoff_iteration(
    rows: list[dict[str, int | None]],
    *,
    min_iteration: int,
    patience: int,
    min_target_hits: int,
    max_new_observed: int,
    max_new_finite: int,
    max_isolated_drop: int,
    max_new_occupied: int,
) -> int | None:
    for idx, row in enumerate(rows):
        end_iteration = int(row["end_iteration"])
        if end_iteration < min_iteration:
            continue
        if int(row["target_hits"]) < min_target_hits:
            continue
        if idx < patience:
            continue
        window = rows[idx - patience : idx + 1]
        if _window_plateaued(
            window,
            max_new_observed=max_new_observed,
            max_new_finite=max_new_finite,
            max_isolated_drop=max_isolated_drop,
            max_new_occupied=max_new_occupied,
        ):
            return end_iteration + 1
    return None


def _window_plateaued(
    rows: list[dict[str, int | None]],
    *,
    max_new_observed: int,
    max_new_finite: int,
    max_isolated_drop: int,
    max_new_occupied: int,
) -> bool:
    total_bins = {row["total_bins"] for row in rows if row["total_bins"] is not None}
    if len(total_bins) > 1:
        return False
    for before, after in zip(rows[:-1], rows[1:]):
        if (
            int(after["observed_interior_states"])
            - int(before["observed_interior_states"])
            > max_new_observed
        ):
            return False
        if int(after["finite_q_states"]) - int(before["finite_q_states"]) > max_new_finite:
            return False
        if int(before["isolated_states"]) - int(after["isolated_states"]) > max_isolated_drop:
            return False
        if before["occupied_bins"] is not None and after["occupied_bins"] is not None:
            if int(after["occupied_bins"]) - int(before["occupied_bins"]) > max_new_occupied:
                return False
    return True


def final_acquisition_summary(
    segments: list[np.ndarray],
    source_center: np.ndarray,
    *,
    grid_size: int,
    top_cells: int,
) -> dict[str, object]:
    source = lambda pts: source_mask(pts, source_center, SOURCE_HALF_WIDTH)
    target = lambda pts: box_mask(pts, ALPHAL_BOX)
    stitch = committor_local_stitch_from_segments(
        segments,
        source,
        target,
        xlim=PHI_PSI_LIMITS,
        ylim=PHI_PSI_LIMITS,
        nx=grid_size,
        ny=grid_size,
    )
    acquisition = score_local_stitch_cells(
        stitch,
        weights=CommittorAcquisitionWeights(),
        markov_min_count=8,
    )
    q = stitch.q
    counts = stitch.counts.sum(axis=1).reshape(q.shape)
    rows = []
    flat_order = np.argsort(acquisition.score.ravel())[::-1]
    for flat in flat_order:
        i, j = divmod(int(flat), grid_size)
        score = float(acquisition.score[i, j])
        if score <= 0:
            break
        rows.append(
            {
                "rank": len(rows) + 1,
                "cell": [i, j],
                "center": [float(stitch.xs[i]), float(stitch.ys[j])],
                "score": score,
                "q": None if not np.isfinite(q[i, j]) else float(q[i, j]),
                "first_exit_count": int(counts[i, j]),
                "transition_region": float(acquisition.transition_region[i, j]),
                "undersampling": float(acquisition.undersampling[i, j]),
                "frontier_bonus": float(acquisition.frontier_bonus[i, j]),
                "markov_pressure": float(acquisition.markov_pressure[i, j]),
            }
        )
        if len(rows) >= top_cells:
            break
    finite = np.isfinite(q)
    return {
        "grid_size": grid_size,
        "finite_q_states": int(finite.sum()),
        "isolated_states": int(len(stitch.isolated)),
        "observed_interior_states": int((stitch.counts.sum(axis=1) > 0).sum()),
        "top_cells": rows,
    }


def load_output_log(path: Path) -> list[dict[str, int]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip() or line.startswith("#") or line.startswith("iteration"):
                continue
            parts = line.rstrip("\n").split("\t")
            rows.append(
                {
                    "iteration": int(parts[0]),
                    "cumulative_frames": int(parts[7]),
                    "occupied_bins": int(parts[8]),
                    "total_bins": int(parts[9]),
                }
            )
    return rows


def load_iteration_segments(run_dir: Path) -> list[list[np.ndarray]]:
    iterations: list[list[np.ndarray]] = []
    for iter_dir in sorted(
        run_dir.glob("iter_*"),
        key=lambda path: int(path.name.removeprefix("iter_")),
    ):
        cv_file = iter_dir / "cvs.npz"
        if not cv_file.exists():
            continue
        cvs = np.load(cv_file)["cvs"][:, :2]
        iteration = int(iter_dir.name.removeprefix("iter_"))
        xtcs = sorted(iter_dir.glob(f"iteration_{iteration}_*.xtc"))
        if not xtcs:
            continue
        iterations.append([chunk.copy() for chunk in np.split(cvs, len(xtcs))])
    return iterations


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
