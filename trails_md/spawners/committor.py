from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from trails_md.analysis.committor_acquisition import (
    CommittorAcquisitionWeights,
    score_local_stitch_cells,
)
from trails_md.analysis.local_stitch import committor_local_stitch_from_segments

from .base import Spawner, SpawnerFactory
from .density import DensitySpawner, _weighted_choice
from .history import pooled_history_iterations


@dataclass(frozen=True)
class _GateMetrics:
    iteration: int
    observed_states: int
    finite_q_states: int
    isolated_states: int
    target_hits: int


class CommittorSpawner(Spawner):
    """Select frames by observed local-stitch committor uncertainty.

    The spawner remains in density-exploration mode until a simple support gate
    says the first exploration phase has plateaued. After that it rebuilds the
    local-stitch committor from completed short unbiased bursts and prioritizes
    observed cells near q=0.5, sparse cells, frontier cells, and locally
    non-Markovian cells.
    """

    def __init__(
        self,
        n_bins: list[int] | None = None,
        min_values: list[float] | None = None,
        max_values: list[float] | None = None,
        mode: str = "explore",
        probabilistic: bool = False,
        target: list[float] | None = None,
        recent_window: int = 5,
        committor_grid_size: int = 36,
        committor_source: list[list[float]] | None = None,
        committor_source_half_width: float = 0.35,
        committor_target: list[list[float]] | None = None,
        committor_min_iteration: int = 100,
        committor_patience: int = 5,
        committor_min_target_hits: int = 1000,
        committor_max_new_observed: int = 2,
        committor_max_new_finite: int = 2,
        committor_max_isolated_drop: int = 2,
        committor_min_count: int = 1,
        committor_markov_min_count: int = 8,
        committor_weight_transition: float = 1.0,
        committor_weight_undersampling: float = 0.5,
        committor_weight_frontier: float = 0.5,
        committor_weight_markov: float = 0.25,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.n_bins = n_bins or [committor_grid_size, committor_grid_size]
        self.min_values = min_values
        self.max_values = max_values
        self.mode = mode
        self.probabilistic = probabilistic
        self.target = target
        self.source_box = _as_box(committor_source)
        self.source_half_width = float(committor_source_half_width)
        self.target_box = _as_box(committor_target)
        self.min_iteration = int(committor_min_iteration)
        self.patience = int(committor_patience)
        self.min_target_hits = int(committor_min_target_hits)
        self.max_new_observed = int(committor_max_new_observed)
        self.max_new_finite = int(committor_max_new_finite)
        self.max_isolated_drop = int(committor_max_isolated_drop)
        self.min_count = int(committor_min_count)
        self.markov_min_count = int(committor_markov_min_count)
        self.weights = CommittorAcquisitionWeights(
            transition=float(committor_weight_transition),
            undersampling=float(committor_weight_undersampling),
            frontier=float(committor_weight_frontier),
            markov=float(committor_weight_markov),
        )
        self.committor_enabled = False
        self.switch_iteration: int | None = None
        self.metric_history: list[dict[str, int]] = []
        self.last_metrics: dict[str, int] | None = None

        self._fallback = DensitySpawner(
            n_bins=n_bins,
            min_values=min_values,
            max_values=max_values,
            mode=mode,
            probabilistic=probabilistic,
            target=target,
            recent_window=recent_window,
            seed=self.seed,
        )

    def sample(
        self, points: np.ndarray, top_n: int, history: dict[int, Any] | None = None
    ) -> list[int]:
        points = np.asarray(points, dtype=float)
        if points.ndim == 1:
            points = points.reshape(-1, 1)
        if points.shape[1] < 2:
            return self._fallback.sample(points, top_n, history=history)

        cumulative, segments = _pooled_points_and_segments(points, history, top_n)
        if cumulative.size == 0 or not segments:
            return self._fallback.sample(points, top_n, history=history)

        source_box = self.source_box or _default_source_box(
            cumulative[0, :2], self.source_half_width
        )
        target_box = self.target_box or _target_box_from_center(
            self.target, self.source_half_width
        )
        if target_box is None or self.min_values is None or self.max_values is None:
            return self._fallback.sample(points, top_n, history=history)

        xlim = (float(self.min_values[0]), float(self.max_values[0]))
        ylim = (float(self.min_values[1]), float(self.max_values[1]))
        nx, ny = _grid_shape(self.n_bins, default=int(self.n_bins[0]))

        try:
            stitch = committor_local_stitch_from_segments(
                segments,
                lambda pts: _box_mask(pts, source_box),
                lambda pts: _box_mask(pts, target_box),
                xlim=xlim,
                ylim=ylim,
                nx=nx,
                ny=ny,
            )
        except (ValueError, np.linalg.LinAlgError) as exc:
            logging.warning("Committor spawner falling back to density: %s", exc)
            return self._fallback.sample(points, top_n, history=history)

        iteration = _current_iteration(history)
        metrics = _metrics(stitch, cumulative[:, :2], target_box, iteration)
        self.last_metrics = asdict(metrics)
        self._record_metrics(metrics)
        if not self.committor_enabled and self._gate_is_open():
            self.committor_enabled = True
            self.switch_iteration = iteration
            logging.info(
                "Committor spawning enabled at iteration %s "
                "(target_hits=%s, observed=%s, finite_q=%s, isolated=%s).",
                iteration,
                metrics.target_hits,
                metrics.observed_states,
                metrics.finite_q_states,
                metrics.isolated_states,
            )

        if not self.committor_enabled:
            return self._fallback.sample(points, top_n, history=history)

        acquisition = score_local_stitch_cells(
            stitch,
            weights=self.weights,
            min_count=self.min_count,
            markov_min_count=self.markov_min_count,
        )
        selected = _select_frames_from_scores(
            cumulative[:, :2],
            acquisition.score,
            xlim=xlim,
            ylim=ylim,
            probabilistic=self.probabilistic,
            top_n=top_n,
            rng=self.rng,
        )
        if not selected:
            return self._fallback.sample(points, top_n, history=history)
        return selected

    def state_dict(self) -> dict[str, Any]:
        state = super().state_dict()
        state.update(
            {
                "committor_enabled": self.committor_enabled,
                "switch_iteration": self.switch_iteration,
                "metric_history": list(self.metric_history),
                "last_metrics": self.last_metrics,
                "fallback": self._fallback.state_dict(),
            }
        )
        return state

    def load_state_dict(self, state: dict[str, Any]) -> None:
        super().load_state_dict(state)
        if not state:
            return
        self.committor_enabled = bool(state.get("committor_enabled", False))
        self.switch_iteration = state.get("switch_iteration")
        self.metric_history = [
            {k: int(v) for k, v in row.items()}
            for row in state.get("metric_history", [])
        ]
        self.last_metrics = state.get("last_metrics")
        # If a committor run resumes from a density checkpoint, reuse the density
        # spawner's persisted recency/RNG state as the exploration fallback.
        self._fallback.load_state_dict(state.get("fallback", state))

    def _record_metrics(self, metrics: _GateMetrics) -> None:
        row = asdict(metrics)
        if (
            self.metric_history
            and self.metric_history[-1]["iteration"] == row["iteration"]
        ):
            self.metric_history[-1] = row
        else:
            self.metric_history.append(row)
        max_rows = max(self.patience + 2, 16)
        if len(self.metric_history) > max_rows:
            self.metric_history = self.metric_history[-max_rows:]

    def _gate_is_open(self) -> bool:
        if not self.metric_history:
            return False
        latest = self.metric_history[-1]
        if latest["iteration"] < self.min_iteration:
            return False
        if latest["target_hits"] < self.min_target_hits:
            return False
        if self.patience <= 0:
            return True
        if len(self.metric_history) < self.patience + 1:
            return False
        window = self.metric_history[-(self.patience + 1) :]
        for before, after in zip(window[:-1], window[1:]):
            if (
                after["observed_states"] - before["observed_states"]
                > self.max_new_observed
            ):
                return False
            if (
                after["finite_q_states"] - before["finite_q_states"]
                > self.max_new_finite
            ):
                return False
            isolated_drop = before["isolated_states"] - after["isolated_states"]
            if isolated_drop > self.max_isolated_drop:
                return False
        return True


def _pooled_points_and_segments(
    points: np.ndarray, history: dict[int, Any] | None, current_walkers: int
) -> tuple[np.ndarray, list[np.ndarray]]:
    arrays: list[np.ndarray] = []
    segments: list[np.ndarray] = []
    target_dim = points.shape[1]
    for iteration in pooled_history_iterations(history, target_dim):
        projection = np.asarray(history[iteration]["projection"], dtype=float)
        if projection.ndim == 1:
            projection = projection.reshape(-1, 1)
        arrays.append(projection)
        segments.extend(
            _split_projection(projection, history[iteration].get("trajectories"))
        )
    arrays.append(points)
    segments.extend(_split_projection(points, [None] * int(current_walkers)))
    return np.vstack(arrays), [segment for segment in segments if len(segment) > 1]


def _split_projection(projection: np.ndarray, trajectories: Any) -> list[np.ndarray]:
    projection = np.asarray(projection, dtype=float)
    if projection.ndim == 1:
        projection = projection.reshape(-1, 1)
    n_traj = len(trajectories) if trajectories else 0
    if n_traj <= 0 or projection.shape[0] % n_traj != 0:
        return [projection]
    frames = projection.shape[0] // n_traj
    if frames <= 0:
        return [projection]
    return [projection[i * frames : (i + 1) * frames] for i in range(n_traj)]


def _current_iteration(history: dict[int, Any] | None) -> int:
    if not history:
        return 0
    nonnegative = [int(i) for i in history if int(i) >= 0]
    return (max(nonnegative) + 1) if nonnegative else 0


def _metrics(
    stitch: Any,
    points: np.ndarray,
    target_box: tuple[tuple[float, float], tuple[float, float]],
    iteration: int,
) -> _GateMetrics:
    q = np.asarray(stitch.q, dtype=float)
    counts = np.asarray(stitch.counts.sum(axis=1), dtype=float)
    boundary = np.asarray(stitch.is_a | stitch.is_b, dtype=bool).ravel()
    return _GateMetrics(
        iteration=iteration,
        observed_states=int(np.count_nonzero((counts > 0.0) & (~boundary))),
        finite_q_states=int(
            np.count_nonzero(np.isfinite(q) & (~stitch.is_a) & (~stitch.is_b))
        ),
        isolated_states=int(np.asarray(stitch.isolated).size),
        target_hits=int(np.count_nonzero(_box_mask(points, target_box))),
    )


def _select_frames_from_scores(
    points: np.ndarray,
    score: np.ndarray,
    *,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    probabilistic: bool,
    top_n: int,
    rng: np.random.Generator,
) -> list[int]:
    nx, ny = score.shape
    dx = (xlim[1] - xlim[0]) / float(nx - 1)
    dy = (ylim[1] - ylim[0]) / float(ny - 1)
    ix = np.clip(np.rint((points[:, 0] - xlim[0]) / dx).astype(int), 0, nx - 1)
    iy = np.clip(np.rint((points[:, 1] - ylim[0]) / dy).astype(int), 0, ny - 1)
    point_score = score[ix, iy]
    candidate_indices = np.flatnonzero(point_score > 0.0)
    if candidate_indices.size == 0:
        return []
    if probabilistic:
        return list(
            map(
                int,
                _weighted_choice(
                    candidate_indices,
                    point_score[candidate_indices],
                    top_n,
                    rng=rng,
                ),
            )
        )

    cell_to_indices: dict[tuple[int, int], list[int]] = {}
    for index in candidate_indices:
        cell_to_indices.setdefault((int(ix[index]), int(iy[index])), []).append(
            int(index)
        )
    ranked_cells = sorted(
        cell_to_indices,
        key=lambda cell: (score[cell], -len(cell_to_indices[cell])),
        reverse=True,
    )
    picks: list[int] = []
    for cell in ranked_cells:
        frames = cell_to_indices[cell]
        picks.append(int(rng.choice(frames)))
        if len(picks) >= top_n:
            return picks
    while len(picks) < top_n:
        cell = ranked_cells[len(picks) % len(ranked_cells)]
        picks.append(int(rng.choice(cell_to_indices[cell])))
    return picks


def _as_box(
    value: list[list[float]] | None,
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    if value is None:
        return None
    if len(value) < 2:
        raise ValueError("committor basin boxes must include at least two dimensions.")
    box = (
        (float(value[0][0]), float(value[0][1])),
        (float(value[1][0]), float(value[1][1])),
    )
    if box[0][0] >= box[0][1] or box[1][0] >= box[1][1]:
        raise ValueError(f"Invalid committor basin box: {value}")
    return box


def _default_source_box(
    center: np.ndarray, half_width: float
) -> tuple[tuple[float, float], tuple[float, float]]:
    return (
        (float(center[0] - half_width), float(center[0] + half_width)),
        (float(center[1] - half_width), float(center[1] + half_width)),
    )


def _target_box_from_center(
    target: list[float] | None, half_width: float
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    if target is None or len(target) < 2:
        return None
    center = np.asarray(target[:2], dtype=float)
    return _default_source_box(center, half_width)


def _box_mask(
    points: np.ndarray,
    box: tuple[tuple[float, float], tuple[float, float]],
) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    return (
        (pts[..., 0] >= box[0][0])
        & (pts[..., 0] <= box[0][1])
        & (pts[..., 1] >= box[1][0])
        & (pts[..., 1] <= box[1][1])
    )


def _grid_shape(n_bins: list[int] | None, *, default: int) -> tuple[int, int]:
    if n_bins is None:
        return default, default
    if len(n_bins) == 1:
        return int(n_bins[0]), int(n_bins[0])
    return int(n_bins[0]), int(n_bins[1])


SpawnerFactory.register("committor", CommittorSpawner)
