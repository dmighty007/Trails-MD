from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

from trails_md.binning.spatial import BinTable, RegularBinner

from .base import Spawner, SpawnerFactory
from .history import pooled_history_iterations


class DensitySpawner(Spawner):
    """Selects frames from sparsely populated regular grid bins."""

    def __init__(
        self,
        n_bins: list[int] | None = None,
        min_values: list[float] | None = None,
        max_values: list[float] | None = None,
        mode: str = "explore",
        probabilistic: bool = True,
        target: list[float] | None = None,
        recent_window: int = 5,
        history_window: int | None = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.n_bins = n_bins or [30, 30]
        self.min_values = min_values
        self.max_values = max_values
        self.mode = mode
        self.probabilistic = probabilistic
        self.target = target
        self.recent_bins: deque[set[Any]] = deque(maxlen=recent_window)
        # Bounds how far back the candidate pool reaches (None = full history).
        self.history_window = history_window
        # Optional landscape-adaptive binner (set by the orchestrator); None -> grid.
        self.binner = None

    def sample(
        self, points: np.ndarray, top_n: int, history: dict[int, Any] | None = None
    ) -> list[int]:
        points = np.asarray(points, dtype=float)
        cumulative_points = _cumulative_points(points, history, self.history_window)
        if self.binner is not None:
            # Keep the adaptive binner in sync with resolution bumps.
            self.binner.n_bins = np.asarray(self.n_bins, dtype=int)
            table = self.binner.fit(cumulative_points)
        else:
            table = RegularBinner(
                n_bins=self.n_bins,
                min_values=self.min_values,
                max_values=self.max_values,
                target=self.target if self.mode == "target" else None,
            ).fit(cumulative_points)
        selected_rows = (
            self._probabilistic_rows(table, top_n)
            if self.probabilistic
            else self._hard_rows(table, top_n)
        )
        if not self.probabilistic:
            self.recent_bins.append({table.ids[row] for row in selected_rows})
        return _sample_frames(table, selected_rows, rng=self.rng)

    def _probabilistic_rows(self, table: BinTable, top_n: int) -> np.ndarray:
        occupied = table.occupied_indices
        if len(occupied) == 0:
            raise ValueError("Cannot sample density bins: no populated bins found.")
        weights = np.zeros(len(table.ids), dtype=float)
        weights[occupied] = 1.0 / table.populations[occupied]
        if self.mode == "target":
            if table.target_closeness is None:
                raise ValueError("Target density sampling requires a target.")
            weights *= table.target_closeness
        return _weighted_choice(occupied, weights[occupied], top_n, rng=self.rng)

    def _hard_rows(self, table: BinTable, top_n: int) -> np.ndarray:
        occupied = table.occupied_indices
        if len(occupied) == 0:
            raise ValueError("Cannot sample density bins: no populated bins found.")

        recent = set().union(*self.recent_bins) if self.recent_bins else set()
        targeting = self.mode == "target" and table.target_closeness is not None
        selected: list[int] = []
        for population in sorted(np.unique(table.populations[occupied])):
            rows = occupied[table.populations[occupied] == population]
            fresh = [row for row in rows if table.ids[int(row)] not in recent]
            stale = [row for row in rows if table.ids[int(row)] in recent]
            ordered = self._order_group(fresh, table, targeting) + self._order_group(
                stale, table, targeting
            )
            selected.extend(int(row) for row in ordered)
            if len(selected) >= top_n:
                return np.asarray(selected[:top_n], dtype=int)

        repeats = int(np.ceil(top_n / len(selected)))
        return np.tile(np.asarray(selected, dtype=int), repeats)[:top_n]

    def _order_group(
        self, group: list, table: BinTable, targeting: bool
    ) -> list:
        """Order one equal-population group of bins.

        Population is the primary key, so this only decides the order *within* a
        group of equally sparse bins. In explore mode that order carries no
        information and is randomised. In target mode it is the one place the
        target enters a `hard` selection, so it must be honoured: closest to the
        target first, with the shuffle applied *before* the sort so that exact
        ties stay randomly broken rather than biased toward low bin indices.

        Randomising after the sort -- as this did until the AdK campaign -- threw
        the ordering away and left `search_mode: target` statistically
        indistinguishable from `explore` under `spawn_type: hard`.
        """
        if len(group) == 0:
            return []
        shuffled = np.asarray(group)[self.rng.permutation(len(group))]
        if not targeting:
            return list(shuffled)
        closeness = table.target_closeness[shuffled.astype(int)]
        return list(shuffled[np.argsort(-closeness, kind="stable")])

    def state_dict(self) -> dict[str, Any]:
        # Persist the hard-mode recency memory so a resumed run reproduces the
        # bin-avoidance behaviour of an uninterrupted one. Bin ids are tuples;
        # store them as lists (round-tripped back to sets on load).
        state = super().state_dict()
        state["recent_bins"] = [
            [
                list(bin_id) if isinstance(bin_id, (tuple, list)) else bin_id
                for bin_id in bins
            ]
            for bins in self.recent_bins
        ]
        return state

    def load_state_dict(self, state: dict[str, Any]) -> None:
        super().load_state_dict(state)
        if state and state.get("recent_bins") is not None:
            self.recent_bins.clear()
            for bins in state["recent_bins"]:
                self.recent_bins.append(
                    {
                        tuple(bin_id) if isinstance(bin_id, list) else bin_id
                        for bin_id in bins
                    }
                )


def _sample_frames(
    table: BinTable, rows: np.ndarray, rng: np.random.Generator | None = None
) -> list[int]:
    choice_fn = rng.choice if rng is not None else np.random.choice
    return [int(choice_fn(table.populated_data[int(row)])) for row in rows]


def _cumulative_points(
    points: np.ndarray,
    history: dict[int, Any] | None,
    window: int | None = None,
) -> np.ndarray:
    historical = _historical_points(points, history, window)
    if historical.size == 0:
        return points
    return np.vstack([historical, points])


def _historical_points(
    points: np.ndarray,
    history: dict[int, Any] | None,
    window: int | None = None,
) -> np.ndarray:
    empty = np.empty((0, points.shape[1]), dtype=float)
    if not history:
        return empty

    # Single source of truth for which historical iterations enter the pool, so
    # the candidate points here stay index-synchronized with the trajectory and
    # frame-record lists that core.py builds for the same spawn indices.
    projections = []
    for iteration in pooled_history_iterations(history, points.shape[1], window):
        projection = np.asarray(history[iteration]["projection"], dtype=float)
        if projection.ndim == 1:
            projection = projection.reshape(-1, 1)
        projections.append(projection)

    return np.vstack(projections) if projections else empty


def _weighted_choice(
    rows: np.ndarray,
    weights: np.ndarray,
    top_n: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    weights = np.asarray(weights, dtype=float)
    weights = np.where(np.isfinite(weights), weights, 0.0)
    if weights.sum() <= 0:
        weights = np.ones_like(weights, dtype=float)
    weights = weights / weights.sum()
    # replace must be True when fewer than top_n entries have nonzero probability,
    # otherwise np.random.choice raises "fewer non-zero entries in p than size"
    # (common in target mode, where target_closeness zeroes out most bins).
    n_nonzero = int(np.count_nonzero(weights))
    replace = len(rows) < top_n or n_nonzero < top_n
    choice_fn = rng.choice if rng is not None else np.random.choice
    return choice_fn(rows, size=top_n, replace=replace, p=weights).astype(int)


SpawnerFactory.register("density", DensitySpawner)
