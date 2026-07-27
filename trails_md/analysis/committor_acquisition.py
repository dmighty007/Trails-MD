"""Acquisition scores for committor-guided adaptive sampling."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .local_stitch import LocalStitchResult


@dataclass(frozen=True)
class CommittorAcquisitionWeights:
    """Weights for ranking cells in committor-guided sampling."""

    transition: float = 1.0
    undersampling: float = 0.5
    frontier: float = 0.5
    markov: float = 0.25


@dataclass(frozen=True)
class CommittorAcquisition:
    """Per-cell acquisition terms and total score."""

    score: np.ndarray
    transition_region: np.ndarray
    undersampling: np.ndarray
    frontier_bonus: np.ndarray
    markov_pressure: np.ndarray
    eligible: np.ndarray


def score_local_stitch_cells(
    stitch: LocalStitchResult,
    *,
    weights: CommittorAcquisitionWeights | None = None,
    min_count: int = 0,
    markov_min_count: int = 8,
) -> CommittorAcquisition:
    """Score grid cells for additional local first-exit sampling.

    The score is an acquisition function, not a committor estimator. It ranks
    cells that should receive more short unbiased bursts because they are near
    the transition region, under-sampled, adjacent to unresolved support, or
    locally non-Markovian across trajectory subsets.
    """

    w = weights or CommittorAcquisitionWeights()
    q = np.asarray(stitch.q, dtype=float)
    counts = np.asarray(stitch.counts[:, : q.size].sum(axis=1), dtype=float).reshape(
        q.shape
    )
    total_counts = np.asarray(stitch.counts.sum(axis=1), dtype=float).reshape(q.shape)
    is_boundary = np.asarray(stitch.is_a | stitch.is_b, dtype=bool)
    finite = np.isfinite(q)
    observed = total_counts > 0
    unresolved = (~finite) & (~is_boundary)
    eligible = (~is_boundary) & (total_counts >= float(min_count))

    transition_region = np.zeros_like(q, dtype=float)
    transition_region[finite] = np.clip(q[finite] * (1.0 - q[finite]), 0.0, 0.25)
    transition_region *= 4.0

    frontier_bonus = _frontier_bonus(observed | finite | is_boundary, unresolved)
    frontier_bonus[is_boundary] = 0.0

    undersampling = 1.0 / np.sqrt(total_counts + 1.0)
    undersampling[~(observed | finite | (frontier_bonus > 0.0))] = 0.0
    undersampling[is_boundary] = 0.0

    markov_pressure = _normalize_finite(
        np.maximum(
            np.asarray(stitch.markov_total_variation, dtype=float),
            np.asarray(stitch.markov_jensen_shannon, dtype=float),
        )
    )
    markov_pressure[~np.isfinite(markov_pressure)] = 0.0
    markov_pressure[total_counts < float(markov_min_count)] = 0.0
    markov_pressure[is_boundary] = 0.0

    score = (
        w.transition * transition_region
        + w.undersampling * undersampling
        + w.frontier * frontier_bonus
        + w.markov * markov_pressure
    )
    score[~eligible] = 0.0
    score[~np.isfinite(score)] = 0.0

    return CommittorAcquisition(
        score=score,
        transition_region=transition_region,
        undersampling=undersampling,
        frontier_bonus=frontier_bonus,
        markov_pressure=markov_pressure,
        eligible=eligible,
    )


def _frontier_bonus(observed_or_boundary: np.ndarray, unresolved: np.ndarray) -> np.ndarray:
    """Return 1 for observed cells adjacent to unresolved cells and conversely."""

    observed = np.asarray(observed_or_boundary, dtype=bool)
    unresolved = np.asarray(unresolved, dtype=bool)
    adjacent_unresolved = _neighbor_any(unresolved)
    adjacent_observed = _neighbor_any(observed)
    bonus = np.zeros(observed.shape, dtype=float)
    bonus[observed & adjacent_unresolved] = 1.0
    bonus[unresolved & adjacent_observed] = 1.0
    return bonus


def _neighbor_any(mask: np.ndarray) -> np.ndarray:
    out = np.zeros(mask.shape, dtype=bool)
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    return out


def _normalize_finite(values: np.ndarray) -> np.ndarray:
    out = np.zeros_like(values, dtype=float)
    finite = np.isfinite(values)
    if not finite.any():
        return out
    max_value = float(np.max(values[finite]))
    if max_value <= 0:
        return out
    out[finite] = values[finite] / max_value
    return out
