"""Within-state Markovianity diagnostics for local committor stitching."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MarkovCheckResult:
    """Per-state disagreement between independently estimated transition rows."""

    total_variation: np.ndarray
    jensen_shannon: np.ndarray
    rejected_states: np.ndarray
    threshold_tv: float
    threshold_js: float | None
    n_subsets: int

    @property
    def max_total_variation(self) -> float:
        finite = self.total_variation[np.isfinite(self.total_variation)]
        return float(np.max(finite)) if finite.size else float("nan")

    @property
    def max_jensen_shannon(self) -> float:
        finite = self.jensen_shannon[np.isfinite(self.jensen_shannon)]
        return float(np.max(finite)) if finite.size else float("nan")


def transition_row(counts: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Return a smoothed transition row over the observed support."""

    row = np.asarray(counts, dtype=float)
    support = row > 0
    if not np.any(support):
        return np.full(row.shape, np.nan, dtype=float)
    smoothed = np.zeros_like(row, dtype=float)
    smoothed[support] = row[support] + alpha
    return smoothed / smoothed.sum()


def total_variation(p: np.ndarray, q: np.ndarray) -> float:
    """Total-variation distance between probability rows."""

    valid = np.isfinite(p) & np.isfinite(q)
    if not np.any(valid):
        return float("nan")
    return float(0.5 * np.abs(p[valid] - q[valid]).sum())


def jensen_shannon_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence in nats between probability rows."""

    valid = np.isfinite(p) & np.isfinite(q)
    if not np.any(valid):
        return float("nan")
    p_valid = p[valid]
    q_valid = q[valid]
    m = 0.5 * (p_valid + q_valid)
    return float(0.5 * _kl(p_valid, m) + 0.5 * _kl(q_valid, m))


def compare_transition_subsets(
    subset_counts: np.ndarray,
    *,
    alpha: float = 0.5,
    threshold_tv: float = 0.25,
    threshold_js: float | None = None,
) -> MarkovCheckResult:
    """Compare transition rows estimated from independent subsets per state.

    Parameters
    ----------
    subset_counts:
        Array of shape ``(n_states, n_subsets, n_destinations)``.
    alpha:
        Jeffreys-style pseudocount applied on each subset's observed support.
    threshold_tv, threshold_js:
        Per-state rejection thresholds. Jensen-Shannon thresholding is optional.
    """

    counts = np.asarray(subset_counts, dtype=float)
    if counts.ndim != 3:
        raise ValueError(
            "subset_counts must have shape (states, subsets, destinations)."
        )
    n_states, n_subsets, _ = counts.shape
    if n_subsets < 2:
        raise ValueError("at least two subsets are required for a Markov check.")

    tv = np.full(n_states, np.nan, dtype=float)
    js = np.full(n_states, np.nan, dtype=float)
    for state in range(n_states):
        rows = [
            transition_row(counts[state, subset], alpha=alpha)
            for subset in range(n_subsets)
        ]
        pair_tv: list[float] = []
        pair_js: list[float] = []
        for left in range(n_subsets):
            for right in range(left + 1, n_subsets):
                pair_tv.append(total_variation(rows[left], rows[right]))
                pair_js.append(jensen_shannon_divergence(rows[left], rows[right]))
        finite_tv = np.asarray(pair_tv, dtype=float)
        finite_tv = finite_tv[np.isfinite(finite_tv)]
        finite_js = np.asarray(pair_js, dtype=float)
        finite_js = finite_js[np.isfinite(finite_js)]
        if finite_tv.size:
            tv[state] = float(np.max(finite_tv))
        if finite_js.size:
            js[state] = float(np.max(finite_js))

    rejected = np.flatnonzero(tv > threshold_tv)
    if threshold_js is not None:
        rejected = np.union1d(rejected, np.flatnonzero(js > threshold_js))
    return MarkovCheckResult(
        total_variation=tv,
        jensen_shannon=js,
        rejected_states=rejected.astype(int),
        threshold_tv=float(threshold_tv),
        threshold_js=None if threshold_js is None else float(threshold_js),
        n_subsets=int(n_subsets),
    )


def _kl(p: np.ndarray, q: np.ndarray) -> float:
    mask = (p > 0) & (q > 0)
    if not np.any(mask):
        return 0.0
    return float(np.sum(p[mask] * np.log(p[mask] / q[mask])))
