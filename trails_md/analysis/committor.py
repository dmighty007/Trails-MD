"""Discrete committor solvers for Markov state models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.sparse import csr_matrix, eye
from scipy.sparse.linalg import spsolve

BasinIdSpace = Literal["active", "full"]


@dataclass(frozen=True)
class DiscreteCommittorResult:
    """Forward committor solution on the active Markov-state index space."""

    q: np.ndarray
    interior_states: np.ndarray
    basin_a_states: np.ndarray
    basin_b_states: np.ndarray
    residual: np.ndarray
    state_symbols: np.ndarray | None = None
    excluded_basin_a: np.ndarray | None = None
    excluded_basin_b: np.ndarray | None = None


def active_indices_from_symbols(
    state_symbols: np.ndarray,
    basin: np.ndarray | list[int] | tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Map full cluster IDs onto largest-connected-set active-state indices.

    ``MSMResult.state_symbols`` is ordered as ``active_index -> full_cluster_id``.
    Basin IDs that were removed by the largest-connected-set restriction are
    returned separately so callers can report them.
    """

    symbols = np.asarray(state_symbols, dtype=int)
    requested = np.asarray(basin, dtype=int).ravel()
    symbol_to_active = {int(symbol): idx for idx, symbol in enumerate(symbols)}

    active: list[int] = []
    excluded: list[int] = []
    for state in requested:
        idx = symbol_to_active.get(int(state))
        if idx is None:
            excluded.append(int(state))
        else:
            active.append(int(idx))
    return np.asarray(active, dtype=int), np.asarray(excluded, dtype=int)


def solve_forward_committor(
    transition_matrix: np.ndarray,
    basin_a: np.ndarray | list[int] | tuple[int, ...],
    basin_b: np.ndarray | list[int] | tuple[int, ...],
    *,
    state_symbols: np.ndarray | None = None,
    basin_id_space: BasinIdSpace = "active",
    row_sum_atol: float = 1.0e-8,
) -> DiscreteCommittorResult:
    """Solve the discrete forward committor ``q = P q`` with A/B boundaries.

    Parameters
    ----------
    transition_matrix:
        Row-stochastic transition matrix in active-state order.
    basin_a, basin_b:
        Reactant and product state IDs. Use active-state IDs by default. If
        ``basin_id_space="full"``, pass full cluster IDs and provide
        ``state_symbols`` for remapping.
    state_symbols:
        Optional active-to-full cluster-ID map from :class:`MSMResult`.
    basin_id_space:
        Whether basin IDs are already active indices or full cluster IDs.
    row_sum_atol:
        Absolute tolerance for validating stochastic rows.
    """

    P = np.asarray(transition_matrix, dtype=float)
    if P.ndim != 2 or P.shape[0] != P.shape[1]:
        raise ValueError("transition_matrix must be a square 2-D array.")
    n_states = P.shape[0]
    if np.any(P < -row_sum_atol):
        raise ValueError("transition_matrix contains negative entries.")
    if not np.allclose(P.sum(axis=1), 1.0, atol=row_sum_atol):
        raise ValueError("transition_matrix rows must sum to 1.")

    excluded_a = np.asarray([], dtype=int)
    excluded_b = np.asarray([], dtype=int)
    if basin_id_space == "full":
        if state_symbols is None:
            raise ValueError("state_symbols is required for full-space basin IDs.")
        a_states, excluded_a = active_indices_from_symbols(state_symbols, basin_a)
        b_states, excluded_b = active_indices_from_symbols(state_symbols, basin_b)
    elif basin_id_space == "active":
        a_states = np.asarray(basin_a, dtype=int).ravel()
        b_states = np.asarray(basin_b, dtype=int).ravel()
    else:
        raise ValueError("basin_id_space must be 'active' or 'full'.")

    a_states = np.unique(a_states)
    b_states = np.unique(b_states)
    if a_states.size == 0:
        raise ValueError("basin_a has no states in the active transition matrix.")
    if b_states.size == 0:
        raise ValueError("basin_b has no states in the active transition matrix.")
    if np.any(a_states < 0) or np.any(a_states >= n_states):
        raise IndexError("basin_a contains active-state indices outside P.")
    if np.any(b_states < 0) or np.any(b_states >= n_states):
        raise IndexError("basin_b contains active-state indices outside P.")
    if np.intersect1d(a_states, b_states).size:
        raise ValueError("basin_a and basin_b must be disjoint.")

    boundary = np.zeros(n_states, dtype=bool)
    boundary[a_states] = True
    boundary[b_states] = True
    interior = np.flatnonzero(~boundary)

    q = np.zeros(n_states, dtype=float)
    q[b_states] = 1.0
    if interior.size:
        P_ii = P[np.ix_(interior, interior)]
        P_ib = P[np.ix_(interior, b_states)]
        rhs = P_ib.sum(axis=1)
        matrix = eye(interior.size, format="csr") - csr_matrix(P_ii)
        q[interior] = spsolve(matrix, rhs)

    residual = np.full(n_states, np.nan, dtype=float)
    residual[interior] = q[interior] - P[interior] @ q
    return DiscreteCommittorResult(
        q=q,
        interior_states=interior,
        basin_a_states=a_states,
        basin_b_states=b_states,
        residual=residual,
        state_symbols=(
            None if state_symbols is None else np.asarray(state_symbols, dtype=int)
        ),
        excluded_basin_a=excluded_a,
        excluded_basin_b=excluded_b,
    )
