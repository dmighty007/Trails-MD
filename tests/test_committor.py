"""Tests for committor analysis utilities."""

from __future__ import annotations

import numpy as np

from examples.toy_committor.muller_brown import MIN_A, MIN_B, basin_predicate, grad
from trails_md.analysis.committor import (
    active_indices_from_symbols,
    solve_forward_committor,
)
from trails_md.analysis.local_stitch import (
    committor_local_stitch,
    committor_local_stitch_from_segments,
)
from trails_md.analysis.markov_check import compare_transition_subsets
from trails_md.analysis.pde_reference import solve_pde_committor
from trails_md.analysis.shooting import committor_shoot


def test_forward_committor_solves_known_three_state_chain():
    transition_matrix = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.25, 0.25, 0.50],
            [0.0, 0.0, 1.0],
        ]
    )
    result = solve_forward_committor(transition_matrix, basin_a=[0], basin_b=[2])
    assert np.allclose(result.q, [0.0, 2.0 / 3.0, 1.0])
    assert np.nanmax(np.abs(result.residual)) < 1.0e-12


def test_forward_committor_enforces_boundaries_and_full_id_remap():
    transition_matrix = np.array(
        [
            [0.9, 0.1, 0.0],
            [0.2, 0.4, 0.4],
            [0.0, 0.1, 0.9],
        ]
    )
    result = solve_forward_committor(
        transition_matrix,
        basin_a=[10, 99],
        basin_b=[30],
        state_symbols=np.array([10.0, 20.0, 30.0]),
        basin_id_space="full",
    )
    assert result.q[result.basin_a_states[0]] == 0.0
    assert result.q[result.basin_b_states[0]] == 1.0
    assert result.excluded_basin_a.tolist() == [99]
    active, excluded = active_indices_from_symbols(np.array([10, 20, 30]), [20, 40])
    assert active.tolist() == [1]
    assert excluded.tolist() == [40]


def test_markov_check_flags_heterogeneous_transition_rows():
    subset_counts = np.zeros((2, 2, 3), dtype=float)
    subset_counts[0, 0, 0] = 20
    subset_counts[0, 1, 2] = 20
    subset_counts[1, :, 1] = 10
    result = compare_transition_subsets(subset_counts, threshold_tv=0.5)
    assert 0 in result.rejected_states
    assert 1 not in result.rejected_states


def test_toy_muller_brown_pipeline_runs_fast():
    in_a = basin_predicate(MIN_A, 0.12)
    in_b = basin_predicate(MIN_B, 0.12)

    pde = solve_pde_committor(
        grad,
        in_a,
        in_b,
        xlim=(-1.6, 1.2),
        ylim=(-0.3, 2.1),
        nx=12,
        ny=12,
    )
    assert pde.q.shape == (12, 12)
    assert np.allclose(pde.q[pde.is_a], 0.0)
    assert np.allclose(pde.q[pde.is_b], 1.0)

    points = np.vstack([MIN_A, 0.5 * (MIN_A + MIN_B), MIN_B])
    shoot = committor_shoot(
        points,
        grad,
        in_a,
        in_b,
        n_shots=2,
        seed=3,
        dt=2.0e-4,
        max_steps=20,
    )
    assert shoot.q_mean.shape == (3,)

    stitch = committor_local_stitch(
        grad,
        in_a,
        in_b,
        xlim=(-1.6, 1.2),
        ylim=(-0.3, 2.1),
        nx=6,
        ny=6,
        n_shots_per_cell=2,
        seed=4,
        dt=2.0e-4,
        max_steps=20,
    )
    assert stitch.q.shape == (6, 6)
    assert stitch.residual.shape == (6, 6)
    assert stitch.markov_total_variation.shape == (6, 6)


def test_local_stitch_from_observed_first_exits():
    def in_a(points):
        pts = np.asarray(points, dtype=float)
        single = pts.ndim == 1
        rows = pts.reshape(1, -1) if single else pts
        mask = np.linalg.norm(rows[:, :2] - np.array([0.0, 1.0]), axis=1) < 0.1
        return mask[0] if single else mask

    def in_b(points):
        pts = np.asarray(points, dtype=float)
        single = pts.ndim == 1
        rows = pts.reshape(1, -1) if single else pts
        mask = np.linalg.norm(rows[:, :2] - np.array([2.0, 1.0]), axis=1) < 0.1
        return mask[0] if single else mask

    segments = [
        np.array([[1.0, 1.0], [2.0, 1.0]]),
        np.array([[1.0, 1.0], [0.0, 1.0]]),
    ]
    stitch = committor_local_stitch_from_segments(
        segments,
        in_a,
        in_b,
        xlim=(0.0, 2.0),
        ylim=(0.0, 2.0),
        nx=3,
        ny=3,
    )

    assert stitch.counts.sum() == 2
    assert stitch.q[1, 1] == 0.5
    assert np.isfinite(stitch.residual[1, 1])
