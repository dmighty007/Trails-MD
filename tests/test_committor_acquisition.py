from __future__ import annotations

import numpy as np

from examples.toy_committor.muller_brown import MIN_A, MIN_B, basin_predicate, grad
from examples.toy_committor.run_committor_guided_adaptive import run_closed_loop
from trails_md.analysis.committor_acquisition import (
    CommittorAcquisitionWeights,
    score_local_stitch_cells,
)
from trails_md.analysis.local_stitch import LocalStitchResult
from trails_md.analysis.markov_check import MarkovCheckResult
from trails_md.analysis.pde_reference import solve_pde_committor


def test_committor_acquisition_prioritizes_transition_and_low_counts():
    q = np.array(
        [
            [0.0, 0.05, np.nan],
            [0.10, 0.50, 0.50],
            [np.nan, 0.95, 1.0],
        ]
    )
    n_states = q.size
    counts = np.zeros((n_states, n_states + 2), dtype=float)
    center = 4
    low_count_transition = 5
    near_basin = 1
    counts[center, center] = 100
    counts[low_count_transition, low_count_transition] = 1
    counts[near_basin, near_basin] = 1
    is_a = np.zeros_like(q, dtype=bool)
    is_b = np.zeros_like(q, dtype=bool)
    is_a[0, 0] = True
    is_b[2, 2] = True
    tv = np.zeros(n_states, dtype=float)
    tv[low_count_transition] = 1.0
    markov = MarkovCheckResult(
        total_variation=tv,
        jensen_shannon=np.zeros(n_states, dtype=float),
        rejected_states=np.array([low_count_transition]),
        threshold_tv=0.25,
        threshold_js=None,
        n_subsets=2,
    )
    stitch = LocalStitchResult(
        xs=np.arange(3, dtype=float),
        ys=np.arange(3, dtype=float),
        q=q,
        state_id=np.arange(n_states).reshape(3, 3),
        transition_matrix=np.zeros((n_states, n_states + 2), dtype=float),
        is_a=is_a,
        is_b=is_b,
        isolated=np.array([2, 6]),
        residual=np.full_like(q, np.nan, dtype=float),
        counts=counts,
        markov_check=markov,
        markov_total_variation=tv.reshape(3, 3),
        markov_jensen_shannon=np.zeros_like(q, dtype=float),
    )

    score = score_local_stitch_cells(
        stitch,
        weights=CommittorAcquisitionWeights(
            transition=1.0,
            undersampling=0.5,
            frontier=0.0,
            markov=0.25,
        ),
    ).score.ravel()

    assert score[low_count_transition] > score[center]
    assert score[center] > score[near_basin]
    assert score[0] == 0.0
    assert score[8] == 0.0


def test_toy_committor_guided_loop_runs_tiny_muller_brown():
    in_a = basin_predicate(MIN_A, 0.12)
    in_b = basin_predicate(MIN_B, 0.12)
    pde = solve_pde_committor(
        grad,
        in_a,
        in_b,
        xlim=(-1.6, 1.2),
        ylim=(-0.3, 2.1),
        nx=10,
        ny=10,
    )
    result = run_closed_loop(
        strategy="adaptive",
        seed=7,
        in_a=in_a,
        in_b=in_b,
        pde=pde,
        grid_size=6,
        iterations=2,
        cells_per_iteration=4,
        shots_per_cell=1,
        switch_mode="fixed",
        warmup_iterations=1,
        exploration_min_iterations=1,
        exploration_patience=1,
        exploration_max_new_observed=0,
        exploration_max_new_finite=0,
        exploration_max_isolated_drop=0,
        dt=2.0e-4,
        max_steps=50,
        weights=CommittorAcquisitionWeights(),
    )

    metrics = result["metrics"]
    assert len(metrics) == 2
    assert metrics[-1]["total_first_exits"] >= metrics[0]["total_first_exits"]
    assert result["stitch"].q.shape == (6, 6)
