from __future__ import annotations

import numpy as np

from trails_md.config import TrailsMDConfig
from trails_md.spawners import SpawnerFactory
from trails_md.spawners.committor import _pooled_points_and_segments


def _line_segments(n_segments: int = 4, n_frames: int = 12) -> list[np.ndarray]:
    segments = []
    for offset in np.linspace(-0.03, 0.03, n_segments):
        x = np.linspace(0.0, 1.0, n_frames)
        y = np.full(n_frames, offset)
        segments.append(np.column_stack([x, y]))
    return segments


def test_committor_config_and_registration():
    cfg = TrailsMDConfig(
        system={"conf_file": "a.gro", "top_file": "a.top"},
        engine={"md_engine": "openmm"},
        spawning={
            "spawn_scheme": "committor",
            "committor_target": [[0.8, 1.1], [-0.2, 0.2]],
        },
    )
    assert cfg.spawning.spawn_scheme == "committor"
    spawner = SpawnerFactory.get(
        "committor",
        n_bins=[6, 6],
        min_values=[0.0, -0.5],
        max_values=[1.0, 0.5],
        committor_target=[[0.8, 1.1], [-0.2, 0.2]],
        seed=7,
    )
    assert spawner is not None


def test_committor_spawner_uses_density_fallback_before_gate():
    points = np.vstack(_line_segments())
    spawner = SpawnerFactory.get(
        "committor",
        n_bins=[6, 6],
        min_values=[0.0, -0.5],
        max_values=[1.0, 0.5],
        committor_source=[[-0.1, 0.2], [-0.2, 0.2]],
        committor_target=[[0.8, 1.1], [-0.2, 0.2]],
        committor_min_iteration=50,
        seed=4,
    )
    picks = spawner.sample(points, top_n=4, history={})
    assert len(picks) == 4
    assert spawner.committor_enabled is False


def test_committor_spawner_switches_after_open_gate():
    segments = _line_segments()
    history = {
        0: {
            "projection": np.vstack(segments[:2]),
            "trajectories": ["w0.xtc", "w1.xtc"],
        }
    }
    points = np.vstack(segments[2:])
    spawner = SpawnerFactory.get(
        "committor",
        n_bins=[6, 6],
        min_values=[0.0, -0.5],
        max_values=[1.0, 0.5],
        committor_source=[[-0.1, 0.2], [-0.2, 0.2]],
        committor_target=[[0.8, 1.1], [-0.2, 0.2]],
        committor_min_iteration=0,
        committor_patience=0,
        committor_min_target_hits=1,
        seed=9,
    )
    picks = spawner.sample(points, top_n=2, history=history)
    assert len(picks) == 2
    assert all(0 <= pick < 48 for pick in picks)
    assert spawner.committor_enabled is True
    assert spawner.switch_iteration == 1
    assert spawner.last_metrics["target_hits"] > 0


def test_current_projection_is_split_by_current_walkers():
    points = np.vstack(_line_segments(n_segments=3, n_frames=5))
    _, segments = _pooled_points_and_segments(points, history=None, current_walkers=3)
    assert [len(segment) for segment in segments] == [5, 5, 5]
