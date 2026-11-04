"""`search_mode: target` must actually steer selection under `spawn_type: hard`.

Regression cover for two defects found in the AdK campaign:

1. ``DensitySpawner._hard_rows`` sorted candidate bins by ``target_closeness``
   and then passed the sorted list through ``rng.permutation``, which discarded
   exactly the ordering target mode exists to impose. Measured over 300 seeds,
   hard+target selected frames 34.45 deg from the target against 34.19 deg for
   hard+explore -- statistically indistinguishable (Welch t = +0.76), i.e. the
   target was doing nothing.  Removing the permutation moved it to 5.96 deg.

2. ``RegularBinner.fit`` computed ``np.where(distance > 0, 1/distance, 0.0)``,
   so the one bin that *contains* the target scored 0.0 -- last of all 1600 bins
   -- instead of first.  Under `soft` its weight is multiplied by zero, so the
   spawner could never select from the target's own bin.

The old ``test_spawner_target_mode`` covers fps/lof/voronoi but not density, and
only asserts that valid indices come back, so neither defect was caught.
"""

from __future__ import annotations

import numpy as np
import pytest

from trails_md.binning.spatial import RegularBinner
from trails_md.spawners.density import DensitySpawner

NB = [40, 40]
LO = [85.0, 25.0]
HI = [165.0, 95.0]
TARGET = [106.4, 45.1]          # the AdK closed state, in LID/NMP degrees


def _pool(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    return np.column_stack([rng.uniform(88, 162, n), rng.uniform(28, 92, n)])


def _mean_distance(mode, probabilistic, seed, points, n_select=16):
    spawner = DensitySpawner(
        n_bins=NB, min_values=LO, max_values=HI, mode=mode,
        probabilistic=probabilistic,
        target=TARGET if mode == "target" else None,
        recent_window=0, seed=seed,
    )
    picked = points[spawner.sample(points, n_select)]
    return float(np.linalg.norm(picked - np.array(TARGET), axis=1).mean())


@pytest.mark.parametrize("probabilistic", [False, True], ids=["hard", "soft"])
def test_target_mode_selects_closer_than_explore(probabilistic):
    """Target mode must pull selections toward the target in BOTH spawn types.

    `hard` is the setting every published AdK arm used; before the fix it failed
    this assertion while `soft` passed.
    """
    points = _pool()
    seeds = range(60)
    target = np.array([_mean_distance("target", probabilistic, s, points) for s in seeds])
    explore = np.array([_mean_distance("explore", probabilistic, s, points) for s in seeds])
    # A real effect, not a seed accident: require a large, consistent gap.
    assert target.mean() < explore.mean() - 5.0, (
        f"target mode is not steering: target {target.mean():.2f} deg vs "
        f"explore {explore.mean():.2f} deg"
    )


def test_target_own_bin_is_the_most_favoured():
    """The bin containing the target must rank first, not last."""
    points = _pool()
    binner = RegularBinner(
        n_bins=NB, min_values=LO, max_values=HI, target=np.asarray(TARGET)
    )
    table = binner.fit(points)
    row = {b: i for i, b in enumerate(table.ids)}[
        tuple(binner.find_bins(np.asarray(TARGET).reshape(1, -1))[0].tolist())
    ]
    closeness = table.target_closeness
    assert closeness is not None
    assert closeness[row] == pytest.approx(closeness.max()), (
        f"target's own bin scored {closeness[row]:.6f} against a max of "
        f"{closeness.max():.6f}"
    )
    # and it must not be beaten by any other bin
    assert int((closeness > closeness[row]).sum()) == 0


def test_target_closeness_still_decreases_with_distance():
    """Guard the fix: closeness must stay monotone in distance away from the target."""
    points = _pool()
    binner = RegularBinner(
        n_bins=NB, min_values=LO, max_values=HI, target=np.asarray(TARGET)
    )
    table = binner.fit(points)
    # Distance is measured from the target BIN CENTRE, which is what the binner
    # itself uses; measuring from the target point instead can reorder bins that
    # are nearly equidistant and would test an invariant the code never claimed.
    row = {b: i for i, b in enumerate(table.ids)}[
        tuple(binner.find_bins(np.asarray(TARGET).reshape(1, -1))[0].tolist())
    ]
    d = np.linalg.norm(table.centers - table.centers[row], axis=1)
    order = np.argsort(d)
    c = table.target_closeness[order]
    assert np.all(np.diff(c) <= 1e-12), "closeness is not monotone decreasing in distance"
