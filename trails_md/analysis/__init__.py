"""Post-hoc MSM analysis, committor, and plotting utilities.

``data`` holds matplotlib-free numerics (loading msm.npz/cvs.npz, free
energies); ``plots`` holds the matplotlib visualisations.
"""

from . import data
from .committor import DiscreteCommittorResult, solve_forward_committor

__all__ = ["DiscreteCommittorResult", "data", "solve_forward_committor"]
