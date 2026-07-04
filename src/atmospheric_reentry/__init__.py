"""
Atmospheric Reentry Model

A package for simulating rocket reentry trajectories with varying
degrees of realism, including atmospheric effects, winds, and
Monte Carlo analyses.
"""

__version__ = "0.1.0"
__author__ = "Jonas Zbinden"

from . import simulate_rocket
from . import utils
from . import windfield
from . import logging
from . import state_estimation

__all__ = ["simulate_rocket", "utils", "windfield", "logging", "state_estimation"]
