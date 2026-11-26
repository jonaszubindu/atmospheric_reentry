"""
Copyright 2026 Jonas Zbinden
This software is licensed under the terms of the MIT License
which can be obtained at https://opensource.org/licenses/MIT or from the
LICENSE file in the root directory of this project.

Monte Carlo ensembles of reentry trajectories, run in parallel with joblib.

Each sample is one full trajectory from the same initial state, differing
only in the analytic wind: the compass direction the wind blows from and the
wind speed, drawn from uniform distributions. Dispersions of the initial
state and vehicle parameters (mass, drag) are not sampled yet.

Typical use::

    from tools.monte_carlo import run_monte_carlo

    trajectories, winds = run_monte_carlo(
        params,
        position_init=[38.5, 285.0, 60000],  # lat, lon, alt (realistic mode)
        velocity_init=[70, 70, -280],  # east, north, up in m/s
        n_samples=200,
        seed=1,
    )

``trajectories`` is a list of DataFrames (one per sample, as returned by
``Logging.get_full_state``) and ``winds`` the (n_samples, 2) array of
[direction, speed] used for each run.
"""

from __future__ import annotations

import warnings
from typing import Optional, Sequence

import numpy as np
import numpy.typing as npt
import pandas as pd
from joblib import Parallel, delayed

from .logging import Logging
from .modes import validate_mode
from .simulate_rocket import Rocket
from .state_estimation import RealState
from .windfield import WindField

DEFAULT_START_TIME = "2026-06-10T13:00:00"
DEFAULT_WIND_DIR_RANGE = (0.0, 360.0)  # deg, direction the wind blows from
DEFAULT_WIND_SPEED_RANGE = (0.0, 20.0)  # m/s


def default_n_steps(params: dict) -> int:
    """Step budget per run (also the size of the log buffers): twice the
    number of max_step steps needed to reach tmax, as in main.py."""
    return 2 * int(np.ceil(params["tmax"] / params["max_step"]))


def sample_uniform_wind(
    n_samples: int,
    wind_dir_range: tuple[float, float] = DEFAULT_WIND_DIR_RANGE,
    wind_speed_range: tuple[float, float] = DEFAULT_WIND_SPEED_RANGE,
    seed: Optional[int] = None,
) -> npt.NDArray[np.float64]:
    """Draw wind conditions from uniform distributions.

    Returns an (n_samples, 2) array of [direction the wind blows from in deg,
    speed in m/s]. The same seed gives the same samples.
    """
    rng = np.random.default_rng(seed)
    directions = rng.uniform(*wind_dir_range, n_samples)
    speeds = rng.uniform(*wind_speed_range, n_samples)
    return np.column_stack([directions, speeds])


def run_simulation(rocket: Rocket, n_steps: int) -> pd.DataFrame:
    """Step a rocket until it reaches the ground, the solver reaches tmax, or
    the step budget runs out, and return the logged state.

    Same stopping rules as main.main, without the verbose output.
    """
    for _ in range(n_steps):
        rocket.update_state()
        # update_state ends the solver at the exact ground crossing
        # (rocket.landed) or at tmax; no extra altitude check needed.
        if rocket.solver.status != "running":
            break
    else:
        warnings.warn(
            f"Simulation stopped after {n_steps} steps at t = "
            f"{rocket.solver.t:.1f} s, before reaching the ground or tmax. "
            "Increase n_steps to continue.",
            UserWarning,
        )
    return rocket.logger.get_full_state()


def _run_sample(
    params: dict,
    position_init: npt.NDArray[np.float64],
    velocity_init: npt.NDArray[np.float64],
    wind_conditions: Sequence[float],
    n_steps: int,
    start_time: str,
) -> pd.DataFrame:
    """Build and run one trajectory. Runs inside a joblib worker, so the
    rocket, logger and wind field are created there rather than pickled."""
    if validate_mode(params) == "realistic":
        lat, lon, _ = position_init
        position = RealState.convert_geodetic_to_cartesian(position_init)
        velocity = RealState.convert_velocity_geodetic_to_cartesian(
            velocity_init, lat, lon
        )
    else:
        position, velocity = position_init, velocity_init

    logger = Logging(
        n_steps, initial_conditions_geodetic=[position_init, velocity_init]
    )
    wind_field = WindField(
        start_time=start_time,
        params=params,
        logger=None,
        wind_field_conditions=[float(wind_conditions[0]), float(wind_conditions[1])],
    )
    rocket = Rocket(
        initial_conditions=[position, velocity],
        params=params,
        logger=logger,
        wind_field=wind_field,
    )
    return run_simulation(rocket, n_steps)


def run_monte_carlo(
    params: dict,
    position_init: npt.ArrayLike,
    velocity_init: npt.ArrayLike,
    n_samples: Optional[int] = None,
    *,
    wind_samples: Optional[npt.ArrayLike] = None,
    wind_dir_range: tuple[float, float] = DEFAULT_WIND_DIR_RANGE,
    wind_speed_range: tuple[float, float] = DEFAULT_WIND_SPEED_RANGE,
    seed: Optional[int] = None,
    n_steps: Optional[int] = None,
    start_time: str = DEFAULT_START_TIME,
    n_jobs: int = -1,
    verbose: int = 0,
) -> tuple[list[pd.DataFrame], npt.NDArray[np.float64]]:
    """Run an ensemble of trajectories in parallel, varying only the wind.

    Parameters:
    params (dict): rocket and simulation parameters as in main.py. A copy
        with wind set to "Monte Carlo" is used; ERA5 wind is not supported.
    position_init, velocity_init: initial state. In realistic mode geodetic
        (lat, lon, alt) and (east, north, up); in simplified mode Cartesian.
    n_samples (int): number of runs when wind_samples is not given.
    wind_samples (array, optional): explicit (n, 2) [direction, speed] wind
        conditions, one run each. Overrides n_samples and the ranges.
    wind_dir_range, wind_speed_range: bounds of the uniform distributions.
    seed (int, optional): seed for the wind sampling (reproducible ensemble).
    n_steps (int, optional): step budget per run; default_n_steps if None.
    start_time (str): simulation start time for the wind field.
    n_jobs (int): joblib workers; -1 uses all cores, 1 runs sequentially.
    verbose (int): joblib progress verbosity.

    Returns:
    (trajectories, wind_samples): one DataFrame per run, in sample order,
    and the (n, 2) wind conditions used.
    """
    if params.get("wind") == "ERA5":
        raise ValueError(
            "run_monte_carlo varies the analytic wind; set params['wind'] to "
            "'default' or 'Monte Carlo', not 'ERA5'."
        )
    run_params = {**params, "wind": "Monte Carlo"}

    if wind_samples is None:
        if n_samples is None:
            raise ValueError("Give either n_samples or wind_samples.")
        wind_samples = sample_uniform_wind(
            n_samples, wind_dir_range, wind_speed_range, seed
        )
    wind_samples = np.atleast_2d(np.asarray(wind_samples, dtype=np.float64))
    if wind_samples.shape[1] != 2:
        raise ValueError("wind_samples must have shape (n, 2): [direction, speed].")

    if n_steps is None:
        n_steps = default_n_steps(run_params)
    position = np.asarray(position_init, dtype=np.float64)
    velocity = np.asarray(velocity_init, dtype=np.float64)

    trajectories = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(_run_sample)(run_params, position, velocity, wind, n_steps, start_time)
        for wind in wind_samples
    )
    return list(trajectories), wind_samples
