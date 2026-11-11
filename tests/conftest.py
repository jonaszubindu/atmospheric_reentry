"""Shared fixtures for the test suite."""

import pathlib

import numpy as np
import pytest

from tools.logging import Logging
from tools.simulate_rocket import Rocket
from tools.state_estimation import RealState
from tools.windfield import WindField

SRC_DIR = pathlib.Path(__file__).resolve().parent.parent / "src"

# Start time and position covered by the ERA5 sample data shipped in
# src/tools/windmodel_data.
ERA5_START_TIME = "2026-06-05T13:00:00"
ERA5_LAT, ERA5_LON = 38.5, 285.0


@pytest.fixture
def params() -> dict:
    """Default rocket and simulation parameters, as used in main.py."""
    return {
        "mass": 700,
        "drag_coefficient": 1.4,
        "drag_coefficient_parachute": 2.0,
        "cross_sectional_area": 1.14,
        "cross_sectional_area_parachute": 18,
        "parachute_opening_altitude": 10000,
        "tmax": 3000,
        "max_step": 0.3,
        "wind": "default",
        "mode": "realistic",
        "verbose": False,
    }


def n_steps_for(params: dict) -> int:
    """Step budget for a run, computed as in main.py."""
    return 2 * int(np.ceil(params["tmax"] / params["max_step"]))


@pytest.fixture
def make_rocket():
    """Factory that builds a Rocket with its logger and wind field.

    In realistic mode the initial conditions are geodetic: position
    (lat, lon, alt) and velocity (east, north, up). In simplified mode they
    are plain Cartesian x, y, z.
    """

    def _make(
        params: dict,
        position_init,
        velocity_init,
        wind_field_conditions=None,
        n_steps=None,
        start_time: str = ERA5_START_TIME,
    ) -> Rocket:
        position_init = np.asarray(position_init, dtype=np.float64)
        velocity_init = np.asarray(velocity_init, dtype=np.float64)
        if params["mode"] == "realistic":
            lat, lon, _ = position_init
            position = RealState.convert_geodetic_to_cartesian(position_init)
            velocity = RealState.convert_velocity_geodetic_to_cartesian(
                velocity_init, lat, lon
            )
        else:
            position, velocity = position_init, velocity_init

        if n_steps is None:
            n_steps = n_steps_for(params)
        logger = Logging(
            n_steps, initial_conditions_geodetic=[position_init, velocity_init]
        )
        wind_field = WindField(
            start_time=start_time,
            params=params,
            logger=logger,
            wind_field_conditions=wind_field_conditions,
        )
        return Rocket(
            initial_conditions=[position, velocity],
            params=params,
            logger=logger,
            wind_field=wind_field,
        )

    return _make


@pytest.fixture
def in_src_dir(monkeypatch):
    """Run the test from src/, where the ERA5 data paths are resolved."""
    monkeypatch.chdir(SRC_DIR)
