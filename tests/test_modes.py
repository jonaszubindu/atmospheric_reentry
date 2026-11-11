"""Tests for the separation of the simplified and the realistic mode."""

import numpy as np
import pytest

from tools import constants as const
from tools.modes import MODES, validate_mode

from conftest import ERA5_LAT, ERA5_LON


def test_there_are_exactly_two_modes():
    assert MODES == ("simplified", "realistic")


@pytest.mark.parametrize("mode", MODES)
def test_valid_modes_are_accepted(params, mode):
    params["mode"] = mode

    assert validate_mode(params) == mode


@pytest.mark.parametrize("mode", ["Realistic", "flat", "", None])
def test_unknown_mode_is_rejected(params, make_rocket, mode):
    params["mode"] = mode

    with pytest.raises(ValueError, match="Invalid mode"):
        validate_mode(params)
    with pytest.raises(ValueError, match="Invalid mode"):
        make_rocket(params, [0, 0, 1000], [0, 0, 0])


def test_missing_mode_is_rejected(params):
    del params["mode"]

    with pytest.raises(ValueError, match="Invalid mode"):
        validate_mode(params)


def test_era5_wind_needs_realistic_mode(params):
    params.update(mode="simplified", wind="ERA5")

    with pytest.raises(ValueError, match="ERA5 wind needs the realistic mode"):
        validate_mode(params)


@pytest.mark.filterwarnings("ignore:Mode. simplified")
def test_simplified_mode_binds_the_flat_earth_models(params, make_rocket):
    """Constant gravity along -z, constant density and altitude = z,
    whatever the position."""
    params["mode"] = "simplified"
    rocket = make_rocket(
        params, [0, 0, 1000], [0, 0, 0], wind_field_conditions=[270.0, 15.0]
    )

    for position in ([0.0, 0.0, 1000.0], [5e4, -2e4, 60000.0]):
        state = np.array([*position, 10.0, 0.0, -50.0])
        assert rocket.calc_altitude_abv_sea_level(state) == position[2]
        assert rocket.grav_acc(state) == pytest.approx([0.0, 0.0, -const.G0])
        assert rocket.get_air_density(state) == const.SEA_LEVEL_AIR_DENSITY
        # The wind is used as given: x is east, y is north.
        assert rocket._update_environment(0.0, state) == pytest.approx(
            [15.0, 0.0, 0.0], abs=1e-9
        )


def test_realistic_mode_binds_the_rotating_earth_models(params, make_rocket):
    """Geodetic altitude, gravity towards Earth's centre, density falling
    with altitude and the wind rotated from east/north/up into ECEF."""
    rocket = make_rocket(
        params,
        [ERA5_LAT, ERA5_LON, 1000.0],
        [0, 0, 0],
        wind_field_conditions=[270.0, 15.0],
    )
    low = rocket.state_cartesian.copy()
    high = np.hstack(
        (rocket.convert_geodetic_to_cartesian([ERA5_LAT, ERA5_LON, 20000.0]), 0, 0, 0)
    )

    assert rocket.calc_altitude_abv_sea_level(low) == pytest.approx(1000.0, abs=1e-6)
    assert rocket.calc_altitude_abv_sea_level(high) == pytest.approx(20000.0, abs=1e-6)
    assert rocket.get_air_density(high) < rocket.get_air_density(low)
    gravity = rocket.grav_acc(low)
    assert np.linalg.norm(gravity) == pytest.approx(9.80, abs=0.02)
    assert np.dot(gravity, low[:3]) < 0.0  # points towards Earth's centre

    wind_ecef = rocket._update_environment(0.0, low)
    wind_enu = rocket.convert_velocity_cartesian_to_geodetic(
        wind_ecef, ERA5_LAT, ERA5_LON
    )
    assert wind_enu == pytest.approx([15.0, 0.0, 0.0], abs=1e-9)


@pytest.mark.filterwarnings("ignore:Mode. simplified")
def test_drag_law_is_shared_by_both_modes(params, make_rocket):
    """Frame-independent physics: for the same air density and air-relative
    velocity, both modes compute the same drag force."""
    realistic = make_rocket(params, [ERA5_LAT, ERA5_LON, 1000.0], [0, 0, 0])
    simplified_params = dict(params, mode="simplified")
    simplified = make_rocket(simplified_params, [0, 0, 1000], [0, 0, 0])

    state = np.array([0.0, 0.0, 0.0, 120.0, -40.0, -300.0])
    wind = np.array([10.0, 5.0, 0.0])
    args = (state, wind, np.float64(0.8), 1.4, 1.14)

    assert simplified.compute_drag_force(*args) == pytest.approx(
        realistic.compute_drag_force(*args)
    )
