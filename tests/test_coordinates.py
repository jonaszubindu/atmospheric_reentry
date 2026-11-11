"""Tests for the coordinate conversions in RealState."""

import numpy as np
import pytest

from tools import constants as const
from tools.state_estimation import RealState


@pytest.mark.parametrize(
    "position_geodetic",
    [
        [0.0, 0.0, 0.0],
        [47.3986, 8.6486, 440.0],
        [38.5, -75.0, 60000.0],
        [-33.9, 151.2, 120000.0],
    ],
)
def test_position_round_trip(position_geodetic):
    position_cartesian = RealState.convert_geodetic_to_cartesian(
        np.array(position_geodetic)
    )
    lat, lon, alt = RealState.convert_cartesian_to_geodetic(position_cartesian)

    assert lat == pytest.approx(position_geodetic[0], abs=1e-9)
    assert lon == pytest.approx(position_geodetic[1], abs=1e-9)
    assert alt == pytest.approx(position_geodetic[2], abs=1e-6)


def test_equator_prime_meridian_is_on_x_axis():
    position = RealState.convert_geodetic_to_cartesian(np.array([0.0, 0.0, 0.0]))

    assert position == pytest.approx([const.RE, 0.0, 0.0], abs=1e-6)


def test_velocity_round_trip():
    lat, lon = 47.3986, 8.6486
    velocity_enu = np.array([70.0, -30.0, -280.0])

    velocity_ecef = RealState.convert_velocity_geodetic_to_cartesian(
        velocity_enu, lat, lon
    )
    velocity_back = RealState.convert_velocity_cartesian_to_geodetic(
        velocity_ecef, lat, lon
    )

    assert velocity_back == pytest.approx(velocity_enu, abs=1e-9)
    assert np.linalg.norm(velocity_ecef) == pytest.approx(np.linalg.norm(velocity_enu))


def test_enu_axes_at_equator_prime_meridian():
    """At lat = lon = 0: up is +x, east is +y and north is +z in ECEF."""
    east = RealState.convert_velocity_geodetic_to_cartesian(
        np.array([1.0, 0.0, 0.0]), 0.0, 0.0
    )
    north = RealState.convert_velocity_geodetic_to_cartesian(
        np.array([0.0, 1.0, 0.0]), 0.0, 0.0
    )
    up = RealState.convert_velocity_geodetic_to_cartesian(
        np.array([0.0, 0.0, 1.0]), 0.0, 0.0
    )

    assert east == pytest.approx([0.0, 1.0, 0.0], abs=1e-12)
    assert north == pytest.approx([0.0, 0.0, 1.0], abs=1e-12)
    assert up == pytest.approx([1.0, 0.0, 0.0], abs=1e-12)
