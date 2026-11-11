"""Tests for gravity, air density and drag in PhysicsFunctions."""

import numpy as np
import pytest

from tools import constants as const
from tools.state_estimation import RealState
from tools.utils import PhysicsFunctions


@pytest.fixture
def physics() -> PhysicsFunctions:
    return PhysicsFunctions()


def state_at_rest(lat: float, lon: float, alt: float) -> np.ndarray:
    position = RealState.convert_geodetic_to_cartesian(np.array([lat, lon, alt]))
    return np.hstack((position, np.zeros(3)))


@pytest.mark.parametrize(
    "lat, expected_gravity",
    [
        (0.0, 9.7803),  # WGS84 normal gravity at the equator
        (90.0, 9.8322),  # WGS84 normal gravity at the poles
    ],
)
def test_surface_gravity_matches_wgs84(physics, lat, expected_gravity):
    """Central gravity + J2 + centrifugal term reproduce normal gravity."""
    state = state_at_rest(lat, 0.0, 0.0)

    acceleration = physics._gravitational_acceleration_ecef(state)

    assert np.linalg.norm(acceleration) == pytest.approx(expected_gravity, abs=5e-3)
    # At the equator and the poles gravity points at Earth's centre.
    direction = -state[:3] / np.linalg.norm(state[:3])
    assert acceleration / np.linalg.norm(acceleration) == pytest.approx(
        direction, abs=1e-9
    )


def test_coriolis_acceleration(physics):
    """A moving body gets the extra acceleration -2 * omega x v."""
    state_rest = state_at_rest(45.0, 10.0, 1000.0)
    velocity = np.array([100.0, -50.0, 20.0])
    state_moving = np.hstack((state_rest[:3], velocity))

    difference = physics._gravitational_acceleration_ecef(
        state_moving
    ) - physics._gravitational_acceleration_ecef(state_rest)

    assert difference == pytest.approx(
        -2 * np.cross(const.EARTH_ROTATION, velocity), abs=1e-12
    )


def test_simple_gravity_is_constant(physics):
    state = np.array([0.0, 0.0, 5000.0, 10.0, 0.0, -3.0])

    assert physics._simple_gravity(state) == pytest.approx([0.0, 0.0, -const.G0])


def test_realistic_air_density_is_exponential(physics):
    sea_level = state_at_rest(10.0, 20.0, 0.0)
    one_scale_height = state_at_rest(10.0, 20.0, const.scale_height)

    assert physics._realistic_airdensity(sea_level) == pytest.approx(
        const.SEA_LEVEL_AIR_DENSITY, rel=1e-6
    )
    assert physics._realistic_airdensity(one_scale_height) == pytest.approx(
        const.SEA_LEVEL_AIR_DENSITY / np.e, rel=1e-6
    )


def test_drag_force_opposes_air_relative_velocity(physics):
    velocity = np.array([30.0, 0.0, -40.0])
    wind = np.array([10.0, 5.0, 0.0])
    state = np.hstack((np.zeros(3), velocity))
    density, drag_coeff, area = 1.2, 1.4, 1.14

    drag = physics.compute_drag_force(
        state, wind, np.float64(density), drag_coeff, area
    )

    relative = velocity - wind
    speed = np.linalg.norm(relative)
    assert np.linalg.norm(drag) == pytest.approx(
        0.5 * density * speed**2 * drag_coeff * area
    )
    assert drag / np.linalg.norm(drag) == pytest.approx(-relative / speed)


def test_no_drag_when_moving_with_the_wind(physics):
    wind = np.array([10.0, 5.0, 0.0])
    state = np.hstack((np.zeros(3), wind))

    drag = physics.compute_drag_force(state, wind, np.float64(1.2), 1.4, 1.14)

    assert drag == pytest.approx(np.zeros(3))
