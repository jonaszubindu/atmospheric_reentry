"""Tests of the RK45 integration settings: tolerances, large steps, the
interpolated output grid, the ground crossing and the pseudo forces."""

import numpy as np
import pytest

from tools import constants as const
from tools.monte_carlo import run_simulation
from tools.utils import PhysicsFunctions

# Short realistic drop with some horizontal speed (parachute open below
# 10 km, so open from the start).
POS = [46.95, 7.45, 3000.0]
VEL = [60.0, 20.0, -50.0]


def impact(df):
    last = df.iloc[-1]
    return last["position geodetic lat"], last["position geodetic lon"], last["time"]


def test_output_grid_and_exact_ground_crossing(params, make_rocket):
    params.update(max_step=5.0, output_interval=0.5)
    rocket = make_rocket(params, POS, VEL, wind_field_conditions=[270.0, 10.0])
    df = run_simulation(rocket, n_steps=10_000)

    assert rocket.landed
    times = df["time"].to_numpy()
    # every row but the last lies on the 0.5 s grid ...
    np.testing.assert_allclose(times[:-1] / 0.5, np.round(times[:-1] / 0.5), atol=1e-9)
    assert np.all(np.diff(times) > 0)
    # ... and the last one is the ground crossing, resolved to ~1e-6 s
    assert -0.01 <= df.iloc[-1]["altitude"] <= 0.0
    assert np.all(df["altitude"].to_numpy()[:-1] > 0.0)


def test_log_buffers_grow_beyond_initial_size(params, make_rocket):
    params.update(max_step=5.0, output_interval=0.5)
    rocket = make_rocket(params, POS, VEL, n_steps=8)  # far too small on purpose
    df = run_simulation(rocket, n_steps=10_000)
    assert rocket.landed and len(df) > 8


def test_large_steps_match_small_steps(params, make_rocket):
    """Accuracy comes from the tolerances, not from a small max_step."""
    small = dict(params, max_step=0.1)
    large = dict(params, max_step=5.0, output_interval=0.5)
    lat_s, lon_s, t_s = impact(
        run_simulation(make_rocket(small, POS, VEL, [270.0, 10.0]), 100_000)
    )
    lat_l, lon_l, t_l = impact(
        run_simulation(make_rocket(large, POS, VEL, [270.0, 10.0]), 100_000)
    )
    metres = 111_320 * np.hypot(
        lat_l - lat_s, (lon_l - lon_s) * np.cos(np.radians(lat_s))
    )
    assert metres < 1.0
    assert t_l == pytest.approx(t_s, abs=0.05)


@pytest.fixture
def physics() -> PhysicsFunctions:
    return PhysicsFunctions()


def test_pseudo_forces_match_cross_products(physics):
    """The closed-form Coriolis and centrifugal terms equal the cross
    products -2 Omega x v and -Omega x (Omega x r)."""
    rng = np.random.default_rng(0)
    for _ in range(5):
        position = rng.normal(0, 4e6, 3) + np.array([4e6, 0, 4e6])
        velocity = rng.normal(0, 3e3, 3)
        state = np.concatenate([position, velocity])

        total = physics._gravitational_acceleration_ecef(state)

        omega = const.EARTH_ROTATION
        pseudo = -2 * np.cross(omega, velocity) - np.cross(
            omega, np.cross(omega, position)
        )
        no_rotation = total - pseudo  # remaining part must be pure gravity
        r = np.linalg.norm(position)
        # gravity points (nearly) to the centre: J2 tilts it by < 1e-3 rad
        cos_angle = -no_rotation @ position / (np.linalg.norm(no_rotation) * r)
        assert cos_angle > 1 - 1e-6
        assert np.linalg.norm(no_rotation) == pytest.approx(const.GM / r**2, rel=3e-3)
