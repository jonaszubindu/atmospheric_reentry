"""End-to-end tests of the simulation loop against known solutions."""

import numpy as np
import pytest

from main import main
from tools import constants as const

from conftest import ERA5_LAT, ERA5_LON, n_steps_for


def run(rocket, n_steps=None):
    if n_steps is None:
        n_steps = n_steps_for(rocket.params)
    return main(n_steps, rocket, verbose=False)


@pytest.fixture
def simplified_params(params) -> dict:
    params["mode"] = "simplified"
    return params


@pytest.mark.filterwarnings("ignore:Mode. simplified")
class TestSimplifiedMode:
    def test_free_fall_without_drag(self, simplified_params, make_rocket):
        """Without drag the fall time is sqrt(2 h / g)."""
        height, speed_x = 2000.0, 30.0
        simplified_params.update(
            drag_coefficient=0.0, drag_coefficient_parachute=0.0, max_step=0.05
        )
        rocket = make_rocket(simplified_params, [0, 0, height], [speed_x, 0, 0])

        df = run(rocket)

        fall_time = np.sqrt(2 * height / const.G0)
        last = df.iloc[-1]
        # The run stops at the first step that ends below the ground.
        assert last["altitude"] <= 0.0
        assert fall_time <= last["time"] <= fall_time + simplified_params["max_step"]
        assert last["position cartesian x"] == pytest.approx(speed_x * last["time"])
        assert last["velocity cartesian z"] == pytest.approx(-const.G0 * last["time"])

    def test_terminal_velocity(self, simplified_params, make_rocket):
        """With constant density the fall settles at sqrt(2 m g / (rho Cd A))."""
        # Same drag with and without parachute, so the opening altitude has
        # no effect.
        simplified_params.update(
            drag_coefficient_parachute=simplified_params["drag_coefficient"],
            cross_sectional_area_parachute=simplified_params["cross_sectional_area"],
        )
        rocket = make_rocket(simplified_params, [0, 0, 5000], [0, 0, 0])

        df = run(rocket)

        terminal_velocity = np.sqrt(
            2
            * simplified_params["mass"]
            * const.G0
            / (
                const.SEA_LEVEL_AIR_DENSITY
                * simplified_params["drag_coefficient"]
                * simplified_params["cross_sectional_area"]
            )
        )
        assert df["velocity cartesian z"].iloc[-1] == pytest.approx(
            -terminal_velocity, rel=1e-3
        )

    def test_altitude_is_logged_on_the_flat_earth(self, simplified_params, make_rocket):
        rocket = make_rocket(simplified_params, [0, 0, 2000], [0, 0, 0])

        df = run(rocket)

        assert df["altitude"].to_numpy() == pytest.approx(
            df["position cartesian z"].to_numpy()
        )
        assert df["altitude"].iloc[0] == pytest.approx(2000.0, abs=1.0)
        assert df["position geodetic lat"].isna().all()


class TestRealisticMode:
    def test_parachute_descent_reaches_the_ground(self, params, make_rocket):
        """Dropped below the opening altitude, the rocket lands at the
        parachute's terminal velocity."""
        start = [ERA5_LAT, ERA5_LON, 3000.0]
        rocket = make_rocket(params, start, [0, 0, 0])

        df = run(rocket)

        altitude = df["position geodetic alt"].to_numpy()
        assert altitude[-1] <= 0.0
        assert altitude[-2] > 0.0
        assert np.all(np.diff(altitude) < 0.0)
        assert np.all(np.diff(df["time"]) > 0.0)

        gravity = 9.80  # effective gravity at this latitude, m/s^2
        terminal_velocity = np.sqrt(
            2
            * params["mass"]
            * gravity
            / (
                const.SEA_LEVEL_AIR_DENSITY
                * params["drag_coefficient_parachute"]
                * params["cross_sectional_area_parachute"]
            )
        )
        assert df["velocity geodetic alt"].iloc[-1] == pytest.approx(
            -terminal_velocity, rel=1e-2
        )
        # Without wind the rocket lands where it was dropped.
        assert df["position geodetic lat"].iloc[-1] == pytest.approx(start[0], abs=1e-3)
        assert df["position geodetic lon"].iloc[-1] == pytest.approx(
            start[1] - 360.0, abs=1e-3
        )

    def test_rocket_drifts_with_the_wind(self, params, make_rocket):
        """Under the parachute the rocket picks up the wind velocity."""
        rocket = make_rocket(
            params,
            [ERA5_LAT, ERA5_LON, 3000.0],
            [0, 0, 0],
            wind_field_conditions=[270.0, 15.0],  # from the west
        )

        df = run(rocket)

        assert df["velocity geodetic East"].iloc[-1] == pytest.approx(15.0, abs=0.1)
        assert df["velocity geodetic North"].iloc[-1] == pytest.approx(0.0, abs=0.1)
        assert (
            df["position geodetic lon"].iloc[-1] > df["position geodetic lon"].iloc[0]
        )

    def test_run_stops_at_tmax(self, params, make_rocket):
        params["tmax"] = 5.0
        rocket = make_rocket(params, [ERA5_LAT, ERA5_LON, 3000.0], [0, 0, 0])

        df = run(rocket)

        assert df["time"].iloc[-1] == pytest.approx(5.0)
        assert df["position geodetic alt"].iloc[-1] > 0.0

    def test_exhausted_step_budget_warns(self, params, make_rocket):
        rocket = make_rocket(params, [ERA5_LAT, ERA5_LON, 3000.0], [0, 0, 0], n_steps=3)

        with pytest.warns(UserWarning, match="before reaching the ground"):
            df = run(rocket, n_steps=3)

        assert len(df) == 3

    @pytest.mark.usefixtures("in_src_dir")
    # Below the lowest ERA5 model level the wind falls back to calm.
    @pytest.mark.filterwarnings("ignore:Interpolation failed")
    def test_era5_run(self, params, make_rocket):
        """A short run with the ERA5 sample data uses the interpolated wind."""
        params.update(wind="ERA5", max_step=0.5)
        rocket = make_rocket(params, [ERA5_LAT, ERA5_LON, 1000.0], [0, 0, 0])

        df = run(rocket)

        assert rocket.params["wind"] == "ERA5"
        assert df["position geodetic alt"].iloc[-1] <= 0.0
        first_wind = df["wind_velocity"].iloc[0]
        assert np.linalg.norm(first_wind[:2]) > 0.0


class TestParameterChecks:
    def test_mass_must_be_positive(self, params, make_rocket):
        params["mass"] = 0.0

        with pytest.raises(ValueError, match="mass must be positive"):
            make_rocket(params, [ERA5_LAT, ERA5_LON, 3000.0], [0, 0, 0])

    def test_only_rk45_is_implemented(self, params, make_rocket):
        params["solver_method"] = "DOP853"

        with pytest.raises(NotImplementedError, match="RK45"):
            make_rocket(params, [ERA5_LAT, ERA5_LON, 3000.0], [0, 0, 0])
