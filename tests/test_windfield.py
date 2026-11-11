"""Tests for the analytic default wind and the ERA5 wind model."""

import numpy as np
import pytest

from tools.logging import Logging
from tools.windfield import WindField

from conftest import ERA5_LAT, ERA5_LON, ERA5_START_TIME


@pytest.mark.parametrize(
    "wind_from_deg, expected_wind",
    [
        (0.0, [0.0, -15.0, 0.0]),  # from north, blowing south
        (90.0, [-15.0, 0.0, 0.0]),  # from east, blowing west
        (180.0, [0.0, 15.0, 0.0]),  # from south, blowing north
        (270.0, [15.0, 0.0, 0.0]),  # from west, blowing east
    ],
)
def test_default_wind_uses_aviation_convention(params, wind_from_deg, expected_wind):
    """The direction is where the wind comes from; x is east, y is north."""
    wind_field = WindField(
        start_time=ERA5_START_TIME,
        params=params,
        wind_field_conditions=[wind_from_deg, 15.0],
    )

    wind = wind_field.get_wind(0.0, np.zeros(3))

    assert wind == pytest.approx(expected_wind, abs=1e-9)


def test_default_wind_is_calm_without_conditions(params):
    wind_field = WindField(start_time=ERA5_START_TIME, params=params)

    assert wind_field.get_wind(0.0, np.zeros(3)) == pytest.approx(np.zeros(3))


def era5_wind_field(params: dict, position_geodetic: list[float]) -> WindField:
    params["wind"] = "ERA5"
    logger = Logging(
        10, initial_conditions_geodetic=[np.array(position_geodetic), np.zeros(3)]
    )
    return WindField(start_time=ERA5_START_TIME, params=params, logger=logger)


@pytest.mark.usefixtures("in_src_dir")
def test_era5_wind_inside_the_data(params):
    position = [ERA5_LAT, ERA5_LON, 5000.0]

    wind_field = era5_wind_field(params, position)
    wind = wind_field.get_wind(0.0, np.array(position))

    assert params["wind"] == "ERA5"
    assert np.all(np.isfinite(wind))
    assert np.linalg.norm(wind[:2]) > 0.0
    assert wind[2] == 0.0  # no vertical wind in the model


@pytest.mark.usefixtures("in_src_dir")
def test_era5_wind_outside_the_data_falls_back_to_calm(params):
    position = [47.3986, 8.6486, 5000.0]  # Dübendorf, not in the sample data

    with pytest.warns(UserWarning, match="WIND NOT USED"):
        wind_field = era5_wind_field(params, position)

    assert params["wind"] == "default"
    assert params["mode"] == "realistic"  # only the wind falls back
    assert wind_field.get_wind(0.0, np.array(position)) == pytest.approx(np.zeros(3))


def test_era5_wind_is_rejected_in_simplified_mode(params):
    params.update(wind="ERA5", mode="simplified")

    with pytest.raises(ValueError, match="ERA5 wind needs the realistic mode"):
        WindField(start_time=ERA5_START_TIME, params=params)
