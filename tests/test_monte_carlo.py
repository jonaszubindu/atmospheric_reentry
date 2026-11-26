"""Tests of the parallel Monte Carlo ensemble (tools.monte_carlo)."""

import numpy as np
import pytest

from tools.monte_carlo import run_monte_carlo, sample_uniform_wind

# Short drop under an open parachute (opening altitude 10 km > 2 km), so
# each trajectory takes ~100 s of simulated time and runs quickly.
LAT, LON, ALT = 46.95, 7.45, 2000.0
KM_PER_DEG_LAT = 111.32


def ground_offset_km(df):
    """East and north offset of the impact point from the drop point (km)."""
    last = df.iloc[-1]
    east = (
        (last["position geodetic lon"] - LON)
        * KM_PER_DEG_LAT
        * np.cos(np.radians(LAT))
    )
    north = (last["position geodetic lat"] - LAT) * KM_PER_DEG_LAT
    return east, north


def test_sample_uniform_wind_ranges_and_seed():
    winds = sample_uniform_wind(500, (0.0, 360.0), (0.0, 20.0), seed=7)
    assert winds.shape == (500, 2)
    assert np.all((winds[:, 0] >= 0.0) & (winds[:, 0] < 360.0))
    assert np.all((winds[:, 1] >= 0.0) & (winds[:, 1] < 20.0))
    np.testing.assert_array_equal(
        winds, sample_uniform_wind(500, (0.0, 360.0), (0.0, 20.0), seed=7)
    )
    assert not np.array_equal(
        winds, sample_uniform_wind(500, (0.0, 360.0), (0.0, 20.0), seed=8)
    )


def test_wind_from_the_west_drifts_east(params):
    """Calm wind lands below the drop point; a wind blowing from 270 deg
    (west) carries the descent to the east."""
    trajectories, winds = run_monte_carlo(
        params,
        [LAT, LON, ALT],
        [0.0, 0.0, 0.0],
        wind_samples=[[0.0, 0.0], [270.0, 10.0]],
        n_jobs=1,
    )
    assert len(trajectories) == 2
    for df in trajectories:
        assert df.iloc[-1]["altitude"] <= 0.0  # every run reaches the ground

    calm_east, calm_north = ground_offset_km(trajectories[0])
    assert abs(calm_east) < 0.05 and abs(calm_north) < 0.05

    windy_east, windy_north = ground_offset_km(trajectories[1])
    assert windy_east > 0.5  # ~10 m/s for ~100 s
    assert abs(windy_north) < 0.1 * windy_east


def test_parallel_matches_sequential(params):
    """Workers build their own rocket, so results must not depend on n_jobs."""
    kwargs = dict(n_samples=3, seed=3)
    sequential, winds_seq = run_monte_carlo(
        params, [LAT, LON, ALT], [0.0, 0.0, 0.0], n_jobs=1, **kwargs
    )
    parallel, winds_par = run_monte_carlo(
        params, [LAT, LON, ALT], [0.0, 0.0, 0.0], n_jobs=2, **kwargs
    )
    np.testing.assert_array_equal(winds_seq, winds_par)
    for a, b in zip(sequential, parallel):
        np.testing.assert_allclose(
            a.iloc[-1][["position geodetic lat", "position geodetic lon"]].to_numpy(
                dtype=float
            ),
            b.iloc[-1][["position geodetic lat", "position geodetic lon"]].to_numpy(
                dtype=float
            ),
        )


def test_era5_wind_is_rejected(params):
    params["wind"] = "ERA5"
    with pytest.raises(ValueError, match="ERA5"):
        run_monte_carlo(params, [LAT, LON, ALT], [0.0, 0.0, 0.0], n_samples=1)
