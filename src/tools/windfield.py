import warnings

import numpy as np
from .logging import Logging
import xarray as xr
import pandas as pd
import cfgrib
from scipy.interpolate import interp1d
import numpy.typing as npt

from .utils import (
    altitude_to_geopotential,
    geopotential_to_altitude,
    _get_cartesian_wind,
)


class WindField:
    """
    Init wind data - These datasets need to be downloaded for the specific
    date, area and time range using the files get_winddata.py and the
    geopotential is converted using compute_geopotential_on_ml.py.
    Otherwise it uses the last obtained datasets.

    Parameters:
    start_time (str): The start time of the simulation in a format
        recognized by pandas.to_datetime.
    params (dict): Dictionary containing simulation parameters, including
        wind data source.
    """

    def update_wind(
        self,
        t: float,
        position_geodetic: npt.NDArray[np.float64],
        wind_field_conditions: list = None,
    ) -> npt.NDArray[np.float64]:
        """Update the wind velocity vector based on the specified wind model.

        This method is a placeholder and will be replaced by either
        update_wind_default or update_wind_model depending on the wind data
        source specified in params during initialization.
        Parameters:
        t (float): The current time in seconds since the start of the
            simulation.
        position_geodetic (npt.NDArray[np.float64]): The current position of the rocket in
            geodetic coordinates (latitude, longitude, altitude).
        wind_field_conditions (list, optional): Default wind conditions
            [wind_direction, wind_speed] if using the default wind model.
        Returns: npt.NDArray[np.float64]: The wind velocity vector in the simulation coordinate system.
            in case of default wind model, the wind velocity vector is in cartesian coordinates.
            in case of ERA5 wind model, the wind velocity vector is in ENU coordinates and needs to be transformed to cartesian coordinates using the rocket's current position.
        """
        raise NotImplementedError(
            "This method should be replaced by a specific wind update function."
        )

    def __init__(self, start_time: str, params: dict, logger: Logging) -> None:
        self.logger = logger
        if params.get("wind") != "ERA5":
            self.u = 0  # eastward wind
            self.v = 0  # northward wind
            self.z = 0  # alt

            self.update_wind = self._update_wind_default
        else:
            # eastward wind
            u = xr.open_dataset(
                "./tools/windmodel_data/era5_ml.grib",
                engine="cfgrib",
                backend_kwargs={
                    "filter_by_keys": {"paramId": 131, "typeOfLevel": "hybrid"}
                },
            )

            # northward wind
            v = xr.open_dataset(
                "./tools/windmodel_data/era5_ml.grib",
                engine="cfgrib",
                backend_kwargs={
                    "filter_by_keys": {"paramId": 132, "typeOfLevel": "hybrid"}
                },
            )
            # geopotential for altitude interpolation
            z = cfgrib.open_dataset(
                "./tools/windmodel_data/z_out.grib",
                backend_kwargs={"filter_by_keys": {"typeOfLevel": "hybrid"}},
            )
            print(
                "Wind data used from: ",
                u.time.values,
                "at latitude: ",
                u.latitude.values,
                "and longitude: ",
                u.longitude.values,
            )

            self.u = u
            self.v = v
            self.z = z
            self.start_time = start_time
            self.update_wind = self._update_wind_model

            # These numbers should be usable to make sure we are staying within
            # the interpolation range before the code fails.
            # For some reason comparing to zmin and zmax did not work though.
            # Therefore the error is now just caught and a warning is issued,
            # and the wind velocity is set to zero.
            self.zmax = self.z.z.max().values
            self.zmin = self.z.z.min().values
            self.zmax_alt = geopotential_to_altitude(self.zmax)
            self.zmin_alt = geopotential_to_altitude(self.zmin)
            self.logger = logger

    def _wind_failure(self, message):
        warnings.warn(message, UserWarning)
        if self.logger is not None:
            self.logger.log_warning(message)
        self.wind_velv = np.zeros(3)
        return None

    def _interpolate_wind(self, dataset, variable, teval, lat, lon, lvl_interp):
        """Interpolate the wind data for a specific variable (u or v) at a
        specific time, latitude, longitude, and level."""
        if (
            teval < dataset.time.min().values
            or teval > dataset.time.max().values
            or lvl_interp < 1
            or lvl_interp > 137
            or lat < dataset.latitude.min().values
            or lat > dataset.latitude.max().values
            or lon < dataset.longitude.min().values
            or lon > dataset.longitude.max().values
        ):
            return self._wind_failure(
                f"Interpolation failed for {variable} at time {teval}, "
                f"lat {lat}, lon {lon}, hybrid level {lvl_interp}. "
                "Returning zero wind velocity."
            )
        else:
            return dataset.interp(
                time=teval, latitude=lat, longitude=lon, hybrid=lvl_interp
            )[variable].values

    def _get_wind_at_altitude(self, t, lat, lon, alt):
        """Get the wind velocity vector at a specific altitude, latitude,
        and longitude at time t, with interpolation of the ERA5 wind
        data."""
        # Get the correct parameters and locations at which to
        # interpolate the wind model
        teval = pd.to_datetime(self.start_time) + pd.to_timedelta(t, unit="s")
        lon = (lon + 360) % 360  # Ensure longitude is in [0, 360) degrees
        geopotential = altitude_to_geopotential(alt)

        try:
            zpos = self.z.interp(latitude=lat, longitude=lon)

            lvls = np.arange(1, 138)
            lvl_interp = interp1d(zpos.z.values, lvls)(geopotential)
        except ValueError as e:
            return self._wind_failure(
                f"Interpolation failed for geopotential at time {teval}, "
                f"lat {lat}, lon {lon}, alt {alt}: {e}. "
                "Returning zero wind velocity."
            )

        # eastward_wind
        u_interp = self._interpolate_wind(self.u, "u", teval, lat, lon, lvl_interp)
        v_interp = self._interpolate_wind(self.v, "v", teval, lat, lon, lvl_interp)

        wind_velv = np.array([u_interp, v_interp, 0])
        # Replace NaN values with 0, which can occur if the rocket goes
        # above the maximum altitude of the wind data or if the
        # interpolation fails for some reason. This is a simple way to
        # handle missing data.
        wind_velv[np.isnan(wind_velv)] = 0
        # Assuming no vertical wind component for simplicity
        self.wind_velv = wind_velv

    def _update_wind_model(self, t, position_geodetic, wind_field_conditions=None):
        """Update the wind velocity vector based on the ERA5 wind model.

        wind_field_conditions is not used for the model, but included for
        compatibility with the default update function.
        Parameters:
        t (float): The current time in seconds since the start of the
            simulation.
        position_geodetic (array): The current position of the rocket in
            geodetic coordinates (latitude, longitude, altitude).

        Returns: numpy.ndarray: The wind velocity vector in the simulation coordinate system.
        """
        # Convert x, y, z to latitude, longitude, altitude
        lat, lon, alt = position_geodetic
        # u is eastward - x direction, v is northward - y direction, matching
        # the simulation coordinate system.
        return self._get_wind_at_altitude(t=t, lat=lat, lon=lon, alt=alt)

    def _update_wind_default(
        self,
        t: float,
        position: npt.NDArray[np.float64],
        wind_field_conditions: list[np.float64, np.float64] = [270.0, 15.0],
    ) -> npt.NDArray[np.float64]:
        """Update the wind velocity vector based on default wind conditions,
        ignoring position and time.

        Default wind conditions: 15 m/s wind speed from 270 degrees (north to
        south). Unused arguments are kept for compatibility.
        """
        wind_dir, wind_vel = wind_field_conditions
        wind_dir_cartesian = _get_cartesian_wind(wind_dir)
        wind_vel_x = wind_vel * np.sin(np.radians(wind_dir_cartesian))
        wind_vel_y = wind_vel * np.cos(np.radians(wind_dir_cartesian))
        # Assuming no vertical wind component for simplicity
        wind_velv = np.array([wind_vel_x, wind_vel_y, 0])

        self.wind_velv = wind_velv
