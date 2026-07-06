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
from .state_estimation import RealState


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

    def get_wind(
        self,
        t: float,
        state: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Update the wind velocity vector based on the specified wind model.

        This method is a placeholder and will be replaced by either
        update_wind_default or update_wind_model depending on the wind data
        source specified in params during initialization.
        Parameters:
        t (float): The current time in seconds since the start of the
            simulation.
        state (npt.NDArray[np.float64]): The current state of the rocket, including position and velocity.
        Returns: npt.NDArray[np.float64]: The wind velocity vector in the simulation coordinate system.
            in case of default wind model, the wind velocity vector is in cartesian coordinates.
            in case of ERA5 wind model, the wind velocity vector is in ENU coordinates and needs to be transformed to cartesian coordinates using the rocket's current position.
        """
        raise NotImplementedError(
            "This method should be replaced by a specific wind update function."
        )

    def _verify_position_within_bounds(
        self, lat: float, lon: float, alt: float
    ) -> None:
        """Verify that the given position is within the bounds of the wind data.

        Parameters:
        lat (float): Latitude in degrees.
        lon (float): Longitude in degrees.
        alt (float): Altitude in meters.
        """
        if (
            lat < self.ymin
            or lat > self.ymax
            or lon < self.xmin
            or lon > self.xmax
            or alt < self.zmin_alt
            or alt > self.zmax_alt
        ):
            message_nowind = (
                f"WARNING: WIND NOT USED! Position (lat: {lat}, lon: {lon}, alt: {alt}) is outside the "
                "bounds of the wind data. Wind velocity will be set to zero."
            )
            self._wind_failure(message_nowind)
            self.get_wind = self._update_wind_default
            self.wind_field_conditions = [
                0.0,
                0.0,
            ]  # Set default wind conditions to zero
            self.params["wind"] = "default"

    def __init__(
        self,
        start_time: str,
        params: dict,
        logger: Logging | None = None,
        wind_field_conditions: list[
            npt.NDArray[np.float64], npt.NDArray[np.float64]
        ] = [0.0, 0.0],
    ) -> None:
        """Initialize the WindField with the specified start time and parameters.
        Parameters:
        start_time (str): The start time of the simulation in a format
            recognized by pandas.to_datetime.
        params (dict): Dictionary containing simulation parameters, including
            wind data source.
        logger (Logging, optional): Logger instance for logging warnings and messages.
        wind_field_conditions (list, optional): List containing wind direction and wind velocity for constant wind conditions.
        Defaults to [0.0, 0.0].
        """
        self.logger = logger
        self.params = params
        if self.params.get("wind") != "ERA5":
            # For consistency, initialize u, v, z to None
            self.u = None
            self.v = None
            self.z = None
            self.start_time = start_time
            self.wind_field_conditions = wind_field_conditions
            self.get_wind = self._update_wind_default
            self.wind_velv = np.zeros(3)  # Default wind velocity vector is zero
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
            self.wind_field_conditions = None
            self.get_wind = self._update_wind_model
            self.wind_velv = np.zeros(3)  # Initialize wind velocity vector to zero

            # At init verify position is within the bounds of the wind data, otherwise warn and set wind to zero.
            self.zmax = self.z.z.max().values
            self.zmin = self.z.z.min().values
            self.zmax_alt = geopotential_to_altitude(self.zmax)
            self.zmin_alt = geopotential_to_altitude(self.zmin)
            self.xmin = self.u.longitude.min().values
            self.xmax = self.u.longitude.max().values
            self.ymin = self.u.latitude.min().values
            self.ymax = self.u.latitude.max().values
            self.tmin = self.u.time.min().values
            self.tmax = self.u.time.max().values

            self._verify_position_within_bounds(
                lat=logger.initial_conditions[0][0],
                lon=logger.initial_conditions[0][1],
                alt=logger.initial_conditions[0][2],
            )

    def _wind_failure(self, message):
        warnings.warn(message, UserWarning)
        if self.logger is not None:
            self.logger.log_warning(message)
        self.wind_velv = np.zeros(3)
        return None

    def _interpolate_wind(self, dataset, variable, teval, lat, lon, lvl_interp):
        """Interpolate the wind data for a specific variable (u or v) at a
        specific time, latitude, longitude, and level.
        Parameters:
        dataset (xarray.Dataset): The wind dataset (u or v).
        variable (str): The variable name ('u' or 'v').
        teval (pd.Timestamp): The time at which to interpolate the wind data.
        lat (float): Latitude in degrees.
        lon (float): Longitude in degrees.
        lvl_interp (float): The hybrid level at which to interpolate the wind data.
        Returns: float: The interpolated wind velocity component.
        """
        if (
            teval < self.tmin
            or teval > self.tmax
            or lvl_interp < 1
            or lvl_interp > 137
            or lat < self.ymin  # lat - y
            or lat > self.ymax
            or lon < self.xmin  # lon - x
            or lon > self.xmax
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

    def _compute_wind_at_altitude(self, t, lat, lon, alt):
        """Get the wind velocity vector at a specific altitude, latitude,
        and longitude at time t, with interpolation of the ERA5 wind
        data.
        Parameters:
        t (float): The current time in seconds since the start of the
            simulation.
        lat (float): Latitude in degrees.
        lon (float): Longitude in degrees.
        alt (float): Altitude in meters.
        Returns: numpy.ndarray: The wind velocity vector in the simulation coordinate system.
        """
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
        return wind_velv

    def _update_wind_model(
        self,
        t: float,
        state: npt.NDArray[np.float64],
    ):
        """Update the wind velocity vector based on the ERA5 wind model.

        Parameters:
        t (float): The current time in seconds since the start of the
            simulation.
        state (array): The current state of the rocket, including position and velocity.

        Returns: numpy.ndarray: The wind velocity vector in the simulation coordinate system.
        """
        # Convert x, y, z to latitude, longitude, altitude
        position_geodetic = RealState.convert_cartesian_to_geodetic(state[:3])
        lat, lon, alt = position_geodetic
        # u is eastward - x direction, v is northward - y direction, matching
        # the simulation coordinate system.
        return self._compute_wind_at_altitude(t=t, lat=lat, lon=lon, alt=alt)

    def _update_wind_default(
        self,
        t: float,
        state: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Update the wind velocity vector based on default wind conditions,
        ignoring position and time.

        Default wind conditions: 0 m/s wind speed from 0 degrees (north to
        south). Unused arguments are kept for compatibility.
        Parameters:
        t (float): The current time in seconds since the start of the
            simulation. (Unused)
        state (array): The current state of the rocket, including position and velocity. (Unused)
        Returns: numpy.ndarray: The wind velocity vector in the simulation coordinate system.
        """
        wind_dir, wind_vel = self.wind_field_conditions
        wind_dir_cartesian = _get_cartesian_wind(wind_dir)
        wind_vel_x = wind_vel * np.sin(np.radians(wind_dir_cartesian))
        wind_vel_y = wind_vel * np.cos(np.radians(wind_dir_cartesian))
        # Assuming no vertical wind component for simplicity
        wind_velv = np.array([wind_vel_x, wind_vel_y, 0])

        self.wind_velv = wind_velv
        return wind_velv
