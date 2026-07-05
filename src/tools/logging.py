import numpy as np
import pandas as pd

from .utils import altitude_to_geopotential


# potentially upgrade type hints to numpy.typing.NDArray[np.float64] for better clarity and type checking
class Logging:
    def __init__(self, n_steps: int) -> None:
        self.i = 0

        # Scalars
        self.time = np.full(n_steps, np.nan)
        self.altitude = np.full(n_steps, np.nan)
        self.drag_force = np.full(n_steps, np.nan)
        self.air_density = np.full(n_steps, np.nan)
        self.grav_acc_norm = np.full(n_steps, np.nan)
        self.geopotential = np.full(n_steps, np.nan)

        # Cartesian vectors
        self.position = np.full((n_steps, 3), np.nan)
        self.velocity = np.full((n_steps, 3), np.nan)

        # Geodetic vectors
        self.position_geodetic = np.full((n_steps, 3), np.nan)
        self.velocity_geodetic = np.full((n_steps, 3), np.nan)

        # Wind
        self.wind_velocity = np.full((n_steps, 3), np.nan)

        # Variable-length information
        self.warnings = [[] for _ in range(n_steps)]

    def log_state(self, rocket: object, t: float) -> None:
        i = self.i

        self.time[i] = t

        self.position[i] = rocket.get_position_cartesian()
        self.velocity[i] = rocket.get_velocity_cartesian()
        altitude = rocket.calc_altitude_abv_sea_level()
        self.altitude[i] = altitude

        if (
            rocket.params.get("mode") == "realistic"
            or rocket.params.get("wind") == "ERA5"
        ):
            self.position_geodetic[i] = rocket.get_position_geodetic()
            self.velocity_geodetic[i] = rocket.get_velocity_geodetic()
            self.geopotential[i] = altitude_to_geopotential(altitude)

        self.drag_force[i] = np.linalg.norm(rocket.get_drag_force())
        self.air_density[i] = rocket.get_air_density()
        self.grav_acc_norm[i] = np.linalg.norm(rocket.grav_acc())

        self.i += 1

    def log_wind(self, wind_velv: np.ndarray) -> None:
        # log wind for the most recently logged timestep
        self.wind_velocity[self.i - 1] = wind_velv

    def log_warning(self, message: str) -> None:
        self.warnings[self.i - 1].append(message)

    def get_full_state(self) -> pd.DataFrame:
        n = self.i

        return pd.DataFrame(
            {
                "time": self.time[:n],
                "position cartesian x": self.position[:n, 0],
                "position cartesian y": self.position[:n, 1],
                "position cartesian z": self.position[:n, 2],
                "velocity cartesian x": self.velocity[:n, 0],
                "velocity cartesian y": self.velocity[:n, 1],
                "velocity cartesian z": self.velocity[:n, 2],
                "position geodetic lat": self.position_geodetic[:n, 0],
                "position geodetic lon": self.position_geodetic[:n, 1],
                "position geodetic alt": self.position_geodetic[:n, 2],
                "velocity geodetic East": self.velocity_geodetic[:n, 0],
                "velocity geodetic North": self.velocity_geodetic[:n, 1],
                "velocity geodetic alt": self.velocity_geodetic[:n, 2],
                "altitude": self.altitude[:n],
                "drag_force": self.drag_force[:n],
                "air_density": self.air_density[:n],
                "grav_acc norm": self.grav_acc_norm[:n],
                "geopotential": self.geopotential[:n],
                "wind_velocity": list(self.wind_velocity[:n]),
                "warnings": self.warnings[:n],
            }
        )

    def get_last_log_entry(self) -> dict | None:
        if self.i == 0:
            return None

        i = self.i - 1

        return {
            "time": self.time[i],
            "position cartesian x": self.position[i, 0],
            "position cartesian y": self.position[i, 1],
            "position cartesian z": self.position[i, 2],
            "velocity cartesian x": self.velocity[i, 0],
            "velocity cartesian y": self.velocity[i, 1],
            "velocity cartesian z": self.velocity[i, 2],
            "position geodetic lat": self.position_geodetic[i, 0],
            "position geodetic lon": self.position_geodetic[i, 1],
            "position geodetic alt": self.position_geodetic[i, 2],
            "velocity geodetic East": self.velocity_geodetic[i, 0],
            "velocity geodetic North": self.velocity_geodetic[i, 1],
            "velocity geodetic alt": self.velocity_geodetic[i, 2],
            "altitude": self.altitude[i],
            "drag_force": self.drag_force[i],
            "air_density": self.air_density[i],
            "grav_acc norm": self.grav_acc_norm[i],
            "geopotential": self.geopotential[i],
            "wind_velocity": self.wind_velocity[i],
            "warnings": self.warnings[i],
        }
