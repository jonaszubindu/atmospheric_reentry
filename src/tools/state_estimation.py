import numpy as np
import pymap3d as pm
import numpy.typing as npt


class StateEstimation:
    """A class for estimating the state of the rocket during simulation and keeping track of the coordinate system used for position and velocity."""

    def __init__(self) -> None:
        """Initialize the state estimation with default values."""
        self.tcurrent: float = 0  # always start at t zero for now
        self.state_cartesian: npt.NDArray[np.float64] = np.zeros(
            6, dtype=np.float64
        )  # [3 position 3 velocity]

    def estimate_state(self, tcurrent: float) -> None:
        """Simple state estimation function that could be expanded to
        include sensor noise and filtering.

        For now, just update the current time at step, but this could be
        modified to include noise and a filter.
        Parameters:
        tcurrent (float): The current time step of the simulation.
        """
        self.tcurrent = tcurrent

    def get_position_cartesian(self) -> npt.NDArray[np.float64]:
        """Get the current position of the rocket."""
        return self.state_cartesian[:3]

    def get_velocity_cartesian(self) -> npt.NDArray[np.float64]:
        """Get the current velocity of the rocket."""
        return self.state_cartesian[3:]

    def set_position_cartesian(self, new_position: npt.NDArray[np.float64]) -> None:
        """Set the current position of the rocket."""
        self.state_cartesian[:3] = np.asarray(new_position, dtype=np.float64)

    def set_velocity_cartesian(self, new_velocity: npt.NDArray[np.float64]) -> None:
        """Set the current velocity of the rocket."""
        self.state_cartesian[3:] = np.asarray(new_velocity, dtype=np.float64)

    def get_position_geodetic(self) -> npt.NDArray[np.float64]:
        """Get the current position of the rocket in geodetic coordinates."""
        return (
            self._cartesian_to_geodetic()
        )  # lazily update geodetic position from cartesian position

    def get_velocity_geodetic(self) -> npt.NDArray[np.float64]:
        """Get the current velocity of the rocket in geodetic coordinates."""
        return (
            self._cartesian_to_geodetic_velocity()
        )  # lazily update geodetic velocity from cartesian velocity

    # def _set_position_geodetic(self, new_position: npt.NDArray[np.float64]) -> None:
    #     """Set the current position of the rocket in geodetic coordinates."""
    #     self.state_geodetic[:3] = np.asarray(new_position, dtype=np.float64)

    # def _set_velocity_geodetic(self, new_velocity: npt.NDArray[np.float64]) -> None:
    #     """Set the current velocity of the rocket in geodetic coordinates."""
    #     self.state_geodetic[3:] = np.asarray(new_velocity, dtype=np.float64)

    def _cartesian_to_geodetic(self) -> None:
        """Convert cartesian coordinates (x, y, z) to geodetic coordinates
        (latitude, longitude, altitude)."""
        position = self.get_position_cartesian()
        return np.asarray(pm.ecef2geodetic(position[0], position[1], position[2]))

    # def _geodetic_to_cartesian(self) -> None:
    #     """Convert geodetic coordinates (latitude, longitude, altitude) to
    #     cartesian coordinates (x, y, z)."""
    #     position = self.get_position_geodetic()
    #     self.set_position_cartesian(self.convert_geodetic_to_cartesian(position))

    def _cartesian_to_geodetic_velocity(self) -> None:
        """Convert cartesian velocity (vx, vy, vz) to geodetic velocity
        (v_lat, v_lon, v_alt)."""
        position = self.get_position_geodetic()
        velocity = self.get_velocity_cartesian()
        lat, lon, _ = position
        return np.asarray(pm.ecef2enuv(velocity[0], velocity[1], velocity[2], lat, lon))

    # def _geodetic_to_cartesian_velocity(self) -> None:
    #     """Convert geodetic velocity (v_lat, v_lon, v_alt) to cartesian
    #     velocity (vx, vy, vz). Assumes velocity is in ENU (East, North, Up)
    #     coordinates for geodetic velocity and we want to convert it to ECEF
    #     coordinates for cartesian velocity."""
    #     position = self.get_position_geodetic()
    #     velocity = self.get_velocity_geodetic()
    #     lat, lon, _ = position
    #     self.set_velocity_cartesian(
    #         self.convert_velocity_geodetic_to_cartesian(velocity, lat, lon)
    #     )

    @staticmethod
    def convert_velocity_geodetic_to_cartesian(
        velocity_geodetic: npt.NDArray[np.float64], lat: float, lon: float
    ) -> npt.NDArray[np.float64]:
        """Convert geodetic velocity (v_lat, v_lon, v_alt) to cartesian
        velocity (vx, vy, vz). Assumes velocity is in ENU (East, North, Up)
        coordinates for geodetic velocity and we want to convert it to ECEF
        coordinates for cartesian velocity."""
        return np.asarray(
            pm.enu2ecefv(
                velocity_geodetic[0],
                velocity_geodetic[1],
                velocity_geodetic[2],
                lat,
                lon,
                deg=True,
            )
        )

    @staticmethod
    def convert_geodetic_to_cartesian(
        position_geodetic: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Convert geodetic coordinates (latitude, longitude, altitude) to cartesian
        coordinates (x, y, z)."""
        lat, lon, alt = position_geodetic
        return np.asarray(pm.geodetic2ecef(lat, lon, alt))


# def _update_full_state(self):
#     """Update the state from cartesian to geodetic coordinates."""
#     self._cartesian_to_geodetic()
#     self._cartesian_to_geodetic_velocity()
