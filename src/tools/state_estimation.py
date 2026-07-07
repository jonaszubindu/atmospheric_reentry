import numpy as np
import pymap3d as pm
import numpy.typing as npt


class RealState:
    """A class for estimating the state of the rocket during simulation and keeping track of the coordinate system used for position and velocity."""

    def __init__(self) -> None:
        """Initialize the state estimation with default values."""
        self.time: float = 0.0  # Current time in seconds
        self.state_cartesian: npt.NDArray[np.float64] = np.zeros(
            6, dtype=np.float64
        )  # [3 position 3 velocity]

    def update_time(self, tcurrent: float) -> None:
        """Update the time of the state estimation."""
        self.time = tcurrent

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

    def _cartesian_to_geodetic(self) -> npt.NDArray[np.float64]:
        """Convert cartesian coordinates (x, y, z) to geodetic coordinates
        (latitude, longitude, altitude)."""
        position = self.get_position_cartesian()
        return np.asarray(pm.ecef2geodetic(position[0], position[1], position[2]))

    def _cartesian_to_geodetic_velocity(self) -> npt.NDArray[np.float64]:
        """Convert cartesian velocity (vx, vy, vz) to geodetic velocity
        (v_east, v_north, v_up)."""
        position = self.get_position_geodetic()
        velocity = self.get_velocity_cartesian()
        lat, lon, _ = position
        return np.asarray(pm.ecef2enuv(velocity[0], velocity[1], velocity[2], lat, lon))

    @staticmethod
    def convert_velocity_geodetic_to_cartesian(
        velocity_geodetic: npt.NDArray[np.float64], lat: float, lon: float
    ) -> npt.NDArray[np.float64]:
        """Convert geodetic velocity (v_east, v_north, v_up) to cartesian
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
    def convert_cartesian_to_geodetic(
        position_cartesian: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Convert cartesian coordinates (x, y, z) to geodetic coordinates
        (latitude, longitude, altitude)."""
        x, y, z = position_cartesian
        return np.asarray(pm.ecef2geodetic(x, y, z))

    @staticmethod
    def convert_velocity_cartesian_to_geodetic(
        velocity_cartesian: npt.NDArray[np.float64], lat: float, lon: float
    ) -> npt.NDArray[np.float64]:
        """Convert cartesian velocity (vx, vy, vz) to geodetic velocity
        (v_east, v_north, v_up). Assumes velocity is in ECEF coordinates for
        cartesian velocity and we want to convert it to ENU (East, North, Up)
        coordinates for geodetic velocity."""
        return np.asarray(
            pm.ecef2enuv(
                velocity_cartesian[0],
                velocity_cartesian[1],
                velocity_cartesian[2],
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
