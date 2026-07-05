#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Optional
import numpy.typing as npt

from .state_estimation import StateEstimation

from . import constants as const


class PhysicsFunctions(StateEstimation):
    """A class containing various physics functions used in the rocket
    simulation. This class is designed to be flexible and can be extended
    with additional physics functions as needed."""

    def __init__(self) -> None:
        """Initialize the PhysicsFunctions class."""
        super().__init__()  # Initialize the StateEstimation class

    def _simple_airdensity(self) -> float:
        """Return a constant air density for the simplified model."""
        return const.SEA_LEVEL_AIR_DENSITY  # Air density at sea level in kg/m^3

    def _realistic_airdensity(self) -> float:
        """Compute air density based on altitude using an exponential model."""
        altitude = self.get_position_geodetic()[2]
        return const.SEA_LEVEL_AIR_DENSITY * np.exp(-altitude / const.scale_height)

    def _simple_gravity(self) -> npt.NDArray[np.float64]:
        """Simple model for gravitational acceleration, assuming constant
        gravity at sea level."""
        # Gravitational acceleration in m/s^2, pointing downwards in z.
        return np.array([0, 0, -const.G0], dtype=np.float64)

    def _compute_gravitational_acceleration(self) -> npt.NDArray[np.float64]:
        """Compute the correct gravitational acceleration with respect to the
        Earth's center, for a spherical Earth, pointing opposite to the position vector."""
        position = self.get_position_cartesian()
        r = np.linalg.norm(position)  # Distance from the center of the Earth
        assert r > 0, "Position vector must be positive to compute gravity."
        return np.asarray(
            -const.GM / r**2 * position / r,
            dtype=np.float64,
        )

    def _calc_alt_simpl(self) -> np.float64:
        """Calculate altitude above the surface of the Earth based on the
        position vector, assuming a flat Earth for simplicity."""
        # Assuming z-component of position is altitude for the simplified model.
        return self.get_position_cartesian()[2]

    def _calc_alt_realistic(self) -> np.float64:
        """Calculate altitude above the surface of the Earth based on the
        position vector."""
        return self.get_position_geodetic()[2]

    def _spherical_earth_forces(self) -> npt.NDArray[np.float64]:
        """Compute the forces arising due to Earth's rotation."""
        position = self.get_position_cartesian()
        velocity = self.get_velocity_cartesian()
        # Compute ellipsoidal gravity with J2 perturbation and add pseudo forces due to Earth's rotation.
        x, y, z = position
        r: float = np.linalg.norm(position)
        assert r > 0, "Position vector must be positive to compute gravity."
        k: float = 1.5 * const.J2 * (const.RE / r) ** 2
        z2r2 = 5.0 * z**2 / r**2
        ax = -const.GM * x / r**3 * (1.0 - k * (z2r2 - 1.0))
        ay = -const.GM * y / r**3 * (1.0 - k * (z2r2 - 1.0))
        az = -const.GM * z / r**3 * (1.0 - k * (z2r2 - 3.0))
        acc = np.array([ax, ay, az], dtype=np.float64)

        coriolis_acc = -2 * np.cross(const.EARTH_ROTATION, velocity)
        centrifugal_acc = -np.cross(
            const.EARTH_ROTATION, np.cross(const.EARTH_ROTATION, position)
        )
        acc += coriolis_acc + centrifugal_acc

        return acc

    def _compute_drag_force(
        self, air_density: float, drag_coeff: float, cross_sectional_area: float
    ) -> npt.NDArray[np.float64]:
        """Calculate the drag force with a vector opposite to velocity."""
        velocity = self.get_velocity_cartesian()
        speed = np.linalg.norm(velocity)
        return (
            -0.5
            * air_density
            * speed**2
            * drag_coeff
            * cross_sectional_area
            * velocity
            / speed
        )

    def _alt_to_pres(self) -> float:
        """Simple model to convert altitude to pressure using the barometric
        formula."""
        altitude = self.get_position_geodetic()[2]
        return const.SEA_LEVEL_PRESSURE * np.exp(-altitude / const.scale_height)

    @staticmethod
    def _press_to_alt(pressure: float) -> float:
        """Simple model to convert pressure to altitude using the barometric
        formula."""
        return -const.scale_height * np.log(pressure / const.SEA_LEVEL_PRESSURE)


###############################################################


def geopotential_to_altitude(geopotential: float) -> float:
    """Convert geopotential to altitude using the formula:
    altitude = geopotential / g0, where g0 is the standard gravity at
    sea level (9.81 m/s^2)."""
    return geopotential / const.G0


def altitude_to_geopotential(altitude: float) -> float:
    """Convert altitude to geopotential using the formula:
    geopotential = altitude * g0, where g0 is the standard gravity at
    sea level (9.81 m/s^2): ERA5,
    https://software.ecmwf.int/wiki/display/IFS/CY41R1+Official+IFS+Documentation
    """
    return altitude * const.G0


def _get_compass_wind(wind_direction: float) -> float:
    """Convert cartesian wind direction to compass degrees, assuming true
    north up (opposite to y-axis) and east to the right (x-axis), and that
    the wind direction is given in degrees counterclockwise from the x-axis
    (east)."""
    if wind_direction <= 90:
        return 90 - wind_direction
    else:
        return 360 - (wind_direction - 90)


def _get_cartesian_wind(wind_direction: float) -> float:
    """Convert compass wind direction to cartesian degrees,
    the wind direction is given in degrees counterclockwise from the x-axis
    (east)."""
    if wind_direction < 90:
        return -wind_direction + 90
    elif wind_direction == 90:
        return 0
    else:
        return (360 - wind_direction) + 90


def plot_results(
    out_results: pd.DataFrame,
    figsize: tuple,
    params: dict,
    lw: float = 0.5,
    lim: Optional[int] = 100,
    scale_fac: float = 1000,
) -> plt.Figure:
    # logger not used yet
    fig = plt.figure(figsize=figsize, dpi=150, constrained_layout=True)
    ax1 = plt.subplot(1, 3, 1)
    ax2 = plt.subplot(1, 3, 2)
    ax3 = plt.subplot(1, 3, 3)
    ax11 = ax1.twinx()

    if params["wind"] != "Monte Carlo":
        t = out_results["time"]
        if params["mode"] == "simplified":
            altitude = out_results["altitude"]
            velocity = np.vstack(
                (
                    out_results["velocity cartesian x"],
                    out_results["velocity cartesian y"],
                    out_results["velocity cartesian z"],
                )
            ).T
            positions = np.vstack(
                (
                    out_results["position cartesian x"],
                    out_results["position cartesian y"],
                )
            ).T
        elif params["mode"] == "realistic":
            altitude = out_results["position geodetic alt"]
            velocity = np.vstack(
                (
                    out_results["velocity geodetic East"],
                    out_results["velocity geodetic North"],
                    out_results["velocity geodetic alt"],
                )
            ).T
            positions = np.vstack(
                (
                    out_results["position geodetic lon"],
                    out_results["position geodetic lat"],
                )
            ).T
        else:
            raise ValueError(
                f"Invalid mode: {params['mode']}. Must be 'simplified' or 'realistic'."
            )

        # actual plotting
        ax1.plot(t, altitude, label="Altitude (m)", color="b", lw=lw)
        ax11.plot(t, velocity[:, 2], label="Vertical Velocity (m/s)", color="r", lw=lw)

        ax2.plot(
            positions[:, 0] / scale_fac,
            positions[:, 1] / scale_fac,
            label="Position (km)",
            color="b",
            lw=lw,
        )
        ax2.scatter(
            positions[-1, 0] / scale_fac,
            positions[-1, 1] / scale_fac,
            color="b",
            s=lw * 2,
        )

        ax3.plot(
            velocity[:, 0],
            velocity[:, 1],
            label="Horizontal Velocity (m/s)",
            color="r",
            lw=lw,
        )
        ax3.scatter(
            velocity[:, 0],
            velocity[:, 1],
            color="r",
            s=0.5 * lw,
        )

    elif params["wind"] == "Monte Carlo":
        for out in out_results:
            if params["mode"] == "realistic":
                t = out["time"]
                altitude = out["position geodetic alt"]
                velocity = np.vstack(
                    (
                        out["velocity geodetic East"],
                        out["velocity geodetic North"],
                        out["velocity geodetic alt"],
                    )
                ).T
                positions = np.vstack(
                    (
                        out["position geodetic lon"],
                        out["position geodetic lat"],
                    )
                ).T
            elif params["mode"] == "simplified":
                t = out["time"]
                altitude = out["altitude"]
                velocity = np.vstack(
                    (
                        out["velocity cartesian x"],
                        out["velocity cartesian y"],
                        out["velocity cartesian z"],
                    )
                ).T
                positions = np.vstack(
                    (
                        out["position cartesian x"],
                        out["position cartesian y"],
                    )
                ).T
            else:
                raise ValueError(
                    f"Invalid mode: {params['mode']}. Must be 'simplified' or 'realistic'."
                )

            # actual plotting
            ax1.plot(t, altitude, label="Altitude (m)", color="b", lw=lw)
            ax11.plot(
                t, velocity[:, 2], label="Vertical Velocity (m/s)", color="r", lw=lw
            )

            ax2.plot(
                positions[:, 0] / scale_fac,
                positions[:, 1] / scale_fac,
                label="Position (km)",
                color="b",
                lw=lw,
            )
            ax2.scatter(
                positions[-1, 0] / scale_fac,
                positions[-1, 1] / scale_fac,
                color="b",
                s=2 * lw,
            )

            ax3.plot(
                velocity[:, 0],
                velocity[:, 1],
                label="Horizontal Velocity (m/s)",
                color="r",
                lw=lw,
            )
            ax3.scatter(
                velocity[:, 0],
                velocity[:, 1],
                color="r",
                s=0.5 * lw,
            )

    ax1.set_title("Rocket altitude over Time")
    ax2.set_title("Rocket Trajectory (X-Y Position)")
    ax3.set_title("Rocket TAS (True Airspeed) (X-Y)")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Altitude (m)")
    ax11.set_ylabel("Vertical Velocity (m/s)")
    if scale_fac == 1:
        ax2.set_xlabel("Longitude (deg)")
        ax2.set_ylabel("Latitude (deg)")
        ax2.set_xticklabels([f"{tick:.3f}°" for tick in ax2.get_xticks()], rotation=45)
        ax2.set_yticklabels([f"{tick:.3f}°" for tick in ax2.get_yticks()])
    else:
        ax2.set_xlabel("X Position (km)")
        ax2.set_ylabel("Y Position (km)")
    if lim is not None:
        ax2.set_xlim(-lim, lim)
        ax2.set_ylim(-lim, lim)
    ax3.set_xlabel("Velocity X (m/s)")
    ax3.set_ylabel("Velocity Y (m/s)")
    ax1.legend()
    ax11.legend()
    ax2.legend()
    ax3.legend()
    plt.show()

    return fig
