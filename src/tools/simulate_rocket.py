import warnings
from typing import Callable
from .utils import PhysicsFunctions
from .windfield import WindField
from .logging import Logging
import numpy as np
import numpy.typing as npt

from scipy.integrate import RK45


class Rocket(PhysicsFunctions):
    # These three are bound to concrete implementations in __init__ based on
    # the simulation mode (strategy dispatch). Declared here as instance
    # attributes so type checkers see them without the "cannot assign to a
    # method" error a placeholder method definition would trigger.
    calc_altitude_abv_sea_level: Callable[[npt.NDArray[np.float64]], np.float64]
    grav_acc: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]
    get_air_density: Callable[[npt.NDArray[np.float64]], np.float64]

    """
    Simulate the rocket's flight using the equations of motion.

    Parameters:
    t (array): Time array for simulation.
    initial_conditions (array): Initial state of the rocket
        [position, velocity].
    params (dict): Dictionary containing rocket parameters (mass, drag
        coefficient, etc.).

    params = {
        'mass': 700,  # mass of the rocket in kg, assumed to be a pointmass
        'drag_coefficient': 1.4,  # drag coefficient rocket (dimensionless)
        'drag_coefficient_parachute': 2.0,  # drag coeff. with parachute
        'cross_sectional_area': 1.14,  # cross-sectional area in m^2
        'cross_sectional_area_parachute': 18,  # parachute area in m^2
        'wind': "ERA5",  # which wind data to use; "ERA5" for realistic
        'mode': 'realistic',  # 'realistic' or 'simplified' assumptions
        'verbose': False  # print rocket state at each time step
    }

    Some of the params could also be made global parameters specified in
    the beginning of a script.

    Returns:
    array: Simulated state of the rocket over time.
    """

    def __init__(
        self,
        initial_conditions: list[npt.NDArray[np.float64]],
        params: dict,
        logger: Logging,
        wind_field: WindField,
    ) -> None:
        """Initialize the Rocket simulation with initial conditions and
        parameters."""
        super().__init__()
        self.params = params
        self.wind_field = wind_field
        self.state_cartesian = np.hstack(
            initial_conditions, dtype=np.float64
        )  # set the initial state
        self.logger = logger

        if self.params["mass"] <= 0:
            raise ValueError("Rocket mass must be positive. Check your parameters.")

        # Set the appropriate methods for altitude, gravity, and air
        # density calculations based on the simulation mode and wind data.

        if (
            self.params.get("mode") == "simplified"
            and self.params.get("wind") != "ERA5"
        ):
            # Use simplified models for altitude, gravity, and air density
            self.calc_altitude_abv_sea_level = self._calc_alt_simpl
            self.grav_acc = self._simple_gravity
            self.get_air_density = self._simple_airdensity
            warnings.warn(
                "Mode: simplified. No wind data or pseudo forces will be "
                "used. Gravity and air density are simplified.",
                UserWarning,
            )
        elif (
            self.params.get("mode") == "realistic" and self.params.get("wind") != "ERA5"
        ):
            # Use realistic models for altitude, gravity, and air density
            self.calc_altitude_abv_sea_level = self._calc_alt_geodetic
            self.get_air_density = self._realistic_airdensity
            self.grav_acc = self._gravitational_acceleration_ecef
        elif self.params.get("wind") == "ERA5":
            # Use realistic models for altitude, gravity, and air density
            # plus ERA5 wind data
            warnings.warn(
                "Using ERA5 wind data automatically sets mode to realistic.",
                UserWarning,
            )
            self.calc_altitude_abv_sea_level = self._calc_alt_geodetic
            self.get_air_density = self._realistic_airdensity
            self.grav_acc = self._gravitational_acceleration_ecef
        else:
            # If no valid mode is specified, default to simplified models
            # and issue a warning
            warnings.warn(
                "No valid mode specified. Defaulting to simplified mode "
                "with no wind data or pseudo forces.",
                UserWarning,
            )
            self.calc_altitude_abv_sea_level = self._calc_alt_simpl
            self.grav_acc = self._simple_gravity
            self.get_air_density = self._simple_airdensity

        # Wind is held constant across the RK stages of a single step
        # (refreshed once per committed step in update_state). Seed it here so
        # it exists for the RHS call RK45 makes during construction.
        self._wind_cartesian = self._update_environment(0.0, self.state_cartesian)

        # Initialize the ODE solver for the rocket's equations of motion
        self.solver = RK45(
            self.equations_of_motion,
            0,
            self.state_cartesian,
            self.params.get("tmax"),
            max_step=self.params.get("max_step"),
        )

    def get_drag_force(
        self, state: npt.NDArray[np.float64], wind_velv: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Calculate the drag force based on velocity and drag
        coefficient."""
        # get altitude from the current position of the rocket
        altitude = self.calc_altitude_abv_sea_level(state)
        if (
            altitude < self.params["parachute_opening_altitude"]
        ):  # If altitude < parachute_opening_altitude, use parachute drag
            drag_coeff = self.params["drag_coefficient_parachute"]
            cross_sectional_area = self.params["cross_sectional_area_parachute"]
        else:
            drag_coeff = self.params["drag_coefficient"]
            cross_sectional_area = self.params["cross_sectional_area"]

        air_density = self.get_air_density(state)

        return self.compute_drag_force(
            state, wind_velv, air_density, drag_coeff, cross_sectional_area
        )

    def _update_environment(
        self, t: float, state: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Update the wind field based on the current position of the rocket
        and the specified wind field conditions. No need to pass the state, since the state
        is already internally updated."""
        if self.params.get("wind") == "ERA5" or self.params.get("mode") == "realistic":
            position_geodetic = self.convert_cartesian_to_geodetic(state[:3])
            # Get current wind for present position
            wind_vel_geodetic = self.wind_field.get_wind(t, position_geodetic)
            wind_velv = self.convert_velocity_geodetic_to_cartesian(
                wind_vel_geodetic, position_geodetic[0], position_geodetic[1]
            )
        else:  # simplified mode, or default wind.
            position = state[:3]
            wind_velv = self.wind_field.get_wind(t, position)
        return wind_velv

    def equations_of_motion(
        self, t: float, state: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """
        EOMs for the rocket, calculating the acceleration based on the
        current state and forces acting on the rocket.
        """
        # Get current position and velocity from state
        velocity = state[3:]

        # Wind is frozen for the whole step (refreshed once per committed step
        # in update_state) and reused across the RK stages, rather than
        # re-interpolating the wind field at every RHS evaluation. Over one
        # <= max_step interval the wind changes negligibly, so this is a
        # standard, accurate approximation that avoids ~6x redundant lookups.
        drag_force = self.get_drag_force(state, self._wind_cartesian)

        g_acc = self.grav_acc(state)

        # Calculate acceleration: F = ma => a = F/m, drag force is negative
        accelerationv = g_acc + drag_force / self.params["mass"]
        state_derivative = np.hstack((velocity, accelerationv))

        return state_derivative

    def update_state(
        self,
    ) -> None:
        """
        Update the state of the rocket based on the equations of motion
        and wind conditions.
        """
        # Refresh the wind once, at the current committed state/time, and hold
        # it constant for all RK stages of the step taken below.
        self._wind_cartesian = self._update_environment(
            self.solver.t, self.state_cartesian
        )
        self.solver.step()

        # Update position and velocity with wind effects
        self.set_position_cartesian(self.solver.y[:3])
        self.set_velocity_cartesian(self.solver.y[3:])
        self.update_time(self.solver.t)  # Update the time of the state estimation

        # Log the state of the rocket at the new time step
        self.logger.log_state(self, self.solver.t)
        self.logger.log_wind(self.wind_field.wind_velv)  # Log the wind velocity
