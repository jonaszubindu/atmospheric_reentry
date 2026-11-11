import warnings
from typing import Callable
from .modes import validate_mode
from .utils import PhysicsFunctions
from .windfield import WindField
from .logging import Logging
import numpy as np
import numpy.typing as npt

from scipy.integrate import RK45


class Rocket(PhysicsFunctions):
    # These four are bound to concrete implementations in __init__ based on
    # the simulation mode (strategy dispatch, see _setup_simplified_mode and
    # _setup_realistic_mode). Declared here as instance
    # attributes so type checkers see them without the "cannot assign to a
    # method" error a placeholder method definition would trigger.
    calc_altitude_abv_sea_level: Callable[[npt.NDArray[np.float64]], np.float64]
    grav_acc: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]
    get_air_density: Callable[[npt.NDArray[np.float64]], np.float64]
    _update_environment: Callable[
        [float, npt.NDArray[np.float64]], npt.NDArray[np.float64]
    ]

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

        if self.params.get("solver_method", "RK45") != "RK45":
            raise NotImplementedError(
                "Only the RK45 solver is implemented. Set solver_method to 'RK45'."
            )

        # Bind the frame and environment models of the simulation mode. This
        # is the only place where the two modes differ in the dynamics, see
        # modes.py.
        if validate_mode(self.params) == "simplified":
            self._setup_simplified_mode()
        else:
            self._setup_realistic_mode()

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

    def _setup_simplified_mode(self) -> None:
        """Bind the models of the simplified mode: a flat, non-rotating Earth
        with constant gravity, constant air density and a constant wind."""
        self.calc_altitude_abv_sea_level = self._calc_alt_simpl
        self.grav_acc = self._simple_gravity
        self.get_air_density = self._simple_airdensity
        self._update_environment = self._wind_simplified
        warnings.warn(
            "Mode: simplified. Flat, non-rotating Earth with constant gravity "
            "and air density, meant as a test box (see tools/modes.py).",
            UserWarning,
        )

    def _setup_realistic_mode(self) -> None:
        """Bind the models of the realistic mode: a rotating WGS84 Earth in
        ECEF coordinates with J2 gravity, pseudo forces, exponential air
        density and a wind given in local east/north/up."""
        self.calc_altitude_abv_sea_level = self._calc_alt_geodetic
        self.grav_acc = self._gravitational_acceleration_ecef
        self.get_air_density = self._realistic_airdensity
        self._update_environment = self._wind_realistic

    def _wind_simplified(
        self, t: float, state: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Wind at the current position in simplified mode. The wind field
        already returns it in the Cartesian x, y, z of the flat Earth."""
        return self.wind_field.get_wind(t, state[:3])

    def _wind_realistic(
        self, t: float, state: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Wind at the current position in realistic mode. The wind field
        returns it in local east/north/up, which is rotated into ECEF."""
        position_geodetic = self.convert_cartesian_to_geodetic(state[:3])
        wind_vel_geodetic = self.wind_field.get_wind(t, position_geodetic)
        return self.convert_velocity_geodetic_to_cartesian(
            wind_vel_geodetic, position_geodetic[0], position_geodetic[1]
        )

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
