import warnings
from typing import Callable, Optional
from .modes import validate_mode
from .utils import PhysicsFunctions
from .windfield import WindField
from .logging import Logging
import numpy as np
import numpy.typing as npt

from scipy.integrate import RK45

# Default integration tolerances (overridable via params "rtol",
# "atol_position" in m and "atol_velocity" in m/s).
DEFAULT_RTOL = 1e-6
DEFAULT_ATOL_POSITION = 1.0
DEFAULT_ATOL_VELOCITY = 0.01


class Rocket(PhysicsFunctions):
    # These four are bound to concrete implementations in __init__ based on
    # the simulation mode (strategy dispatch, see _setup_simplified_mode and
    # _setup_realistic_mode). Declared here as instance
    # attributes so type checkers see them without the "cannot assign to a
    # method" error a placeholder method definition would trigger.
    calc_altitude_abv_sea_level: Callable[[npt.NDArray[np.float64]], np.float64]
    grav_acc: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]
    # (state, altitude) -> density; pass the altitude when already known,
    # or None to compute it from the state.
    get_air_density: Callable[[npt.NDArray[np.float64], Optional[float]], np.float64]
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

        # Initialize the ODE solver for the rocket's equations of motion.
        # Tolerances in physical units: scipy's defaults (rtol=1e-3) would
        # allow kilometres of error per step on ECEF positions of ~6.4e6 m,
        # leaving max_step as the only accuracy control. With these, the step
        # size adapts to the dynamics and max_step can be large.
        atol_position = self.params.get("atol_position", DEFAULT_ATOL_POSITION)
        atol_velocity = self.params.get("atol_velocity", DEFAULT_ATOL_VELOCITY)
        self.solver = RK45(
            self.equations_of_motion,
            0,
            self.state_cartesian,
            self.params.get("tmax"),
            max_step=self.params.get("max_step") or np.inf,
            rtol=self.params.get("rtol", DEFAULT_RTOL),
            atol=np.array([atol_position] * 3 + [atol_velocity] * 3),
        )

        # Output grid: with output_interval set, states are logged every
        # output_interval seconds by interpolating within the solver steps
        # (dense output), independent of the step size. Without it, every
        # solver step is logged.
        self.output_interval = self.params.get("output_interval")
        if self.output_interval is not None and self.output_interval <= 0:
            raise ValueError("output_interval must be positive (or None).")
        self._next_output_t = self.output_interval or 0.0
        # True once the trajectory has reached the ground (see update_state).
        self.landed = False

    def get_drag_force(
        self,
        state: npt.NDArray[np.float64],
        wind_velv: npt.NDArray[np.float64],
        altitude: Optional[float] = None,
    ) -> npt.NDArray[np.float64]:
        """Calculate the drag force based on velocity and drag
        coefficient. Pass the altitude if it is already known (one
        Cartesian -> geodetic conversion serves both the parachute check and
        the air density)."""
        if altitude is None:
            altitude = self.calc_altitude_abv_sea_level(state)
        if (
            altitude < self.params["parachute_opening_altitude"]
        ):  # If altitude < parachute_opening_altitude, use parachute drag
            drag_coeff = self.params["drag_coefficient_parachute"]
            cross_sectional_area = self.params["cross_sectional_area_parachute"]
        else:
            drag_coeff = self.params["drag_coefficient"]
            cross_sectional_area = self.params["cross_sectional_area"]

        air_density = self.get_air_density(state, altitude)

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
        # The altitude is needed for the parachute check and the air density:
        # compute it once per evaluation.
        altitude = self.calc_altitude_abv_sea_level(state)
        drag_force = self.get_drag_force(state, self._wind_cartesian, altitude)

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
        t_old = self.solver.t
        self.solver.step()
        if self.solver.status == "failed":
            raise RuntimeError(f"RK45 step failed at t = {t_old:.3f} s.")
        t_end, y_end = self.solver.t, self.solver.y.copy()

        # The solver interpolant over the step just taken (built lazily: only
        # needed for the ground crossing or a fixed output grid).
        dense = None

        # Ground crossing: if the step ends below the ground, find the exact
        # crossing on the interpolant and end the trajectory there, instead
        # of at a point up to a whole step below the surface.
        if self.calc_altitude_abv_sea_level(y_end) <= 0.0:
            dense = self.solver.dense_output()
            t_end = self._ground_crossing_time(dense, t_old, t_end)
            y_end = dense(t_end)
            self.landed = True
            # Stop: callers' loops end on solver.status != "running".
            self.solver.status = "finished"

        if self.output_interval:
            # Log on the fixed output grid within (t_old, t_end].
            if dense is None:
                dense = self.solver.dense_output()
            while self._next_output_t <= t_end:
                self._log_at(self._next_output_t, dense(self._next_output_t))
                self._next_output_t += self.output_interval
            # Always log the final state (ground crossing or tmax) as well.
            if (
                self.solver.status != "running"
                and self.logger.time[self.logger.i - 1] != t_end
            ):
                self._log_at(t_end, y_end)
        else:
            # Log every solver step (the crossing state on the last one).
            self._log_at(t_end, y_end)

        # Commit the state at the end of the step (or at the ground crossing).
        self.set_position_cartesian(y_end[:3])
        self.set_velocity_cartesian(y_end[3:])
        self.update_time(t_end)

    def _log_at(self, t: float, y: npt.NDArray[np.float64]) -> None:
        """Log the (possibly interpolated) state y at time t. The logger reads
        the rocket's state, so it is set to y first; update_state commits the
        end-of-step state afterwards."""
        self.set_position_cartesian(y[:3])
        self.set_velocity_cartesian(y[3:])
        self.update_time(t)
        self.logger.log_state(self, t)
        self.logger.log_wind(self.wind_field.wind_velv)  # Log the wind velocity

    def _ground_crossing_time(self, dense, t_above: float, t_below: float) -> float:
        """Time at which the trajectory reaches the ground within a step,
        by bisection on the step interpolant. Returns a time at or just after
        the crossing (altitude <= 0), resolved to 1e-6 s."""
        if self.calc_altitude_abv_sea_level(dense(t_above)) <= 0.0:
            return t_above  # already at/below the ground at the step start
        for _ in range(100):
            if t_below - t_above <= 1e-6:
                break
            t_mid = 0.5 * (t_above + t_below)
            if self.calc_altitude_abv_sea_level(dense(t_mid)) > 0.0:
                t_above = t_mid
            else:
                t_below = t_mid
        return t_below
