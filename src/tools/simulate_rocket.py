import warnings
from .utils import PhysicsFunctions
from .windfield import WindField
import numpy as np
import numpy.typing as npt


class Rocket(PhysicsFunctions):
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
        'pseudo_forces': True,  # include Earth-rotation pseudo-forces
        'verbose': False  # print rocket state at each time step
    }

    Some of the params could also be made global parameters specified in
    the beginning of a script.

    Returns:
    array: Simulated state of the rocket over time.
    """

    def __init__(
        self,
        initial_conditions: list[npt.NDArray[np.float64], npt.NDArray[np.float64]],
        params: dict,
        logger: object,
    ) -> None:
        """Initialize the Rocket simulation with initial conditions and
        parameters."""
        super().__init__()
        self.params = params
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
            self.calc_altitude_abv_sea_level = self._calc_alt_realistic
            self.get_air_density = self._realistic_airdensity
            # Use realistic gravity model, with option to include pseudo
            # forces if specified
            if self.params.get("pseudo_forces") is True:
                warnings.warn(
                    "Using pseudo forces (Coriolis and centrifugal) in realistic mode.",
                    UserWarning,
                )
                self.grav_acc = self._spherical_earth_forces
            else:
                self.grav_acc = self._compute_gravitational_acceleration
        elif self.params.get("wind") == "ERA5":
            # Use realistic models for altitude, gravity, and air density
            # plus ERA5 wind data
            warnings.warn(
                "Using ERA5 wind data automatically sets mode to realistic.",
                UserWarning,
            )
            self.calc_altitude_abv_sea_level = self._calc_alt_realistic
            self.get_air_density = self._realistic_airdensity
            # Use realistic gravity model, with option to include pseudo
            # forces if specified
            if self.params.get("pseudo_forces") is True:
                warnings.warn(
                    "Using pseudo forces (Coriolis and centrifugal) in realistic mode.",
                    UserWarning,
                )
                self.grav_acc = self._spherical_earth_forces
            else:
                self.grav_acc = self._compute_gravitational_acceleration
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

    def get_drag_force(self) -> npt.NDArray[np.float64]:
        """Calculate the drag force based on velocity and drag
        coefficient."""
        # get altitude from the current position of the rocket
        altitude = self.calc_altitude_abv_sea_level()
        if altitude < self.params.get(
            "parachute_opening_altitude"
        ):  # If altitude < parachute_opening_altitude, use parachute drag
            drag_coeff = self.params["drag_coefficient_parachute"]
            cross_sectional_area = self.params["cross_sectional_area_parachute"]
        else:
            drag_coeff = self.params["drag_coefficient"]
            cross_sectional_area = self.params["cross_sectional_area"]

        air_density = self.get_air_density()

        return self._compute_drag_force(air_density, drag_coeff, cross_sectional_area)

    def equations_of_motion(
        self,
    ) -> list[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """
        EOMs for the rocket, calculating the acceleration based on the
        current state and forces acting on the rocket.
        """
        # Get current position and velocity from state
        velocity = self.get_velocity_cartesian()

        # Calculate drag force - coordinate system assertion happens
        # already in subfunctions.
        drag_force = self.get_drag_force()
        grav_acc = self.grav_acc()

        # Calculate acceleration: F = ma => a = F/m, drag force is negative
        accelerationv = grav_acc + drag_force / self.params["mass"]

        return [velocity, accelerationv]

    def update_state(
        self,
        t: npt.NDArray[np.float64],
        wind_field: WindField,
        wind_field_conditions: list[npt.NDArray[np.float64], npt.NDArray[np.float64]],
    ) -> None:
        """
        Update the state of the rocket based on the equations of motion
        and wind conditions.
        """
        velocity, acceleration = self.equations_of_motion()
        position = self.get_position_cartesian()

        # Need to get the correct wind velocity vector in the cartesian
        # coordinate system of the simulation; wind is given as
        # horizontal to the surface of the Earth. fix this, might be able to make nicer
        if self.params.get("wind") == "ERA5" or self.params.get("mode") == "realistic":
            position_geodetic = self.get_position_geodetic()
            wind_field.update_wind(t[0], position_geodetic, wind_field_conditions)
            wind_vel_geodetic = wind_field.wind_velv
            lat, lon, _ = position_geodetic
            wind_velv = self.convert_velocity_geodetic_to_cartesian(
                wind_vel_geodetic, lat, lon
            )
        else:  # simplified mode, or default wind.
            wind_field.update_wind(t[0], position, wind_field_conditions)
            wind_velv = wind_field.wind_velv

        new_velocity = velocity + acceleration * (t[1] - t[0])
        new_position = (
            position + new_velocity * (t[1] - t[0]) + wind_velv * (t[1] - t[0])
        )

        self.set_position_cartesian(new_position)
        self.set_velocity_cartesian(new_velocity)
        self.estimate_state(t[1])

        # Log the state of the rocket at the new time step
        self.logger.log_state(self, t[1])
        self.logger.log_wind(wind_velv)  # Log the wind velocity
