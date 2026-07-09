import numpy as np
import pandas

from tools.logging import Logging
from tools.simulate_rocket import Rocket
from tools.state_estimation import RealState
from tools.utils import (
    plot_results,
)
from tools.windfield import WindField


def verbose_message(
    rocket_simulation: Rocket, wind_field: WindField, time0: float
) -> None:
    if (
        rocket_simulation.params.get("mode") == "realistic"
        or rocket_simulation.params.get("wind") == "ERA5"
    ):
        lat, lon, alt = rocket_simulation.get_position_geodetic()
        v_east, v_north, v_up = rocket_simulation.get_velocity_geodetic()
        state_derivative = rocket_simulation.equations_of_motion(
            rocket_simulation.time, rocket_simulation.state_cartesian
        )
        acc_east, acc_north, acc_up = RealState.convert_velocity_cartesian_to_geodetic(
            state_derivative[3:], lat, lon
        )
        wind_vel_east, wind_vel_north, wind_vel_up = wind_field.wind_velv

        print(
            f"Time: {time0:.2f} s, "
            f"Position (lat, lon, alt): "
            f"({lat:.3f}°, {lon:.3f}°, {alt:.3f} m), "
            f"Acceleration (a_east, a_north, a_up): ",
            f"({acc_east:.2f}, {acc_north:.2f}, {acc_up:.2f}) m/s², "
            f"Velocity (v_east, v_north, v_up): "
            f"({v_east:.2f}, {v_north:.2f}, {v_up:.2f}) m/s, "
            f"Wind Velocity (v_wind_east, v_wind_north, v_wind_up): "
            f"({wind_vel_east:.2f}, {wind_vel_north:.2f}, {wind_vel_up:.2f}) m/s",
        )
    else:
        position = rocket_simulation.get_position_cartesian()
        velocity = rocket_simulation.get_velocity_cartesian()
        print(
            f"Time: {time0:.2f} s, "
            f"Position (x, y, z): "
            f"({position[0]:.2f}, {position[1]:.2f}, "
            f"{position[2]:.2f} m), "
            f"Velocity (vx, vy, vz): "
            f"({velocity[0]:.2f}, {velocity[1]:.2f}, "
            f"{velocity[2]:.2f}) m/s"
        )


def main(
    n_steps: int,
    rocket_simulation: Rocket,
    verbose: bool = False,
) -> pandas.DataFrame:
    """Run the rocket simulation with the given wind conditions."""

    for ii in range(n_steps):
        # Update state and conditions
        rocket_simulation.update_state()
        altitude = rocket_simulation.calc_altitude_abv_sea_level(
            rocket_simulation.get_position_cartesian()
        )

        if verbose and ii % 100 == 0:  # Print every 100 steps to reduce output
            verbose_message(
                rocket_simulation,
                rocket_simulation.wind_field,
                rocket_simulation.solver.t,
            )

        if altitude <= 0.0:
            # The rocket has hit the ground, stop the simulation
            break

    # Adjust time array to match the length of the state array
    if verbose:
        for warning_ in rocket_simulation.logger.warnings:
            if warning_:
                print(warning_)
        print("Simulation completed successfully.")

    return rocket_simulation.logger.get_full_state()


def menu() -> tuple[str, str, bool, bool]:
    """Display a menu for the user to select simulation parameters."""
    print("Welcome to the Rocket Simulation!")
    print("Please select the simulation mode:")
    print("1. Realistic")
    print("2. Simplified")
    mode_choice = input("Enter your choice (1 or 2): ")

    if mode_choice == "1":
        mode = "realistic"
    elif mode_choice == "2":
        mode = "simplified"
    else:
        print("Invalid choice. Defaulting to 'realistic' mode.")
        mode = "realistic"

    print("\nPlease select the wind data source:")
    print("1. ERA5 (Realistic Wind Data)")
    print("2. Default (No Wind Data)")
    wind_choice = input("Enter your choice (1 or 2): ")

    if wind_choice == "1":
        wind_data = "ERA5"
    elif wind_choice == "2":
        wind_data = "default"
    else:
        print("Invalid choice. Defaulting to 'ERA5' wind data.")
        wind_data = "ERA5"

    if (
        mode == "realistic" or wind_data == "ERA5"
    ):  # required for map drawing, must be aware of coordinate systems
        print("\nPlease select whether to draw a basemap:")
        basemap_choice = input("Enter your choice (y/n): ")
        if basemap_choice.lower() == "y":
            basemap = True
        else:
            basemap = False

    print("\nPlease select whether to enable verbose mode:")
    verbose_choice = input("Enter your choice (y/n): ")
    if verbose_choice.lower() == "y":
        verbose = True
    else:
        verbose = False

    return mode, wind_data, basemap, verbose


if __name__ == "__main__":
    # Get user input for simulation parameters
    mode, wind_data, basemap, verbose = menu()

    params = {
        # mass of the rocket in kg, assumed to be a pointmass
        "mass": 700,  # default 700,
        # drag coefficient rocket (dimensionless)
        "drag_coefficient": 1.4,  # default 1.4,
        # drag coefficient with parachute deployed (dimensionless)
        "drag_coefficient_parachute": 2.0,  # default 2.0,
        # cross-sectional area in m^2
        "cross_sectional_area": 1.14,  # default 1.14,
        # cross-sectional area of parachute in m^2
        "cross_sectional_area_parachute": 18,  # default 18,
        # which wind data to use; use "default" for default behaviour,
        # use "ERA5" for realistic wind data.
        "wind": wind_data,
        # Set altitude at which the parachute should open
        "parachute_opening_altitude": 10000,
        # maximum simulation time in seconds
        "tmax": 3000,
        # maximum stepsize
        "max_step": 0.3,
        # solver method for the ODE integration, choose from
        # "RK45", "RK23", "DOP853", "Radau", "BDF", "LSODA" default is "RK45".
        # See scipy.integrate.solve_ivp for more details.
        "solver_method": "RK45",
        # use 'realistic' to use all realistic assumptions and
        # parameters, or use 'simplified' assumptions for testing and
        # debugging, e.g. no drag, constant gravity, etc.
        "mode": mode,
        # draw an ICAO / aeronautical basemap behind the ground-track panel
        # (realistic mode only). Fetches map tiles over the network; airspaces
        # need an openAIP API key (see OPENAIP_API_KEY below).
        "basemap": basemap,
        # print the state of the rocket at each time step for
        # debugging and analysis
        "verbose": verbose,
    }

    # Initialize parameters and initial conditions.
    # Default wind conditions: 270 degrees (north to south), 15 m/s wind
    # speed, not used if using ERA5 wind data. Degrees are given as
    # cartesian degrees, meaning 0 degrees is eastward, 90 degrees is
    # northward, 180 degrees is westward and 270 degrees is southward,
    # matching the coordinate system of the simulation.
    n_steps = params["tmax"]
    wind_field_conditions = [
        270.0,
        15.0,
    ]  # [270.0, 15.0] # make it that it reflects aviation convention.

    if params["wind"] == "default" and params["mode"] == "simplified":
        # This for now also automatically assumes a flat earth.
        # Initial position (x, y, z) in meters, rocket dropped at 60 km
        # height.
        position_vec = np.array([0, 0, 2000], dtype=np.float64)
        # Initial velocity (vx, vy, vz) in m/s, rocket dropped at 70 m/s
        # horizontal in x and y and -280 m/s vertical velocity.
        velocity_vec = np.array([30.0, 0.0, 0.0], dtype=np.float64)

    if params["mode"] == "realistic" or params["wind"] == "ERA5":
        # Initial coordinates: dummy values for a mid-latitude coastal
        # launch site where the wind data is already available.
        # lon = 360 - 75.0
        # lat = 38.5
        # alt = 60000

        # Dübendorf airfield (LSMD), Switzerland — aerodrome reference point.
        lat = 47.3986  # in deg, positive north, negative south
        lon = 8.6486  # in deg, positive east, negative west
        alt = 60000  # entry/drop altitude in m (airfield elevation is ~440 m)

        # Any airfield (LSMD), Switzerland — aerodrome reference point.
        # lat = -17.3986  # in deg, positive north, negative south
        # lon = 38.6486  # in deg, positive east, negative west
        # alt = 200000  # entry/drop altitude in m (airfield elevation is ~440 m)

        # Initial velocity is given in GEODETIC coordinates. Same caveat
        # as COORDINATE_SYSTEM_POS above applies here.
        # Initial velocity (vx, vy, vz) in m/s, rocket dropped at 70 m/s
        # horizontal in x and y and -280 m/s vertical velocity.
        position_init = np.array([lat, lon, alt], dtype=np.float64)
        velocity_init = np.array(
            [100.0, 0.0, 0.0], dtype=np.float64
        )  # vE, vN, vU in m/s

        velocity_vec = RealState.convert_velocity_geodetic_to_cartesian(
            velocity_init, lat, lon
        )
        position_vec = RealState.convert_geodetic_to_cartesian(position_init)
        # Scale factor for plotting, to convert from meters to
        # kilometers for position and from m/s to km/s for velocity, for
        # better visualization of the trajectory and velocity.
        scale_fac = 1
        # Default wind conditions: 15 m/s wind speed from 270 degrees
        # (north to south), not used if using ERA5 wind data.
        wind_field_conditions = (
            wind_field_conditions
            if params["wind"] == "default"
            else [
                0.0,
                0.0,
            ]  # for consistency fallback. If the position is not at ERA5 modelled coordinates, the wind will be set to zero.
        )

    # Initial position and velocity
    initial_conditions = [position_vec, velocity_vec]
    print(
        "Initial conditions - Position (x, y, z): ",
        position_vec,
        "Velocity (vx, vy, vz): ",
        velocity_vec,
    )

    logger = Logging(
        n_steps, initial_conditions_geodetic=[position_init, velocity_init]
    )
    # Initialize the rocket simulation with the given parameters and initial conditions
    rocket_simulation = Rocket(
        initial_conditions=initial_conditions,
        params=params,
        logger=logger,
        wind_field=WindField(
            start_time="2026-06-05T13:00:00",
            params=params,
            logger=logger,
            wind_field_conditions=wind_field_conditions,
        ),
    )

    # Example usage for default wind parameters - 15 m/s wind speed from
    # 270 degrees (north to south)

    _ = main(
        n_steps,
        rocket_simulation,
        verbose=params.get("verbose"),
    )

    scale_fac = 1 if params["mode"] == "realistic" or params["wind"] == "ERA5" else 1000
    # set a limit for the horizontal trajectory plot to investigate
    # initial trajectory.
    if params["mode"] == "realistic" or params["wind"] == "ERA5":
        lim = None
    else:
        lim = 20

    plot_results(
        rocket_simulation.logger.get_full_state(),
        params=rocket_simulation.params,
        lim=lim,
        scale_fac=scale_fac,
    )

# TODO: Create test functions, test cases to test each function and class in the codebase.
# Check unit testing frameworks like pytest or unittest for Python.
# Fix typehinting reviewed from Claude. Check out mypy for type checking.
# Improve the wind data model loading and Monte Carlo perturbations. Add more realistic wind data and
# perturbations to the wind model. Maybe atmospheric model, for density, etc.
# Check specifically the J2 term physics and if it matches the geodetic model from pymap3d.
#
