import numpy as np
import pandas

from tools.logging import Logging
from tools.simulate_rocket import Rocket
from tools.state_estimation import StateEstimation
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
        v_lat, v_lon, v_alt = rocket_simulation.get_velocity_geodetic()
        wind_vel_geodetic = wind_field.wind_velv

        print(
            f"Time: {time0:.2f} s, "
            f"Position (lat, lon, alt): "
            f"({lat:.2f}°, {lon:.2f}°, {alt:.2f} m), "
            f"Velocity (v_lat, v_lon, v_alt): "
            f"({v_lat:.2f}, {v_lon:.2f}, {v_alt:.2f}) m/s"
            f"Wind Velocity (v_wind_lat, v_wind_lon, v_wind_alt): "
            f"({wind_vel_geodetic[0]:.2f}, {wind_vel_geodetic[1]:.2f}, {wind_vel_geodetic[2]:.2f}) m/s"
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
    t: np.ndarray,
    rocket_simulation: Rocket,
    wind_field: WindField,
    wind_field_conditions: list,
    verbose: bool = False,
) -> pandas.DataFrame:
    """Run the rocket simulation with the given wind conditions."""

    for ii in range(len(t)):
        time0 = t[ii]
        time1 = t[ii + 1] if ii + 1 < len(t) else t[ii]
        # Update state and conditions
        rocket_simulation.update_state(
            [time0, time1], wind_field, wind_field_conditions
        )
        altitude = rocket_simulation.calc_altitude_abv_sea_level()
        if verbose and ii % 100 == 0:  # Print every 100 steps to reduce output
            verbose_message(rocket_simulation, wind_field, time0)

        if altitude < 0:
            # The rocket has hit the ground, stop the simulation
            break

    # Adjust time array to match the length of the state array
    if verbose:
        print("Simulation completed successfully.")

    return rocket_simulation.logger.get_full_state()


def menu() -> tuple[str, str]:
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

    return mode, wind_data


if __name__ == "__main__":
    # Get user input for simulation parameters
    mode, wind_data = menu()

    params = {
        # mass of the rocket in kg, assumed to be a pointmass
        "mass": 700,
        # drag coefficient rocket (dimensionless)
        "drag_coefficient": 1.4,
        # drag coefficient with parachute deployed (dimensionless)
        "drag_coefficient_parachute": 2.0,
        # cross-sectional area in m^2
        "cross_sectional_area": 1.14,
        # cross-sectional area of parachute in m^2
        "cross_sectional_area_parachute": 18,
        # which wind data to use; use "default" for default behaviour,
        # use "ERA5" for realistic wind data.
        "wind": wind_data,
        # use 'realistic' to use all realistic assumptions and
        # parameters, or use 'simplified' assumptions for testing and
        # debugging, e.g. no drag, constant gravity, etc.
        "mode": mode,
        # include pseudo-forces due to Earth's rotation in the
        # equations of motion
        "pseudo_forces": True,
        # print the state of the rocket at each time step for
        # debugging and analysis
        "verbose": True,
    }

    # Initialize parameters and initial conditions.
    # Time array from 0 to 3000 seconds with 10000 time steps was so far
    # stable in all simulation runs. At 3000 steps, oscillations show up
    # in the velocity and altitude.
    t = np.linspace(0, 3000, n_steps := 10000)
    # Default wind conditions: 270 degrees (north to south), 15 m/s wind
    # speed, not used if using ERA5 wind data. Degrees are given as
    # cartesian degrees, meaning 0 degrees is eastward, 90 degrees is
    # northward, 180 degrees is westward and 270 degrees is southward,
    # matching the coordinate system of the simulation.
    wind_field_conditions = [270, 15]

    if params["wind"] == "default" and params["mode"] == "simplified":
        # This for now also automatically assumes a flat earth.
        # Initial position (x, y, z) in meters, rocket dropped at 60 km
        # height.
        position_vec = np.array([0, 0, 60000])
        # Initial velocity (vx, vy, vz) in m/s, rocket dropped at 70 m/s
        # horizontal in x and y and -280 m/s vertical velocity.
        velocity_vec = np.array([70, 70, -280])
        wind_field_conditions = wind_field_conditions

    if params["mode"] == "realistic" or params["wind"] == "ERA5":
        # Initial coordinates: dummy values for a mid-latitude coastal
        # launch site.
        lon = 360 - 75
        lat = 38.5
        alt = 60000

        # Initial velocity is given in GEODETIC coordinates. Same caveat
        # as COORDINATE_SYSTEM_POS above applies here.
        # Initial velocity (vx, vy, vz) in m/s, rocket dropped at 70 m/s
        # horizontal in x and y and -280 m/s vertical velocity.
        position_init = np.array([lat, lon, alt])
        velocity_init = np.array([70, 70, -280])

        velocity_vec = StateEstimation.convert_velocity_geodetic_to_cartesian(
            velocity_init, lat, lon
        )
        position_vec = StateEstimation.convert_geodetic_to_cartesian(position_init)
        # Scale factor for plotting, to convert from meters to
        # kilometers for position and from m/s to km/s for velocity, for
        # better visualization of the trajectory and velocity.
        scale_fac = 1
        # Default wind conditions: 15 m/s wind speed from 270 degrees
        # (north to south), not used if using ERA5 wind data.
        wind_field_conditions = (
            wind_field_conditions if params["wind"] == "default" else None
        )

    # Initial position and velocity
    initial_conditions = [position_vec, velocity_vec]
    print(
        "Initial conditions - Position (x, y, z): ",
        position_vec,
        "Velocity (vx, vy, vz): ",
        velocity_vec,
    )
    logger = Logging(n_steps)  # Initialize the logger

    # Initialize the rocket simulation with the given parameters and initial conditions
    rocket_simulation = Rocket(
        initial_conditions=initial_conditions, params=params, logger=logger
    )

    # Initialize wind field with ERA5 data, parameters are not used
    # since the data is loaded from files in the windfield class
    wind_field = WindField(
        start_time="2026-06-05T13:00:00", params=params, logger=logger
    )

    # Example usage for default wind parameters - 15 m/s wind speed from
    # 270 degrees (north to south)

    _ = main(
        t,
        rocket_simulation,
        wind_field,
        wind_field_conditions=wind_field_conditions,
        verbose=params.get("verbose"),
    )

    scale_fac = 1 if params["mode"] == "realistic" or params["wind"] == "ERA5" else 1000
    # set a limit for the horizontal trajectory plot to investigate
    # initial trajectory.
    lim = None

    plot_results(
        logger.get_full_state(),
        figsize=(24, 5),
        params=rocket_simulation.params,
        lim=lim,
        scale_fac=scale_fac,
    )
