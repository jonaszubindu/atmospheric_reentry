"""
Copyright 2026 Jonas Zbinden
This software is licensed under the terms of the MIT License
which can be obtained at https://opensource.org/licenses/MIT or from the
LICENSE file in the root directory of this project.

This software is provided "as is", without warranty of any kind, express
or implied, including but not limited to the warranties of
merchantability, fitness for a particular purpose and non-infringement.
In no event shall the authors or copyright holders be liable for any
claim, damages or other liability, whether in an action of contract,
tort or otherwise, arising from, out of or in connection with the
software or the use or other dealings in the software.

The code runs the rocket reentry model for multiple runs with different
initial wind conditions, and plots the trajectories of all runs in the
same figure for comparison. The wind is varied by sampling wind
directions from a uniform distribution between 0 and 360 degrees, and
wind speeds from a uniform distribution between 0 and 20 m/s. The
results are plotted in a single figure to visualize the impact of
different wind conditions on the rocket's trajectory.

The ensemble itself is run by tools.monte_carlo.run_monte_carlo, which other
programs can import directly; this script only sets the scenario and plots.
"""

import numpy as np

from tools.monte_carlo import default_n_steps, run_monte_carlo
from tools.utils import plot_results

nsamples = 200
# Uniform sampling ranges: the compass direction the wind blows from (deg)
# and the wind speed (m/s).
wind_dir_range = (0.0, 360.0)
wind_speed_range = (0.0, 20.0)
seed = None  # set an int for a reproducible ensemble

# Rocket and simulation parameters, see main.py for a description of each.
params: dict = {
    "mass": 700,  # mass of the rocket in kg, assumed to be a pointmass
    "drag_coefficient": 1.4,  # drag coefficient rocket (dimensionless)
    "drag_coefficient_parachute": 2.0,  # drag coeff. with parachute
    "cross_sectional_area": 1.14,  # cross-sectional area in m^2
    "cross_sectional_area_parachute": 18,  # parachute area in m^2
    "parachute_opening_altitude": 10000,  # parachute opening altitude in m
    "tmax": 3000,  # maximum simulation time in seconds
    # Maximum solver step in seconds. Accuracy is set by the error
    # tolerances (rtol, atol_position, atol_velocity; engine defaults), so
    # the step can be large where the flight is smooth.
    "max_step": 5.0,
    "output_interval": 0.5,  # log a state every 0.5 s (interpolated)
    # Monte Carlo sampled analytic wind (set by run_monte_carlo).
    "wind": "Monte Carlo",
    # 'realistic' or 'simplified', see tools/modes.py.
    "mode": "realistic",
}

if params["mode"] == "realistic":
    # Initial position (lat, lon, alt): rocket dropped at 60 km height.
    position_init = np.array([38.5, 360 - 75, 60000])
    # Initial velocity (east, north, up) in m/s.
    velocity_init = np.array([70, 70, -280])
    # Scale factor for plotting (positions already in metres on the globe).
    scale_fac = 1
else:
    # Initial position (x, y, z) in meters, rocket dropped at 60 km height.
    position_init = np.array([0, 0, 60000])
    # Initial velocity (vx, vy, vz) in m/s.
    velocity_init = np.array([70, 70, -280])
    # Scale factor for plotting, to convert from meters to kilometers.
    scale_fac = 1000

start_time = "2026-06-10T13:00:00"  # Start time for the simulation


if __name__ == "__main__":
    print(
        "Initial conditions - Position: ",
        position_init,
        "Velocity: ",
        velocity_init,
    )
    # Runs in parallel on all cores (n_jobs=-1); each run gets its own
    # logger, wind field and rocket.
    out, winds = run_monte_carlo(
        params,
        position_init,
        velocity_init,
        n_samples=nsamples,
        wind_dir_range=wind_dir_range,
        wind_speed_range=wind_speed_range,
        seed=seed,
        n_steps=default_n_steps(params),
        start_time=start_time,
        n_jobs=-1,
        verbose=10,
    )

    # Plot the trajectories of all runs in the same figure
    plot_results(out, lw=0.1, params=params, lim=None, scale_fac=scale_fac)
