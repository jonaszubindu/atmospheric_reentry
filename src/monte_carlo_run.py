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
"""

import numpy as np
from joblib import Parallel, delayed

from main import main
from tools.utils import plot_results
from tools.windfield import WindField
from tools.state_estimation import RealState
from tools.simulate_rocket import Rocket
from tools.logging import Logging

nsamples = 200
# Sample initial wind conditions: the compass direction the wind blows from
# and the wind speed.
wind_dir = np.array([np.random.uniform(0, 360) for i in range(nsamples)])  # deg
wind_vel = np.array([np.random.uniform(0, 20) for i in range(nsamples)])  # m/s

# Rocket and simulation parameters, see main.py for a description of each.
params: dict = {
    "mass": 700,  # mass of the rocket in kg, assumed to be a pointmass
    "drag_coefficient": 1.4,  # drag coefficient rocket (dimensionless)
    "drag_coefficient_parachute": 2.0,  # drag coeff. with parachute
    "cross_sectional_area": 1.14,  # cross-sectional area in m^2
    "cross_sectional_area_parachute": 18,  # parachute area in m^2
    "parachute_opening_altitude": 10000,  # parachute opening altitude in m
    "tmax": 3000,  # maximum simulation time in seconds
    "max_step": 0.3,  # maximum stepsize in seconds
    # Monte Carlo runs this will be overridden.
    "wind": "default",
    # 'realistic' or 'simplified', see tools/modes.py.
    "mode": "realistic",
}

# Initialize parameters and initial conditions
# Step budget per run, which is also the size of the log buffers (see main.py).
n_steps = 2 * int(np.ceil(params["tmax"] / params["max_step"]))

# Set wind parameter to indicate that we are using Monte Carlo sampled
# wind conditions
params["wind"] = "Monte Carlo"

if params["mode"] == "realistic":
    lon = 360 - 75
    lat = 38.5
    alt = 60000
    # Initial position (x, y, z) in meters, rocket dropped at 60 km
    # height.
    position_init = np.array([lat, lon, alt])
    velocity_init = np.array([70, 70, -280])

    velocity_vec = RealState.convert_velocity_geodetic_to_cartesian(
        velocity_init, lat, lon
    )
    position_vec = RealState.convert_geodetic_to_cartesian(position_init)
    # Scale factor for plotting, to convert from meters to kilometers
    # for position and from m/s to km/s for velocity, for better
    # visualization of the trajectory and velocity.
    scale_fac = 1
else:
    # Initial position (x, y, z) in meters, rocket dropped at 60 km
    # height.
    position_vec = np.array([0, 0, 60000])
    # Initial velocity (vx, vy, vz) in m/s, rocket dropped at 70 m/s
    # horizontal in x and y and -280 m/s vertical velocity.
    velocity_vec = np.array([70, 70, -280])
    # No geodetic frame on a flat Earth; the logger only reads these in
    # ERA5 mode.
    position_init = position_vec
    velocity_init = velocity_vec
    # Scale factor for plotting, to convert from meters to kilometers
    # for position
    scale_fac = 1000


initial_conditions = [position_vec, velocity_vec]
start_time = "2026-06-10T13:00:00"  # Start time for the simulation
print(
    "Initial conditions - Position (x, y, z): ",
    position_vec,
    "Velocity (vx, vy, vz): ",
    velocity_vec,
)
# Run main with different winds in parallel using joblib. Each run gets its
# own logger, wind field and rocket.
out = Parallel(n_jobs=-1, verbose=10)(
    delayed(main)(
        n_steps,
        rocket_simulation=Rocket(
            initial_conditions=initial_conditions,
            params=params,
            logger=Logging(
                n_steps, initial_conditions_geodetic=[position_init, velocity_init]
            ),
            wind_field=WindField(
                start_time=start_time,
                params=params,
                logger=None,
                wind_field_conditions=[wind_dir[i], wind_vel[i]],
            ),
        ),
        verbose=False,
    )
    for i in range(nsamples)
)


# Plot the trajectories of all runs in the same figure
plot_results(out, lw=0.1, params=params, lim=None, scale_fac=scale_fac)
