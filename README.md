Copyright 2026 Jonas Zbinden
This software is licensed under the terms of the MIT GNU AGPLv3 License which can be obtained at https://opensource.org/licenses/MIT or
from the LICENSE file in the root directory of this project.

This software is provided "as is", without warranty of any kind, express or implied, including but not limited to the warranties of merchantability,
fitness for a particular purpose and non-infringement. In no event shall the authors or copyright holders be liable for any claim,
damages or other liability, whether in an action of contract, tort or otherwise, arising from, out of or in connection with the software or the
use or other dealings in the software.

This code contains the core_functions to run the rocket reentry model, including the equations of motion, drag force calculation, gravitational acceleration,
and wind field interpolation. The simulate_rocket class simulates the rocket's flight using the equations of motion, while the state_estimation class can be
expanded to include sensor noise and filtering. The logging class logs the state of the rocket at each time step for analysis, debugging and visualization.

The toolkit is to run rocket reentry simulations with varying degree of realism.

To install the toolkid please create a virtual environment with

python -m venv myenv

activate it with

source myenv/bin/activate

cd into the code directory with the .toml file and install the repo with

pip install .

Check your installation by running:

python -c "import atmospheric_reentry; print('dev install OK')"

If any of these steps don't work, please email: jonas_zbinden@bluewin.ch


The main functions are stored in "main.py". To change the degree of realism, there are two specific keywords that can be set in a
param dictionary within main.py.

Set "mode" to "realistic" to run a simulation on a spherical Earth, with a geodetic coordinate system and ECEF system
to propagate the equations of motion. The code automatically switches between Cartesian and geodetic coordinates for different tasks.
Realistic additionally includes realistic gravity, air resistance changing with altitude. Additionally, there is a function that would
include the Coriolis and centrifugal forces, but this function is not implemented into the EOM's yet.

A wind model can be predownloaded and interpolated at the required positions. To activate it, set the keyword "wind" to "ERA5".
The wind model can only be used when running main.py.

All results are plotted with a plotting routine that automatically adjusts for the realistic case to show coordinates, and in the simple case km and m/s.

The routine Monte_carlo_run.py runs in parallel simulations with varying degrees of initial wind. The final graphic depicts the positions where the rocket would impact.
In the case of the realistic simulation runs, an initial position in coordinates has to be chosen. For now, a default value is set for lon and lat.

The tool is still a work in progress and has not been fully tested yet.
