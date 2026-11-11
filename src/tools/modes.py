"""
The two simulation modes.

The tool runs in one of two modes, selected with ``params["mode"]``. Both
modes share all physics that does not depend on the reference frame, and
differ only in the frame and the environment models.

``"simplified"`` is a test box
    A flat, non-rotating Earth in plain Cartesian coordinates: x and y are
    horizontal (x east, y north) and z is the altitude. Gravity is the
    constant G0 along -z, the air density is the constant sea-level value and
    the wind is one constant horizontal vector. There are no geodetic
    coordinates, no ERA5 wind and no map.
    Use it to develop and test frame-independent parts on their own, such as
    aerodynamic models, the parachute logic, solver settings and control
    loops. Exact solutions exist here (free fall, terminal velocity), so
    results can be checked by hand.

``"realistic"`` is the reentry model
    A rotating WGS84 Earth with the state in ECEF coordinates. Gravity
    includes the J2 term plus the Coriolis and centrifugal accelerations, the
    air density falls off exponentially with the geodetic altitude, and the
    wind is given in local east/north/up (constant or interpolated from
    ERA5) and rotated into ECEF. ERA5 wind and the map need this mode.

Shared by both modes
    The state layout [position, velocity], the equations of motion, the drag
    law, the parachute switch, the solver, the logging and the main loop. A
    model that only uses these inputs and outputs behaves the same in both
    modes, so it can be tested in simplified mode and flown in realistic
    mode.

Where the modes separate in the code
    - ``Rocket._setup_simplified_mode`` and ``Rocket._setup_realistic_mode``
      (simulate_rocket.py) bind the altitude, gravity, air density and wind
      lookup of the mode. New frame- or environment-dependent models belong
      there, and nowhere else in the dynamics.
    - ``Logging.log_state`` fills the geodetic columns in realistic mode only.
    - ``plot_results`` (utils.py) picks the axes for the mode.
    - main.py and monte_carlo_run.py set the initial conditions for the mode.
"""

MODES = ("simplified", "realistic")


def validate_mode(params: dict) -> str:
    """Check ``params["mode"]`` and return it.

    Raises ValueError for an unknown mode, and for ERA5 wind in simplified
    mode: ERA5 wind is given on the real Earth and needs the realistic mode.
    """
    mode = params.get("mode")
    if mode not in MODES:
        raise ValueError(f"Invalid mode: {mode!r}. Must be one of {MODES}.")
    if mode == "simplified" and params.get("wind") == "ERA5":
        raise ValueError(
            "ERA5 wind needs the realistic mode. The simplified mode only "
            "supports the constant default wind."
        )
    return mode
