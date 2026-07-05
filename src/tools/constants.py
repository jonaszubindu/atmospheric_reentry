import numpy as np

EARTH_RADIUS: float = 6371000  # Earth's radius in meters
G0: float = 9.80665  # Standard gravity at sea level in m/s^2
SEA_LEVEL_AIR_DENSITY: float = 0.0  # 1.225  # Air density at sea level in kg/m^3
SEA_LEVEL_PRESSURE: float = 101325  # Pressure at sea level in Pascals
GM: float = 3.986004418e14  # WGS84 defining constant, m^3/s^2
RE: float = 6378137.0  # WGS84 semi-major axis, m (defined exact)
J2: float = 1.08262668e-3  # Earth's dynamic form factor (EGM96)
OMEGA_E: float = 7.292115e-5  # WGS84 Earth rotation rate, rad/s  (you have this)
EARTH_ROTATION: np.ndarray = np.array([0, 0, OMEGA_E])  # Earth's rotation axis
# typical scale height of the atmosphere in meters,
# used for air density calculations. Can be adjusted or replaced at a later point
# with a more accurate model
scale_height: float = 8500
