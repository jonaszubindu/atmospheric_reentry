import numpy as np

EARTH_RADIUS: float = 6371000  # Earth's radius in meters
G0: float = 9.80665  # Standard gravity at sea level in m/s^2
SEA_LEVEL_AIR_DENSITY: float = 1.225  # Air density at sea level in kg/m^3
SEA_LEVEL_PRESSURE: float = 101325  # Pressure at sea level in Pascals
G: float = 6.67430e-11  # Gravitational constant in m^3 kg^-1 s^-2
M: float = 5.972e24  # Mass of the Earth in kg
EARTH_ROTATION: np.ndarray = np.array([0, 0, 7.2921159e-5])  # Earth's rotation axis
# typical scale height of the atmosphere in meters,
# used for air density calculations. Can be adjusted or replaced at a later point
# with a more accurate model
scale_height: float = 8500
