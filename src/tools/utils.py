#!/usr/bin/env python3

import os
import pathlib
import warnings

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
import pandas as pd
from typing import Optional
import numpy.typing as npt

from .state_estimation import RealState

from . import constants as const


class PhysicsFunctions(RealState):
    """A class containing various physics functions used in the rocket
    simulation. This class is designed to be flexible and can be extended
    with additional physics functions as needed."""

    def __init__(self) -> None:
        """Initialize the PhysicsFunctions class."""
        super().__init__()  # Initialize the RealState class

    def _simple_airdensity(self, state: npt.NDArray[np.float64]) -> np.float64:
        """Return a constant air density for the simplified model."""
        # Air density at sea level in kg/m^3
        return np.float64(const.SEA_LEVEL_AIR_DENSITY)

    def _realistic_airdensity(self, state: npt.NDArray[np.float64]) -> np.float64:
        """Compute air density based on altitude using an exponential model."""
        altitude = self._calc_alt_geodetic(state)
        return np.float64(
            const.SEA_LEVEL_AIR_DENSITY * np.exp(-altitude / const.scale_height)
        )

    def _simple_gravity(
        self, state: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Simple model for gravitational acceleration, assuming constant
        gravity at sea level."""
        # Gravitational acceleration in m/s^2, pointing downwards in z.
        return np.array([0, 0, -const.G0], dtype=np.float64)

    def _calc_alt_simpl(self, state: npt.NDArray[np.float64]) -> np.float64:
        """Calculate altitude above the surface of the Earth based on the
        position vector, assuming a flat Earth for simplicity."""
        # Assuming z-component of position is altitude for the simplified model.
        return state[2]

    def _calc_alt_geodetic(self, state: npt.NDArray[np.float64]) -> np.float64:
        """Calculate altitude above the surface of the Earth based on the
        position vector, using geodetic coordinates."""
        return self.convert_cartesian_to_geodetic(state[:3])[2]

    def _gravitational_acceleration_ecef(
        self, state: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Compute the forces arising due to Earth's rotation. Should only be used in ECEF
        coordinates."""
        position = state[:3]
        velocity = state[3:]
        # Compute ellipsoidal gravity with J2 perturbation and add pseudo forces due to Earth's rotation.
        x, y, z = position
        r = np.linalg.norm(position)
        if r == 0:
            raise ValueError(
                "Position vector cannot be zero for gravitational acceleration calculation."
            )
        k = 1.5 * const.J2 * (const.RE / r) ** 2
        z2r2 = 5.0 * z**2 / r**2
        ax = -const.GM * x / r**3 * (1.0 - k * (z2r2 - 1.0))
        ay = -const.GM * y / r**3 * (1.0 - k * (z2r2 - 1.0))
        az = -const.GM * z / r**3 * (1.0 - k * (z2r2 - 3.0))
        acc = np.array([ax, ay, az], dtype=np.float64)

        coriolis_acc = -2 * np.cross(const.EARTH_ROTATION, velocity)
        centrifugal_acc = -np.cross(
            const.EARTH_ROTATION, np.cross(const.EARTH_ROTATION, position)
        )
        acc += coriolis_acc + centrifugal_acc

        return acc

    def compute_drag_force(
        self,
        state: npt.NDArray[np.float64],
        wind_velv: npt.NDArray[np.float64],
        air_density: np.float64,
        drag_coeff: float,
        cross_sectional_area: float,
    ) -> npt.NDArray[np.float64]:
        """Calculate the drag force with a vector opposite to velocity."""
        velocity = state[3:]  # Extract velocity vector from state
        true_air_velocity = velocity - wind_velv  # correct for local wind
        speed = np.linalg.norm(true_air_velocity)
        if speed == 0:
            return np.zeros(3, dtype=np.float64)  # No drag if no relative motion
        return (
            -0.5
            * air_density
            * speed**2
            * drag_coeff
            * cross_sectional_area
            * true_air_velocity
            / speed
        )

    def _alt_to_pres(self, state: npt.NDArray[np.float64]) -> float:
        """Simple model to convert altitude to pressure using the barometric
        formula."""
        altitude = self._calc_alt_geodetic(state)
        return const.SEA_LEVEL_PRESSURE * np.exp(-altitude / const.scale_height)

    @staticmethod
    def _press_to_alt(pressure: float) -> float:
        """Simple model to convert pressure to altitude using the barometric
        formula."""
        return -const.scale_height * np.log(pressure / const.SEA_LEVEL_PRESSURE)


###############################################################


def geopotential_to_altitude(geopotential: float) -> float:
    """Convert geopotential to altitude using the formula:
    altitude = geopotential / g0, where g0 is the standard gravity at
    sea level (9.81 m/s^2)."""
    return geopotential / const.G0


def altitude_to_geopotential(altitude: float) -> float:
    """Convert altitude to geopotential using the formula:
    geopotential = altitude * g0, where g0 is the standard gravity at
    sea level (9.81 m/s^2): ERA5,
    https://software.ecmwf.int/wiki/display/IFS/CY41R1+Official+IFS+Documentation
    """
    return altitude * const.G0


def _get_compass_wind(wind_direction: float) -> float:
    """Convert cartesian wind direction to compass degrees, assuming true
    north up (opposite to y-axis) and east to the right (x-axis)."""
    if wind_direction <= 90:
        return 90 - wind_direction
    else:
        return 360 - (wind_direction - 90)


def _get_cartesian_wind(wind_direction: float) -> float:
    """Convert compass wind direction to cartesian degrees,
    the wind direction is given in degrees counterclockwise from the x-axis
    (east)."""
    if wind_direction < 90:
        return -wind_direction + 90
    elif wind_direction == 90:
        return 0
    else:
        return (360 - wind_direction) + 90


def plot_results(
    out_results: pd.DataFrame,
    params: dict,
    lw: float = 0.5,
    lim: Optional[int] = 100,
    scale_fac: float = 1000,
) -> plt.Figure:
    # logger not used yet
    fig = plt.figure(figsize=(24, 7), dpi=150, constrained_layout=True)
    ax1 = plt.subplot(1, 3, 1)
    ax2 = plt.subplot(1, 3, 2)
    ax3 = plt.subplot(1, 3, 3)
    ax11 = ax1.twinx()

    if params["wind"] != "Monte Carlo":
        t = out_results["time"]
        if params["mode"] == "simplified":
            altitude = out_results["altitude"]
            velocity = np.vstack(
                (
                    out_results["velocity cartesian x"],
                    out_results["velocity cartesian y"],
                    out_results["velocity cartesian z"],
                )
            ).T
            positions = np.vstack(
                (
                    out_results["position cartesian x"],
                    out_results["position cartesian y"],
                )
            ).T
        elif params["mode"] == "realistic":
            altitude = out_results["position geodetic alt"]
            velocity = np.vstack(
                (
                    out_results["velocity geodetic East"],
                    out_results["velocity geodetic North"],
                    out_results["velocity geodetic alt"],
                )
            ).T
            positions = np.vstack(
                (
                    out_results["position geodetic lon"],
                    out_results["position geodetic lat"],
                )
            ).T
        else:
            raise ValueError(
                f"Invalid mode: {params['mode']}. Must be 'simplified' or 'realistic'."
            )

        # actual plotting
        ax1.plot(t, altitude, label="Altitude (m)", color="b", lw=lw)
        ax11.plot(t, velocity[:, 2], label="Vertical Velocity (m/s)", color="r", lw=lw)

        ax2.plot(
            positions[:, 0] / scale_fac,
            positions[:, 1] / scale_fac,
            label="Position (km)",
            color="b",
            lw=lw,
        )
        ax2.scatter(
            positions[-1, 0] / scale_fac,
            positions[-1, 1] / scale_fac,
            color="b",
            s=lw * 2,
        )

        ax3.plot(
            velocity[:, 0],
            velocity[:, 1],
            label="Horizontal Velocity (m/s)",
            color="r",
            lw=lw,
        )
        ax3.scatter(
            velocity[:, 0],
            velocity[:, 1],
            color="r",
            s=0.5 * lw,
        )

    elif params["wind"] == "Monte Carlo":
        for out in out_results:
            if params["mode"] == "realistic":
                t = out["time"]
                altitude = out["position geodetic alt"]
                velocity = np.vstack(
                    (
                        out["velocity geodetic East"],
                        out["velocity geodetic North"],
                        out["velocity geodetic alt"],
                    )
                ).T
                positions = np.vstack(
                    (
                        out["position geodetic lon"],
                        out["position geodetic lat"],
                    )
                ).T
            elif params["mode"] == "simplified":
                t = out["time"]
                altitude = out["altitude"]
                velocity = np.vstack(
                    (
                        out["velocity cartesian x"],
                        out["velocity cartesian y"],
                        out["velocity cartesian z"],
                    )
                ).T
                positions = np.vstack(
                    (
                        out["position cartesian x"],
                        out["position cartesian y"],
                    )
                ).T
            else:
                raise ValueError(
                    f"Invalid mode: {params['mode']}. Must be 'simplified' or 'realistic'."
                )

            # actual plotting
            ax1.plot(t, altitude, label="Altitude (m)", color="b", lw=lw)
            ax11.plot(
                t, velocity[:, 2], label="Vertical Velocity (m/s)", color="r", lw=lw
            )

            ax2.plot(
                positions[:, 0] / scale_fac,
                positions[:, 1] / scale_fac,
                label="Position (km)",
                color="b",
                lw=lw,
            )
            ax2.scatter(
                positions[-1, 0] / scale_fac,
                positions[-1, 1] / scale_fac,
                color="b",
                s=2 * lw,
            )

            ax3.plot(
                velocity[:, 0],
                velocity[:, 1],
                label="Horizontal Velocity (m/s)",
                color="r",
                lw=lw,
            )
            ax3.scatter(
                velocity[:, 0],
                velocity[:, 1],
                color="r",
                s=0.5 * lw,
            )

    ax1.set_title("Rocket altitude over Time")
    ax2.set_title("Rocket Trajectory (X-Y Position)")
    ax3.set_title("Rocket TAS (True Airspeed) (X-Y)")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Altitude (m)")
    ax11.set_ylabel("Vertical Velocity (m/s)")
    if scale_fac == 1:
        ax2.set_xlabel("Longitude (deg)")
        ax2.set_ylabel("Latitude (deg)")
        axrange = max(
            np.abs(positions[:, 0].min() - positions[:, 0].max()),
            np.abs(positions[:, 1].min() - positions[:, 1].max()),
        )
        ax2.set_xlim(
            (
                -axrange + positions[:, 0].mean(),
                axrange + positions[:, 0].mean(),
            )
        )
        ax2.set_ylim(
            (
                -axrange + positions[:, 1].mean(),
                axrange + positions[:, 1].mean(),
            )
        )
        # Format ticks as degrees via a formatter (rather than overwriting
        # tick labels on an auto locator, which mislabels on redraw).
        # Precision adapts to the span so a tightly-zoomed track does not show
        # the same rounded value on every tick.
        deg_dec = _deg_decimals(2.0 * axrange)
        ax2.xaxis.set_major_locator(MaxNLocator(5))
        ax2.yaxis.set_major_locator(MaxNLocator(5))
        ax2.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.{deg_dec}f}°"))
        ax2.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.{deg_dec}f}°"))
        for label in ax2.get_xticklabels():
            label.set_rotation(45)

    else:
        ax2.set_xlabel("X Position (km)")
        ax2.set_ylabel("Y Position (km)")
    if lim is not None:
        ax2.set_xlim(-lim, lim)
        ax2.set_ylim(-lim, lim)
    else:
        axrange = max(
            np.abs(velocity[:, 0].min() - velocity[:, 0].max()),
            np.abs(velocity[:, 1].min() - velocity[:, 1].max()),
        )
        ax3.set_xlim(
            (
                -axrange + velocity[:, 0].mean(),
                axrange + velocity[:, 0].mean(),
            )
        )
        ax3.set_ylim(
            (
                -axrange + velocity[:, 1].mean(),
                axrange + velocity[:, 1].mean(),
            )
        )
    ax3.set_xlabel("Velocity X (m/s)")
    ax3.set_ylabel("Velocity Y (m/s)")
    # ax1 and ax11 share the same box (twinx), so give them a single combined
    # legend instead of two overlapping ones.
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles11, labels11 = ax11.get_legend_handles_labels()
    ax1.legend(handles1 + handles11, labels1 + labels11, loc="upper right", fontsize=8)
    ax2.legend(loc="upper right", fontsize=8)
    ax3.legend(loc="upper right", fontsize=8)

    # Make every panel a square box (independent of the data scales).
    for panel in (ax1, ax2, ax3):
        panel.set_box_aspect(1)

    # Optional: draw an ICAO / aeronautical basemap behind the ground track.
    # Only meaningful in realistic mode (ax2 holds lon/lat) and for a single
    # run. Gated on params["basemap"] so offline/headless runs are unaffected.
    if (
        params.get("basemap")
        and params.get("mode") == "realistic"
        and params["wind"] != "Monte Carlo"
    ):
        add_icao_basemap(ax2, positions[:, 0], positions[:, 1], lw=lw)

    plt.show()

    return fig


# openAIP aeronautical (ICAO-style) airspace/navaid tile overlay. Needs a free
# API key; without it only the base map draws (no airspaces).
_ICAO_OVERLAY_URL = (
    "https://api.tiles.openaip.net/api/data/openaip/{z}/{x}/{y}.png?apiKey=OPENAIP_KEY"
)


def _load_openaip_key() -> Optional[str]:
    """Return the openAIP API key from the environment, or from a .env file in
    the working directory or up to three parent directories (matching the
    reentry_globe convention). Returns None if not found."""
    key = os.environ.get("OPENAIP_API_KEY")
    if key:
        return key
    here = pathlib.Path.cwd()
    for directory in [here, *list(here.parents)[:3]]:
        env_file = directory / ".env"
        if env_file.is_file():
            for line in env_file.read_text().splitlines():
                line = line.strip()
                if line.startswith("OPENAIP_API_KEY="):
                    return line.split("=", 1)[1].strip().strip("'\"")
    return None


def _deg_decimals(span_deg: float) -> int:
    """Number of decimal places so that ~5 ticks across `span_deg` degrees
    render as distinct labels. Fixed precision (e.g. .2f) collapses to
    repeated labels when the track is small; this scales with the span."""
    if span_deg <= 0 or not np.isfinite(span_deg):
        return 3
    spacing = span_deg / 5.0
    decimals = int(np.ceil(-np.log10(spacing))) + 1
    return int(np.clip(decimals, 2, 7))


def _basemap_zoom(
    lon: np.ndarray, lat: np.ndarray, zmin: int = 4, zmax: int = 12
) -> int:
    """Pick a web-map zoom level from the track's degree extent, clamped so a
    near-vertical drop (tiny extent) does not request absurdly deep tiles."""
    span = float(max(lon.max() - lon.min(), lat.max() - lat.min(), 1e-4))
    zoom = int(np.floor(np.log2(360.0 / span)))
    return int(np.clip(zoom, zmin, zmax))


def add_icao_basemap(
    ax: plt.Axes, lon: np.ndarray, lat: np.ndarray, lw: float = 0.5
) -> None:
    """Re-plot a lon/lat ground track in Web Mercator (EPSG:3857) and draw an
    aeronautical basemap behind it, undistorted, on a square panel with
    degree tick labels.

    contextily and pyproj are optional dependencies; if either is missing (or
    the tile server is unreachable), this degrades gracefully to a warning so
    headless/offline runs never break. Airspaces come from the openAIP overlay,
    which requires OPENAIP_API_KEY (env var or a .env file); without it only
    the base map is drawn.
    """
    try:
        import contextily as cx
        from pyproj import Transformer
    except ImportError:
        warnings.warn(
            "basemap requested but contextily/pyproj are not installed; "
            "keeping the plain ground-track plot. Install the basemap with: "
            "pip install 'atmospheric-reentry[basemap]'"
        )
        return  # leave the plain lon/lat plot already drawn on the axis

    # Persist fetched tiles to disk so repeat views of the same region+zoom
    # are served from cache and work offline after the first fetch.
    cache_dir = pathlib.Path.home() / ".cache" / "atmospheric_reentry" / "tiles"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cx.set_cache_dir(str(cache_dir))

    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)

    # Reproject the track to Web Mercator so the map is metric and isotropic
    # (no cos(lat) distortion).
    to_merc = Transformer.from_crs(4326, 3857, always_xy=True)
    to_deg = Transformer.from_crs(3857, 4326, always_xy=True)
    xm, ym = to_merc.transform(lon, lat)

    # Square, padded extent (in metres) centred on the track.
    cx_c = 0.5 * (xm.min() + xm.max())
    cy_c = 0.5 * (ym.min() + ym.max())
    half = max(0.5 * (xm.max() - xm.min()), 0.5 * (ym.max() - ym.min()))
    half = max(half * 1.3, 1000.0)  # pad ~30%, at least 1 km
    xmin, xmax = cx_c - half, cx_c + half
    ymin, ymax = cy_c - half, cy_c + half

    zoom = _basemap_zoom(lon, lat)
    api_key = _load_openaip_key()
    if api_key is None:
        warnings.warn(
            "basemap: no OPENAIP_API_KEY found, so airspaces are not drawn "
            "(base map only). Set the OPENAIP_API_KEY environment variable or "
            "add it to a .env file to show ICAO airspaces."
        )

    # Fetch the tiles FIRST (from cache if available). If this fails — no
    # internet and not cached — keep the plain lon/lat ground-track plot that
    # is already on the axis, so the basemap option still produces a figure.
    try:
        base_img, base_ext = cx.bounds2img(
            xmin,
            ymin,
            xmax,
            ymax,
            zoom=zoom,
            source=cx.providers.OpenStreetMap.Mapnik,
        )
        overlay = None
        if api_key:
            overlay = cx.bounds2img(
                xmin,
                ymin,
                xmax,
                ymax,
                zoom=zoom,
                source=_ICAO_OVERLAY_URL.replace("OPENAIP_KEY", api_key),
            )
    except Exception as exc:  # offline + uncached, or tile server error
        warnings.warn(
            f"Basemap tiles unavailable ({exc}); keeping the plain lon/lat "
            "ground-track plot."
        )
        return

    # Tiles are in hand: replace the plain plot with the mapped, undistorted
    # ground track.
    ax.clear()
    ax.imshow(base_img, extent=base_ext, interpolation="bilinear", zorder=0)
    if overlay is not None:
        ov_img, ov_ext = overlay
        ax.imshow(ov_img, extent=ov_ext, interpolation="bilinear", zorder=1)
    ax.plot(xm, ym, color="b", lw=max(lw, 1.0), label="Ground track", zorder=2)
    ax.scatter(xm[-1], ym[-1], color="b", s=12, label="Impact", zorder=3)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.set_box_aspect(1)

    # Sensical ticks: label Web-Mercator metres back as lon/lat degrees
    # (in Mercator, lon depends only on x and lat only on y). Precision adapts
    # to the view span so a tightly-zoomed track does not show repeated labels.
    lon_span = abs(to_deg.transform(xmax, 0.0)[0] - to_deg.transform(xmin, 0.0)[0])
    lat_span = abs(to_deg.transform(0.0, ymax)[1] - to_deg.transform(0.0, ymin)[1])
    xdec = _deg_decimals(lon_span)
    ydec = _deg_decimals(lat_span)
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda x, _: f"{to_deg.transform(x, 0.0)[0]:.{xdec}f}°")
    )
    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda y, _: f"{to_deg.transform(0.0, y)[1]:.{ydec}f}°")
    )
    for label in ax.get_xticklabels():
        label.set_rotation(45)

    ax.set_title("Ground track over aeronautical map")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend(loc="upper right", fontsize=8)
