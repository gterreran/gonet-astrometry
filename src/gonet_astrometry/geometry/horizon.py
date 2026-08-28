"""Local-horizon coordinate helpers for absolute camera orientation.

The local frame used throughout this module is right handed ``ENU``:
``+x`` points east, ``+y`` points north, and ``+z`` points toward the zenith.
Azimuth follows the Astropy convention, increasing eastward from north.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
from numpy.typing import NDArray

from gonet_astrometry.catalogs.base import CatalogStar
from gonet_astrometry.models.frame import ObserverLocation


def ncp_enu_vector(latitude_deg: float) -> NDArray[np.float64]:
    """Return the North Celestial Pole direction in the local ENU frame.

    Parameters
    ----------
    latitude_deg
        Geodetic observing latitude in degrees, positive northward.

    Returns
    -------
    numpy.ndarray
        Unit vector ``[east, north, up]`` pointing toward the NCP.

    Notes
    -----
    The NCP has azimuth zero and altitude equal to the observer latitude.
    At southern latitudes the returned direction therefore lies below the
    astronomical horizon, as expected.
    """
    latitude = np.deg2rad(float(latitude_deg))
    return np.asarray(
        [0.0, np.cos(latitude), np.sin(latitude)], dtype=np.float64
    )


def altaz_to_enu(
    azimuth_deg: NDArray[np.float64] | float,
    altitude_deg: NDArray[np.float64] | float,
) -> NDArray[np.float64]:
    """Convert altitude/azimuth coordinates to local ENU unit vectors.

    Parameters
    ----------
    azimuth_deg
        Azimuth in degrees eastward from north.
    altitude_deg
        Altitude in degrees above the horizon.

    Returns
    -------
    numpy.ndarray
        Array with shape ``broadcast(azimuth, altitude).shape + (3,)``.
    """
    azimuth, altitude = np.broadcast_arrays(
        np.deg2rad(np.asarray(azimuth_deg, dtype=np.float64)),
        np.deg2rad(np.asarray(altitude_deg, dtype=np.float64)),
    )
    cosine_altitude = np.cos(altitude)
    return np.stack(
        (
            cosine_altitude * np.sin(azimuth),
            cosine_altitude * np.cos(azimuth),
            np.sin(altitude),
        ),
        axis=-1,
    )


def enu_to_altaz(vector: NDArray[np.float64]) -> tuple[float, float]:
    """Return ``(azimuth_deg, altitude_deg)`` for one ENU direction.

    Parameters
    ----------
    vector
        Three-component ENU vector. The input need not already be normalized.

    Returns
    -------
    tuple[float, float]
        Azimuth eastward from north and altitude above the horizon, in degrees.
    """
    parsed = np.asarray(vector, dtype=np.float64)
    if parsed.shape != (3,):
        raise ValueError("ENU vector must have shape (3,)")
    norm = float(np.linalg.norm(parsed))
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("ENU vector must be finite and non-zero")
    unit = parsed / norm
    altitude = np.rad2deg(np.arcsin(np.clip(unit[2], -1.0, 1.0)))
    azimuth = np.mod(np.rad2deg(np.arctan2(unit[0], unit[1])), 360.0)
    return float(azimuth), float(altitude)


def catalog_stars_to_enu_rays(
    stars: tuple[CatalogStar, ...],
    epoch: datetime,
    location: ObserverLocation,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Transform catalog ICRS positions to geometric local-horizon rays.

    Parameters
    ----------
    stars
        Catalog stars carrying ICRS right ascension and declination.
    epoch
        Timezone-aware epoch at which the apparent local directions are needed.
    location
        Geodetic observing location.

    Returns
    -------
    rays, altitude_deg : tuple[numpy.ndarray, numpy.ndarray]
        ENU unit vectors and corresponding geometric altitudes.

    Notes
    -----
    Astropy's ``AltAz`` frame is evaluated with zero pressure so the catalog
    directions represent geometric/topocentric positions without an assumed
    atmospheric refraction correction.
    """
    if epoch.tzinfo is None:
        raise ValueError("Catalog transformation epoch must be timezone-aware")
    if not stars:
        return (
            np.empty((0, 3), dtype=np.float64),
            np.empty(0, dtype=np.float64),
        )

    from astropy import units as u  # type: ignore
    from astropy.coordinates import AltAz, EarthLocation, SkyCoord  # type: ignore
    from astropy.time import Time  # type: ignore

    sky = SkyCoord(
        ra=np.asarray([star.right_ascension_deg for star in stars]) * u.deg,
        dec=np.asarray([star.declination_deg for star in stars]) * u.deg,
        frame="icrs",
    )
    earth_location = EarthLocation.from_geodetic(
        lon=location.longitude_deg * u.deg,
        lat=location.latitude_deg * u.deg,
        height=location.elevation_m * u.m,
    )
    frame = AltAz(
        obstime=Time(epoch),
        location=earth_location,
        pressure=0.0 * u.hPa,
    )
    local = sky.transform_to(frame)
    azimuth = np.asarray(local.az.to_value(u.deg), dtype=np.float64)
    altitude = np.asarray(local.alt.to_value(u.deg), dtype=np.float64)
    return altaz_to_enu(azimuth, altitude), altitude
