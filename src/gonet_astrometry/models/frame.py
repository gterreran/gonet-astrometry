"""Image and observing-metadata models."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class ObserverLocation:
    """Geodetic observing location.

    Parameters
    ----------
    latitude_deg
        Geodetic latitude in degrees, positive northward.
    longitude_deg
        Longitude in degrees, positive eastward.
    elevation_m
        Elevation above the reference ellipsoid in metres.

    Raises
    ------
    ValueError
        If latitude or longitude lies outside its valid range.
    """

    latitude_deg: float
    longitude_deg: float
    elevation_m: float = 0.0

    def __post_init__(self) -> None:
        if not -90.0 <= self.latitude_deg <= 90.0:
            raise ValueError("latitude_deg must lie in [-90, 90]")
        if not -180.0 <= self.longitude_deg <= 180.0:
            raise ValueError("longitude_deg must lie in [-180, 180]")


@dataclass(frozen=True, slots=True)
class ImageMetadata:
    """Metadata required to interpret a GONet exposure astrometrically.

    Parameters
    ----------
    exposure_start
        Timezone-aware UTC time at which the exposure began.
    exposure_duration_s
        Exposure duration in seconds. The astrometric epoch is evaluated at
        the exposure midpoint unless a future rolling-readout model overrides
        that convention.
    location
        Geographic observing location.
    source_path
        Optional path from which the image and metadata were loaded.
    sensor_orientation
        Human-readable description of any rotation, reflection, or crop that
        has already been applied. The default assumes native sensor order.

    Raises
    ------
    ValueError
        If ``exposure_start`` is timezone-naive or ``exposure_duration_s`` is
        not strictly positive.

    Notes
    -----
    This model deliberately distinguishes exposure start from exposure
    midpoint. File creation and modification timestamps must not be used as
    substitutes without an explicit loader policy.
    """

    exposure_start: datetime
    exposure_duration_s: float
    location: ObserverLocation
    source_path: Path | None = None
    sensor_orientation: str = "native"

    def __post_init__(self) -> None:
        if self.exposure_start.tzinfo is None:
            raise ValueError("exposure_start must be timezone-aware")
        if self.exposure_duration_s <= 0:
            raise ValueError("exposure_duration_s must be strictly positive")

    @property
    def exposure_midpoint(self) -> datetime:
        """Return the timezone-aware midpoint of the exposure."""
        return self.exposure_start + timedelta(seconds=self.exposure_duration_s / 2)


@dataclass(frozen=True, slots=True)
class ImageFrame:
    """A native GONet image and the metadata needed to interpret it.

    Parameters
    ----------
    data
        Two-dimensional native sensor array. Bayer mosaics should remain in
        their original pixel geometry and should not be demosaiced by loaders.
    metadata
        Exposure and location metadata.
    mask
        Optional Boolean mask with ``True`` values for invalid or excluded
        pixels. It must have the same shape as ``data``.

    Raises
    ------
    ValueError
        If ``data`` is not two-dimensional or if ``mask`` has a different
        shape.
    """

    data: NDArray[np.generic]
    metadata: ImageMetadata
    mask: NDArray[np.bool_] | None = None

    def __post_init__(self) -> None:
        if self.data.ndim != 2:
            raise ValueError("ImageFrame.data must be a two-dimensional array")
        if self.mask is not None and self.mask.shape != self.data.shape:
            raise ValueError("ImageFrame.mask must match ImageFrame.data shape")

    @property
    def shape(self) -> tuple[int, int]:
        """Return the image shape as ``(rows, columns)``."""
        return self.data.shape[0], self.data.shape[1]
