"""Adapter for the portable Grid Calibration geometric product.

The adapter deliberately depends only on the public ``grid_calibration`` API.
The Grid package is imported lazily so source detection and image-plane tracking
remain usable when Grid Calibration is not installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
from numpy.typing import NDArray

from gonet_astrometry.models.grid import GridCalibration

_EXPECTED_COORDINATE_CONVENTION = (
    "x=column,y=row;origin=upper-left;+x=right;+y=down;"
    "pixel-centers-at-integer-coordinates"
)


class _GridEvaluator(Protocol):
    """Structural subset of ``grid_calibration.GridCalibration`` used here."""

    sensor_width_px: int
    sensor_height_px: int
    image_coordinate_convention: str
    calibrated_angular_range_deg: tuple[float, float]

    def pixel_to_angle(self, x: Any, y: Any, **kwargs: Any) -> tuple[Any, Any]:
        """Invert pixel coordinates to nominal angular coordinates."""
        ...

    def angle_to_pixel(self, r_deg: Any, theta_deg: Any) -> tuple[Any, Any]:
        """Evaluate nominal angular coordinates in pixel space."""
        ...


@dataclass(frozen=True, slots=True)
class PixelRayConversion:
    """Grid-inversion result for a collection of sensor positions.

    Parameters
    ----------
    rays
        Unit Grid-frame rays with shape ``x.shape + (3,)``. Invalid rows are
        filled with ``NaN``.
    valid
        Boolean mask selecting coordinates that inverted inside the calibrated
        angular domain and reproduced the original pixel position.
    reprojection_error_px
        Pixel-space forward/inverse round-trip residual for every coordinate.
    r_deg, theta_deg
        Nominal polar angles returned by the Grid model.
    """

    rays: NDArray[np.float64]
    valid: NDArray[np.bool_]
    reprojection_error_px: NDArray[np.float64]
    r_deg: NDArray[np.float64]
    theta_deg: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class PortableGridTransform:
    """Pixel-to-ray transform backed by a portable Grid calibration artifact.

    Parameters
    ----------
    evaluator
        Public Grid Calibration object loaded from the portable artifact.
    inverse_reprojection_tolerance_px
        Maximum forward/inverse pixel-space residual accepted when converting
        a tracked detection to a Grid-frame ray.
    """

    evaluator: _GridEvaluator
    inverse_reprojection_tolerance_px: float = 1e-2

    def pixel_to_ray(
        self,
        x: NDArray[np.float64],
        y: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Convert sensor pixels to Grid-frame rays, requiring every point valid."""
        converted = self.convert(x, y)
        if not np.all(converted.valid):
            count = int(np.size(converted.valid) - np.count_nonzero(converted.valid))
            raise ValueError(
                f"Grid calibration could not invert {count} pixel coordinate(s) "
                "inside the calibrated domain"
            )
        return converted.rays

    def convert(
        self,
        x: NDArray[np.float64],
        y: NDArray[np.float64],
        *,
        reprojection_tolerance_px: float | None = None,
    ) -> PixelRayConversion:
        """Convert pixels to rays while retaining a per-coordinate validity mask."""
        x_array, y_array = np.broadcast_arrays(
            np.asarray(x, dtype=np.float64),
            np.asarray(y, dtype=np.float64),
        )
        r_value, theta_value = self.evaluator.pixel_to_angle(
            x_array,
            y_array,
            extrapolate=False,
            strict=False,
        )
        r_deg = np.asarray(r_value, dtype=np.float64)
        theta_deg = np.asarray(theta_value, dtype=np.float64)
        x_roundtrip, y_roundtrip = self.evaluator.angle_to_pixel(r_deg, theta_deg)
        x_model = np.asarray(x_roundtrip, dtype=np.float64)
        y_model = np.asarray(y_roundtrip, dtype=np.float64)
        error = np.hypot(x_model - x_array, y_model - y_array)

        _, r_max = self.evaluator.calibrated_angular_range_deg
        tolerance = (
            self.inverse_reprojection_tolerance_px
            if reprojection_tolerance_px is None
            else float(reprojection_tolerance_px)
        )
        if tolerance <= 0:
            raise ValueError("reprojection_tolerance_px must be positive")
        valid = (
            np.isfinite(r_deg)
            & np.isfinite(theta_deg)
            & np.isfinite(error)
            & (r_deg >= -1e-9)
            & (r_deg <= float(r_max) + 1e-9)
            & (error <= tolerance)
        )
        rays = angles_to_grid_rays(r_deg, theta_deg)
        rays = np.asarray(rays, dtype=np.float64)
        rays[~valid] = np.nan
        return PixelRayConversion(
            rays=rays,
            valid=np.asarray(valid, dtype=np.bool_),
            reprojection_error_px=np.asarray(error, dtype=np.float64),
            r_deg=r_deg,
            theta_deg=theta_deg,
        )

    def ray_to_angle(
        self,
        ray: NDArray[np.float64],
    ) -> tuple[float, float]:
        """Return ``(r_deg, theta_deg)`` for one Grid-frame unit vector."""
        vector = np.asarray(ray, dtype=np.float64)
        if vector.shape != (3,):
            raise ValueError("Grid-frame ray must have shape (3,)")
        norm = float(np.linalg.norm(vector))
        if not np.isfinite(norm) or norm <= 0:
            raise ValueError("Grid-frame ray must be finite and non-zero")
        unit = vector / norm
        radius = np.rad2deg(np.arccos(np.clip(unit[2], -1.0, 1.0)))
        theta = np.mod(np.rad2deg(np.arctan2(unit[1], unit[0])), 360.0)
        return float(radius), float(theta)

    def ray_to_pixel(
        self,
        ray: NDArray[np.float64],
    ) -> tuple[float, float] | None:
        """Project one Grid-frame ray to pixels when inside calibration coverage."""
        radius, theta = self.ray_to_angle(ray)
        _, r_max = self.evaluator.calibrated_angular_range_deg
        if radius > float(r_max):
            return None
        x_value, y_value = self.evaluator.angle_to_pixel(radius, theta)
        return float(np.asarray(x_value)), float(np.asarray(y_value))


def angles_to_grid_rays(
    r_deg: NDArray[np.float64],
    theta_deg: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Convert Grid nominal polar angles to right-handed unit vectors.

    The Grid frame is defined only relative to the fitted printed grid:
    ``r=0`` is the optical axis, ``theta=0`` is ``+x``, ``theta=90`` is ``+y``,
    and ``+z`` is the optical axis. No geographic orientation is assumed.
    """
    radius = np.deg2rad(np.asarray(r_deg, dtype=np.float64))
    theta = np.deg2rad(np.asarray(theta_deg, dtype=np.float64))
    sin_radius = np.sin(radius)
    return np.stack(
        (
            sin_radius * np.cos(theta),
            sin_radius * np.sin(theta),
            np.cos(radius),
        ),
        axis=-1,
    )


def load_grid_calibration(path: Path) -> GridCalibration:
    """Load and validate one portable ``*_calibration.npz`` Grid product.

    Parameters
    ----------
    path
        Portable Grid Calibration product written with pickle disabled.

    Returns
    -------
    gonet_astrometry.models.grid.GridCalibration
        Package-level wrapper exposing the Grid model through the common
        pixel-to-ray protocol.

    Raises
    ------
    RuntimeError
        If the Grid Calibration package is unavailable.
    ValueError
        If the artifact uses an unsupported coordinate convention.
    """
    try:
        from grid_calibration import load_calibration
    except ImportError as exc:
        raise RuntimeError(
            "Grid-calibration support requires the grid_calibration package. "
            "Install or activate the Grid Calibration project before using "
            "--grid-calibration."
        ) from exc

    source = Path(path).expanduser().resolve()
    evaluator = cast(_GridEvaluator, load_calibration(source))
    if evaluator.image_coordinate_convention != _EXPECTED_COORDINATE_CONVENTION:
        raise ValueError(
            "Unsupported Grid image-coordinate convention: "
            f"{evaluator.image_coordinate_convention!r}"
        )
    transform = PortableGridTransform(evaluator)
    return GridCalibration(
        transform=transform,
        image_shape=(evaluator.sensor_height_px, evaluator.sensor_width_px),
        coordinate_convention=evaluator.image_coordinate_convention,
        source=str(source),
    )


def validated_pixel_rays(
    calibration: GridCalibration,
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    *,
    reprojection_tolerance_px: float | None = None,
) -> PixelRayConversion:
    """Convert pixels with the validity information needed by spherical fits."""
    transform = calibration.transform
    if not isinstance(transform, PortableGridTransform):
        raise TypeError("Grid calibration does not use the portable Grid transform")
    return transform.convert(
        x,
        y,
        reprojection_tolerance_px=reprojection_tolerance_px,
    )


def grid_pole_pixel(
    calibration: GridCalibration,
    axis: NDArray[np.float64],
) -> tuple[float, float] | None:
    """Project a fitted Grid-frame rotation axis into sensor coordinates."""
    transform = calibration.transform
    if not isinstance(transform, PortableGridTransform):
        raise TypeError("Grid calibration does not use the portable Grid transform")
    return transform.ray_to_pixel(axis)
