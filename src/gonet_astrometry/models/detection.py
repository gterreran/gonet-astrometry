"""Source-detection data models."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

DetectionClass = Literal[
    "compact",
    "elongated",
    "extended",
    "mask-adjacent",
    "backend-flagged",
    "unclassified",
]
"""Non-destructive diagnostic class assigned to one source candidate."""


@dataclass(frozen=True, slots=True)
class DetectionDiagnostics:
    """Optional shape and backend measurements for one detection.

    Parameters
    ----------
    peak_value
        Peak value in the shared locally normalized detection image.
    area_pixels
        Number of pixels associated with the source footprint.
    semimajor_sigma_px, semiminor_sigma_px
        Intensity-weighted source-shape standard deviations in native sensor
        pixels. Backends may provide their own measurements; otherwise the
        common diagnostics pass estimates them from the prepared image.
    orientation_deg
        Position angle of the semimajor axis in degrees, measured
        counterclockwise from the positive sensor-x direction.
    ellipticity
        Shape ellipticity defined as ``1 - b / a`` for semimajor axis ``a`` and
        semiminor axis ``b``.
    backend_flags
        Integer quality bitmask reported by the detector backend, when
        available.
    sharpness
        DAOStarFinder sharpness measurement, when available.
    roundness1, roundness2
        DAOStarFinder roundness measurements, when available.

    Notes
    -----
    Missing measurements remain ``None``. Diagnostics are descriptive only;
    detections are not removed because of these values.
    """

    peak_value: float | None = None
    area_pixels: int | None = None
    semimajor_sigma_px: float | None = None
    semiminor_sigma_px: float | None = None
    orientation_deg: float | None = None
    ellipticity: float | None = None
    backend_flags: int | None = None
    sharpness: float | None = None
    roundness1: float | None = None
    roundness2: float | None = None

    def __post_init__(self) -> None:
        optional_floats = (
            self.peak_value,
            self.semimajor_sigma_px,
            self.semiminor_sigma_px,
            self.orientation_deg,
            self.ellipticity,
            self.sharpness,
            self.roundness1,
            self.roundness2,
        )
        if any(
            value is not None and not math.isfinite(value) for value in optional_floats
        ):
            raise ValueError("Detection diagnostics must be finite when provided")
        if self.area_pixels is not None and self.area_pixels <= 0:
            raise ValueError("Detection area_pixels must be positive when provided")
        for axis in (self.semimajor_sigma_px, self.semiminor_sigma_px):
            if axis is not None and axis <= 0:
                raise ValueError("Detection shape axes must be positive when provided")
        if self.ellipticity is not None and not 0.0 <= self.ellipticity <= 1.0:
            raise ValueError("Detection ellipticity must lie in the interval [0, 1]")
        if self.backend_flags is not None and self.backend_flags < 0:
            raise ValueError("Detection backend_flags cannot be negative")

    @property
    def has_shape(self) -> bool:
        """Return whether both source-shape axes are available."""
        return (
            self.semimajor_sigma_px is not None and self.semiminor_sigma_px is not None
        )


@dataclass(frozen=True, slots=True)
class Detection:
    """One source candidate measured in a single image.

    Parameters
    ----------
    identifier
        Identifier unique within the containing detection catalog.
    x
        Zero-indexed horizontal sensor coordinate in pixels.
    y
        Zero-indexed vertical sensor coordinate in pixels.
    flux
        Background-subtracted detector response in implementation-defined
        units.
    signal_to_noise
        Estimated source signal-to-noise ratio.
    x_uncertainty
        One-sigma uncertainty on ``x`` in pixels.
    y_uncertainty
        One-sigma uncertainty on ``y`` in pixels.
    elongation
        Optional semimajor-to-semiminor axis ratio.
    flags
        Tuple of machine-readable quality flags. Detections are retained with
        flags whenever possible rather than being aggressively discarded.
    diagnostics
        Optional common and backend-specific source measurements.

    Raises
    ------
    ValueError
        If a coordinate uncertainty is not strictly positive.
    """

    identifier: int
    x: float
    y: float
    flux: float
    signal_to_noise: float
    x_uncertainty: float
    y_uncertainty: float
    elongation: float | None = None
    flags: tuple[str, ...] = ()
    diagnostics: DetectionDiagnostics = field(default_factory=DetectionDiagnostics)

    def __post_init__(self) -> None:
        if self.x_uncertainty <= 0 or self.y_uncertainty <= 0:
            raise ValueError("Detection coordinate uncertainties must be positive")

    @property
    def diagnostic_class(self) -> DetectionClass:
        """Return the highest-priority non-destructive diagnostic class."""
        if "backend-flagged" in self.flags:
            return "backend-flagged"
        if "near-bright-mask" in self.flags or "near-field-edge" in self.flags:
            return "mask-adjacent"
        if "extended" in self.flags:
            return "extended"
        if "elongated" in self.flags:
            return "elongated"
        if (
            self.diagnostics.peak_value is not None
            or self.diagnostics.area_pixels is not None
            or self.diagnostics.has_shape
        ):
            return "compact"
        return "unclassified"


@dataclass(frozen=True, slots=True)
class DetectionCatalog:
    """Source candidates measured in one image.

    Parameters
    ----------
    frame_identifier
        Stable identifier for the associated image frame.
    detections
        Immutable tuple of detections.
    detector_name
        Name of the detector backend and, optionally, its version.
    """

    frame_identifier: str
    detections: tuple[Detection, ...]
    detector_name: str

    def __len__(self) -> int:
        """Return the number of detections."""
        return len(self.detections)

    def diagnostic_counts(self) -> dict[DetectionClass, int]:
        """Return mutually exclusive counts by diagnostic class."""
        counts: dict[DetectionClass, int] = {
            "compact": 0,
            "elongated": 0,
            "extended": 0,
            "mask-adjacent": 0,
            "backend-flagged": 0,
            "unclassified": 0,
        }
        for detection in self.detections:
            counts[detection.diagnostic_class] += 1
        return counts
