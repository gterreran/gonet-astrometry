"""Source-detection data models."""

from __future__ import annotations

from dataclasses import dataclass


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
        Optional source-shape elongation diagnostic.
    flags
        Tuple of machine-readable quality flags. Detections are retained with
        flags whenever possible rather than being aggressively discarded.

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

    def __post_init__(self) -> None:
        if self.x_uncertainty <= 0 or self.y_uncertainty <= 0:
            raise ValueError("Detection coordinate uncertainties must be positive")


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
