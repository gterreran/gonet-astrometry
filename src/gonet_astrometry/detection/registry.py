"""Registry of source-detection backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from gonet_astrometry.detection.base import SourceDetector
from gonet_astrometry.detection.config import DetectionConfig
from gonet_astrometry.detection.photutils_backends import (
    DAOStarFinderDetector,
    PhotutilsSegmentationDetector,
)
from gonet_astrometry.detection.scipy_peaks import ScipyPeakDetector
from gonet_astrometry.detection.sep_backend import SEPDetector

DetectorIdentifier = Literal[
    "scipy-local-max",
    "photutils-segmentation",
    "dao-star-finder",
    "sep",
]
"""Stable identifier for a registered source-detection backend."""


@dataclass(frozen=True, slots=True)
class DetectorSpec:
    """User-facing metadata for one source-detection backend."""

    identifier: DetectorIdentifier
    label: str
    description: str


DETECTOR_SPECS: tuple[DetectorSpec, ...] = (
    DetectorSpec(
        "sep",
        "SEP extraction",
        "Source Extractor algorithms exposed through the SEP Python library.",
    ),
    DetectorSpec(
        "dao-star-finder",
        "DAOStarFinder",
        "Gaussian-kernel point-source detector from Photutils.",
    ),
    DetectorSpec(
        "photutils-segmentation",
        "Photutils segmentation",
        "Connected-source detection with optional watershed deblending.",
    ),
    DetectorSpec(
        "scipy-local-max",
        "SciPy local maxima",
        "Simple smoothed local-maximum baseline available without optional tools.",
    ),
)
"""Ordered source-detection backends shown by the portal."""


def detector_options() -> list[dict[str, str]]:
    """Return Dash-compatible detector-selector options."""
    return [
        {
            "label": spec.label,
            "value": spec.identifier,
            "title": spec.description,
        }
        for spec in DETECTOR_SPECS
    ]


def create_detector(
    identifier: str,
    config: DetectionConfig | None = None,
) -> SourceDetector:
    """Construct a registered detector backend.

    Parameters
    ----------
    identifier
        Stable detector identifier.
    config
        Shared detector configuration. Defaults are used when omitted.

    Returns
    -------
    SourceDetector
        Newly constructed backend.

    Raises
    ------
    ValueError
        If ``identifier`` is not registered.
    """
    settings = config or DetectionConfig()
    if identifier == "scipy-local-max":
        return ScipyPeakDetector(settings)
    if identifier == "photutils-segmentation":
        return PhotutilsSegmentationDetector(settings)
    if identifier == "dao-star-finder":
        return DAOStarFinderDetector(settings)
    if identifier == "sep":
        return SEPDetector(settings)
    raise ValueError(f"Unknown source detector: {identifier!r}")
