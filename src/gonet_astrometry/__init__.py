"""Astrometric calibration tools for wide-field GONet images.

The package separates numerical astrometry from GONet Wizard compatibility and
from any future graphical interface. Its public API begins with data models for
images, detections, tracks, Grid calibrations, and astrometric solutions.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("gonet-astrometry")
except PackageNotFoundError:  # pragma: no cover - source tree without install
    __version__ = "0.0.0"

from gonet_astrometry.models.detection import Detection, DetectionCatalog
from gonet_astrometry.models.frame import (
    ImageFrame,
    ImageMetadata,
    ObserverLocation,
)
from gonet_astrometry.models.grid import GridCalibration
from gonet_astrometry.models.solution import AstrometricSolution
from gonet_astrometry.models.track import StarTrack

__all__ = [
    "AstrometricSolution",
    "Detection",
    "DetectionCatalog",
    "GridCalibration",
    "ImageFrame",
    "ImageMetadata",
    "ObserverLocation",
    "StarTrack",
    "__version__",
]
