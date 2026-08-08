"""Pluggable source detection for native GONet images."""

from gonet_astrometry.detection.base import SourceDetector
from gonet_astrometry.detection.config import DetectionConfig
from gonet_astrometry.detection.diagnostics import (
    LocalSourceDiagnostics,
    enrich_detection_catalog,
    measure_local_source,
)
from gonet_astrometry.detection.errors import (
    DetectionBackendUnavailableError,
    DetectionInputError,
    SourceDetectionError,
)
from gonet_astrometry.detection.photutils_backends import (
    DAOStarFinderDetector,
    PhotutilsSegmentationDetector,
)
from gonet_astrometry.detection.preprocessing import (
    PreparedDetectionImage,
    estimate_provisional_field_mask,
    prepare_bayer_detection_image,
)
from gonet_astrometry.detection.registry import (
    DETECTOR_SPECS,
    DetectorIdentifier,
    DetectorSpec,
    create_detector,
    detector_options,
)
from gonet_astrometry.detection.scipy_peaks import ScipyPeakDetector
from gonet_astrometry.detection.sep_backend import SEPDetector
from gonet_astrometry.detection.timing import DetectionTiming

__all__ = [
    "DAOStarFinderDetector",
    "DETECTOR_SPECS",
    "DetectionBackendUnavailableError",
    "DetectionConfig",
    "DetectionInputError",
    "LocalSourceDiagnostics",
    "DetectionTiming",
    "DetectorIdentifier",
    "DetectorSpec",
    "PhotutilsSegmentationDetector",
    "PreparedDetectionImage",
    "SEPDetector",
    "ScipyPeakDetector",
    "SourceDetectionError",
    "SourceDetector",
    "create_detector",
    "detector_options",
    "enrich_detection_catalog",
    "estimate_provisional_field_mask",
    "measure_local_source",
    "prepare_bayer_detection_image",
]
