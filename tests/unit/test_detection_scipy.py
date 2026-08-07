from datetime import datetime, timezone

import numpy as np

from gonet_astrometry.detection.config import DetectionConfig
from gonet_astrometry.detection.scipy_peaks import ScipyPeakDetector
from gonet_astrometry.models.frame import ImageFrame, ImageMetadata, ObserverLocation


def _synthetic_frame() -> ImageFrame:
    rng = np.random.default_rng(7)
    yy, xx = np.indices((64, 64), dtype=np.float64)
    data = rng.normal(100.0, 1.0, size=(64, 64))
    data += 30.0 * np.exp(-((xx - 31.2) ** 2 + (yy - 27.7) ** 2) / (2 * 1.4**2))
    metadata = ImageMetadata(
        datetime(2026, 8, 6, tzinfo=timezone.utc),
        10.0,
        ObserverLocation(0.0, 0.0),
    )
    return ImageFrame(data, metadata)


def test_scipy_detector_finds_synthetic_source() -> None:
    detector = ScipyPeakDetector(
        DetectionConfig(
            threshold_sigma=4.0,
            fwhm_px=3.0,
            min_separation_px=4.0,
            max_sources=5,
        )
    )

    catalog = detector.detect("synthetic", _synthetic_frame())

    assert catalog.detector_name == "scipy-local-max"
    assert len(catalog) >= 1
    nearest = min(
        catalog.detections,
        key=lambda detection: (detection.x - 31.2) ** 2 + (detection.y - 27.7) ** 2,
    )
    assert abs(nearest.x - 31.2) < 1.5
    assert abs(nearest.y - 27.7) < 1.5
    assert nearest.signal_to_noise > 4.0
    assert nearest.x_uncertainty > 0


def test_scipy_detector_returns_empty_catalog_at_high_threshold() -> None:
    detector = ScipyPeakDetector(DetectionConfig(threshold_sigma=1000.0))

    catalog = detector.detect("synthetic", _synthetic_frame())

    assert len(catalog) == 0
