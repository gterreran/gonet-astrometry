import numpy as np
import pytest

from gonet_astrometry.detection.config import DetectionConfig
from gonet_astrometry.detection.diagnostics import (
    enrich_detection_catalog,
    measure_local_source,
)
from gonet_astrometry.detection.preprocessing import PreparedDetectionImage
from gonet_astrometry.models.detection import (
    Detection,
    DetectionCatalog,
    DetectionDiagnostics,
)


def _prepared(data: np.ndarray) -> PreparedDetectionImage:
    mask = np.zeros(data.shape, dtype=bool)
    return PreparedDetectionImage(
        data=np.asarray(data, dtype=np.float64),
        mask=mask,
        field_mask=~mask,
        dynamic_mask=mask.copy(),
        backgrounds=(0.0, 0.0, 0.0, 0.0),
        noises=(1.0, 1.0, 1.0, 1.0),
    )


def test_measure_local_source_reports_shape_and_orientation() -> None:
    yy, xx = np.indices((31, 31), dtype=np.float64)
    data = 20.0 * np.exp(-0.5 * (((xx - 15.0) / 3.0) ** 2 + (yy - 15.0) ** 2))

    diagnostics = measure_local_source(
        data,
        np.zeros(data.shape, dtype=bool),
        15.0,
        15.0,
        radius=8,
        threshold_sigma=3.0,
    )

    assert diagnostics.peak_value == pytest.approx(20.0)
    assert diagnostics.area_pixels is not None
    assert diagnostics.semimajor_sigma_px is not None
    assert diagnostics.semiminor_sigma_px is not None
    assert diagnostics.semimajor_sigma_px > diagnostics.semiminor_sigma_px
    assert diagnostics.elongation is not None
    assert diagnostics.elongation > 2.0
    assert diagnostics.ellipticity is not None
    assert diagnostics.orientation_deg is not None


def test_enrich_catalog_merges_backend_and_common_diagnostics() -> None:
    yy, xx = np.indices((31, 31), dtype=np.float64)
    data = 20.0 * np.exp(-0.5 * (((xx - 15.0) / 3.0) ** 2 + (yy - 15.0) ** 2))
    detection = Detection(
        9,
        15.0,
        15.0,
        100.0,
        20.0,
        0.1,
        0.1,
        diagnostics=DetectionDiagnostics(backend_flags=4),
    )
    catalog = DetectionCatalog("frame", (detection,), "fake")

    enriched = enrich_detection_catalog(
        catalog,
        _prepared(data),
        DetectionConfig(threshold_sigma=3.0, bright_mask_sigma=None),
    )
    result = enriched.detections[0]

    assert result.diagnostics.backend_flags == 4
    assert result.diagnostics.peak_value == pytest.approx(20.0)
    assert result.diagnostics.area_pixels is not None
    assert "backend-flagged" in result.flags
    assert "elongated" in result.flags
    assert "extended" in result.flags
    assert result.diagnostic_class == "backend-flagged"


def test_enrich_catalog_marks_mask_adjacency_without_rejecting_source() -> None:
    data = np.zeros((20, 20), dtype=np.float64)
    data[10, 3] = 10.0
    field = np.ones(data.shape, dtype=bool)
    field[:, :2] = False
    dynamic = np.zeros(data.shape, dtype=bool)
    dynamic[8:13, 8:13] = True
    prepared = PreparedDetectionImage(
        data=data,
        mask=~field | dynamic,
        field_mask=field,
        dynamic_mask=dynamic,
        backgrounds=(0.0, 0.0, 0.0, 0.0),
        noises=(1.0, 1.0, 1.0, 1.0),
    )
    detection = Detection(1, 3.0, 10.0, 10.0, 10.0, 0.1, 0.1)

    enriched = enrich_detection_catalog(
        DetectionCatalog("frame", (detection,), "fake"),
        prepared,
        DetectionConfig(fwhm_px=3.0, bright_mask_sigma=None),
    )

    result = enriched.detections[0]
    assert len(enriched) == 1
    assert "near-field-edge" in result.flags
    assert result.diagnostic_class == "mask-adjacent"
