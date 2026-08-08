from datetime import datetime, timezone

import numpy as np
import pytest

from gonet_astrometry.detection import photutils_backends
from gonet_astrometry.detection.config import DetectionConfig
from gonet_astrometry.detection.errors import DetectionBackendUnavailableError
from gonet_astrometry.detection.photutils_backends import (
    DAOStarFinderDetector,
    PhotutilsSegmentationDetector,
)
from gonet_astrometry.models.frame import ImageFrame, ImageMetadata, ObserverLocation


def _frame() -> ImageFrame:
    data = np.arange(256, dtype=np.float64).reshape(16, 16)
    metadata = ImageMetadata(
        datetime(2026, 8, 6, tzinfo=timezone.utc),
        10.0,
        ObserverLocation(0.0, 0.0),
    )
    return ImageFrame(data, metadata)


def test_daostarfinder_backend_converts_table_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeFinder:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

        def __call__(
            self,
            _data: np.ndarray,
            mask: np.ndarray,
        ) -> list[dict[str, float]]:
            assert mask.shape == (16, 16)
            return [
                {
                    "xcentroid": 4.5,
                    "ycentroid": 7.5,
                    "flux": 30.0,
                    "peak": 8.0,
                    "roundness1": 0.2,
                    "roundness2": -0.1,
                    "sharpness": 0.7,
                    "npix": 9,
                },
                {"xcentroid": np.nan, "ycentroid": 1.0},
            ]

    monkeypatch.setattr(photutils_backends, "_load_daofinder", lambda: FakeFinder)
    detector = DAOStarFinderDetector(DetectionConfig(max_sources=10))

    catalog = detector.detect("frame", _frame())

    assert len(catalog) == 1
    assert catalog.detections[0].x == 4.5
    result = catalog.detections[0]
    assert result.elongation == pytest.approx(1.2)
    assert result.diagnostics.peak_value == 8.0
    assert result.diagnostics.area_pixels == 9
    assert result.diagnostics.sharpness == 0.7
    assert result.diagnostics.roundness1 == 0.2
    assert result.diagnostics.roundness2 == -0.1
    assert captured["threshold"] == 5.0


def test_daostarfinder_backend_handles_no_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeFinder:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def __call__(self, _data: np.ndarray, mask: np.ndarray) -> None:
            return None

    monkeypatch.setattr(photutils_backends, "_load_daofinder", lambda: FakeFinder)

    assert len(DAOStarFinderDetector().detect("frame", _frame())) == 0


def test_segmentation_backend_converts_source_catalog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeFinder:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def __call__(
            self,
            _data: np.ndarray,
            _threshold: float,
            mask: np.ndarray,
        ) -> object:
            assert mask.shape == (16, 16)
            return object()

    class FakeCatalog:
        labels = np.array([9])
        x_centroid = np.array([6.25])
        y_centroid = np.array([8.75])
        segment_flux = np.array([42.0])
        max_value = np.array([7.0])
        semimajor_axis = np.array([2.0])
        semiminor_axis = np.array([1.0])
        area = np.array([13.0])
        orientation = np.array([27.5])

        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

    monkeypatch.setattr(
        photutils_backends,
        "_load_segmentation_tools",
        lambda: (FakeFinder, FakeCatalog),
    )

    catalog = PhotutilsSegmentationDetector().detect("frame", _frame())

    assert len(catalog) == 1
    assert catalog.detections[0].x == 6.25
    result = catalog.detections[0]
    assert result.elongation == 2.0
    assert result.diagnostics.area_pixels == 13
    assert result.diagnostics.semimajor_sigma_px == 2.0
    assert result.diagnostics.semiminor_sigma_px == 1.0
    assert result.diagnostics.ellipticity == 0.5
    assert result.diagnostics.orientation_deg == 27.5


def test_segmentation_backend_handles_no_segments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeFinder:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def __call__(self, *_args: object, **_kwargs: object) -> None:
            return None

    monkeypatch.setattr(
        photutils_backends,
        "_load_segmentation_tools",
        lambda: (FakeFinder, object),
    )

    assert len(PhotutilsSegmentationDetector().detect("frame", _frame())) == 0


def test_missing_photutils_raises_focused_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_import() -> object:
        raise DetectionBackendUnavailableError("missing")

    monkeypatch.setattr(photutils_backends, "_load_daofinder", fail_import)

    with pytest.raises(DetectionBackendUnavailableError, match="missing"):
        DAOStarFinderDetector().detect("frame", _frame())


def test_supported_keyword_handles_modern_and_legacy_signatures() -> None:
    class ModernFinder:
        def __init__(self, *, n_brightest: int) -> None:
            pass

    class LegacyFinder:
        def __init__(self, *, brightest: int) -> None:
            pass

    assert (
        photutils_backends._supported_keyword(ModernFinder, "n_brightest", "brightest")
        == "n_brightest"
    )
    assert (
        photutils_backends._supported_keyword(LegacyFinder, "n_brightest", "brightest")
        == "brightest"
    )


def test_supported_keyword_handles_kwargs_and_uninspectable_callables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FlexibleFinder:
        def __init__(self, **_kwargs: object) -> None:
            pass

    assert (
        photutils_backends._supported_keyword(FlexibleFinder, "n_pixels", "npixels")
        == "n_pixels"
    )

    def fail_signature(_factory: object) -> object:
        raise ValueError("no signature")

    monkeypatch.setattr(photutils_backends, "signature", fail_signature)
    assert (
        photutils_backends._supported_keyword(object(), "n_pixels", "npixels")
        == "n_pixels"
    )


def test_supported_keyword_rejects_unknown_api() -> None:
    class UnsupportedFinder:
        def __init__(self, *, unrelated: int) -> None:
            pass

    with pytest.raises(
        DetectionBackendUnavailableError,
        match="expected keyword 'n_pixels' or 'npixels'",
    ):
        photutils_backends._supported_keyword(UnsupportedFinder, "n_pixels", "npixels")
