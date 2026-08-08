from datetime import datetime, timezone

import numpy as np
import pytest

from gonet_astrometry.detection import sep_backend
from gonet_astrometry.detection.config import DetectionConfig
from gonet_astrometry.detection.errors import DetectionBackendUnavailableError
from gonet_astrometry.detection.sep_backend import SEPDetector
from gonet_astrometry.models.frame import ImageFrame, ImageMetadata, ObserverLocation


def _frame() -> ImageFrame:
    data = np.arange(256, dtype=np.float64).reshape(16, 16)
    metadata = ImageMetadata(
        datetime(2026, 8, 6, tzinfo=timezone.utc),
        10.0,
        ObserverLocation(0.0, 0.0),
    )
    return ImageFrame(data, metadata)


def test_sep_backend_converts_structured_objects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dtype = [
        ("x", "f8"),
        ("y", "f8"),
        ("flux", "f8"),
        ("peak", "f8"),
        ("a", "f8"),
        ("b", "f8"),
        ("flag", "i4"),
        ("npix", "i4"),
        ("theta", "f8"),
    ]
    objects = np.array(
        [
            (4.5, 7.5, 20.0, 8.0, 2.0, 1.0, 0, 11, 0.5),
            (9.0, 2.0, 5.0, 4.0, 1.0, 1.0, 2, 5, 0.0),
        ],
        dtype=dtype,
    )
    captured: dict[str, object] = {}

    class FakeSEP:
        @staticmethod
        def extract(data: np.ndarray, threshold: float, **kwargs: object) -> np.ndarray:
            captured["contiguous"] = data.flags.c_contiguous
            captured["threshold"] = threshold
            captured.update(kwargs)
            return objects

    monkeypatch.setattr(sep_backend, "_load_sep", lambda: FakeSEP)
    detector = SEPDetector(DetectionConfig(deblend=False, min_pixels=7))

    catalog = detector.detect("frame", _frame())

    assert len(catalog) == 2
    assert catalog.detections[0].signal_to_noise == 8.0
    first = catalog.detections[0]
    second = catalog.detections[1]
    assert first.diagnostics.area_pixels == 11
    assert first.diagnostics.ellipticity == pytest.approx(0.5)
    assert first.diagnostics.orientation_deg == pytest.approx(28.6479, rel=1e-4)
    assert second.flags == ("sep:2", "backend-flagged")
    assert second.diagnostics.backend_flags == 2
    assert captured["deblend_cont"] == 1.0
    assert captured["minarea"] == 7
    assert captured["contiguous"] is True


def test_missing_sep_raises_focused_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_import() -> object:
        raise DetectionBackendUnavailableError("missing sep")

    monkeypatch.setattr(sep_backend, "_load_sep", fail_import)

    with pytest.raises(DetectionBackendUnavailableError, match="missing sep"):
        SEPDetector().detect("frame", _frame())
