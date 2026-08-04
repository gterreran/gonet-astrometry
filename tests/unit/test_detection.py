import pytest

from gonet_astrometry.models.detection import Detection, DetectionCatalog


def make_detection() -> Detection:
    return Detection(
        identifier=4,
        x=12.5,
        y=18.5,
        flux=100.0,
        signal_to_noise=8.0,
        x_uncertainty=0.2,
        y_uncertainty=0.3,
        flags=("near-mask",),
    )


def test_detection_catalog_length() -> None:
    catalog = DetectionCatalog("frame-1", (make_detection(),), "test-detector")
    assert len(catalog) == 1


def test_detection_rejects_invalid_uncertainty() -> None:
    with pytest.raises(ValueError, match="uncertainties"):
        Detection(1, 1.0, 2.0, 3.0, 4.0, 0.0, 0.1)
