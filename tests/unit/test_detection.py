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


def test_detection_diagnostics_validate_values_and_classify_sources() -> None:
    from gonet_astrometry.models.detection import DetectionDiagnostics

    diagnostics = DetectionDiagnostics(
        peak_value=12.0,
        area_pixels=9,
        semimajor_sigma_px=2.0,
        semiminor_sigma_px=1.0,
        ellipticity=0.5,
    )
    detection = Detection(
        1,
        2.0,
        3.0,
        10.0,
        8.0,
        0.1,
        0.1,
        diagnostics=diagnostics,
    )

    assert diagnostics.has_shape is True
    assert detection.diagnostic_class == "compact"

    with pytest.raises(ValueError, match="ellipticity"):
        DetectionDiagnostics(ellipticity=1.1)


def test_detection_catalog_counts_mutually_exclusive_diagnostics() -> None:
    from dataclasses import replace

    base = make_detection()
    catalog = DetectionCatalog(
        "frame",
        (
            replace(base, identifier=1, flags=(), diagnostics=base.diagnostics),
            replace(base, identifier=2, flags=("elongated",)),
            replace(base, identifier=3, flags=("extended", "elongated")),
            replace(base, identifier=4, flags=("near-field-edge",)),
            replace(base, identifier=5, flags=("backend-flagged",)),
        ),
        "fake",
    )

    assert catalog.diagnostic_counts() == {
        "compact": 0,
        "elongated": 1,
        "extended": 1,
        "mask-adjacent": 1,
        "backend-flagged": 1,
        "unclassified": 1,
    }
