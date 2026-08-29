import pytest

from gonet_astrometry.detection.config import DetectionConfig
from gonet_astrometry.detection.registry import (
    DETECTOR_SPECS,
    create_detector,
    detector_options,
)
from gonet_astrometry.detection.scipy_peaks import ScipyPeakDetector


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"threshold_sigma": 0.0}, "threshold_sigma"),
        ({"fwhm_px": 0.0}, "fwhm_px"),
        ({"min_separation_px": -1.0}, "min_separation_px"),
        ({"min_pixels": 0}, "min_pixels"),
        ({"max_sources": 0}, "max_sources"),
        ({"background_box_size_px": 4}, "background_box_size_px"),
        (
            {"background_min_valid_fraction": 0.0},
            "background_min_valid_fraction",
        ),
        ({"background_sigma_clip": 0.0}, "background_sigma_clip"),
        ({"background_clip_iterations": 0}, "background_clip_iterations"),
        ({"footprint_threshold_fraction": 1.0}, "footprint_threshold_fraction"),
        ({"footprint_smoothing_px": -1.0}, "footprint_smoothing_px"),
        ({"footprint_erosion_px": -1.0}, "footprint_erosion_px"),
        ({"bright_mask_sigma": 0.0}, "bright_mask_sigma"),
        ({"bright_mask_min_pixels": 0}, "bright_mask_min_pixels"),
        ({"bright_mask_dilation_px": -1.0}, "bright_mask_dilation_px"),
    ],
)
def test_detection_config_rejects_invalid_values(
    kwargs: dict[str, float | int],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        DetectionConfig(**kwargs)


def test_detection_config_default_threshold_is_3p5_sigma() -> None:
    assert DetectionConfig().threshold_sigma == 3.5


def test_detector_registry_exposes_and_constructs_backends() -> None:
    options = detector_options()

    assert len(options) == len(DETECTOR_SPECS) == 4
    assert options[0]["value"] == "sep"
    assert isinstance(create_detector("scipy-local-max"), ScipyPeakDetector)

    with pytest.raises(ValueError, match="Unknown source detector"):
        create_detector("unknown")
