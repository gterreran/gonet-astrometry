import numpy as np
import pytest

from gonet_astrometry.detection.config import DetectionConfig
from gonet_astrometry.detection.multichannel import (
    CHANNEL_OFFSETS,
    ChannelDetection,
    MultiChannelSEPConfig,
    compact_detection_config,
    compact_to_full,
    full_to_compact,
    fuse_channel_detections,
    weighted_coordinate,
)


@pytest.mark.parametrize("channel", tuple(CHANNEL_OFFSETS))
def test_compact_full_coordinate_round_trip(channel: str) -> None:
    x_compact = np.asarray([0.0, 12.25, 100.5])
    y_compact = np.asarray([1.0, 30.75, 85.5])

    x_full, y_full = compact_to_full(
        x_compact,
        y_compact,
        channel,  # type: ignore[arg-type]
    )
    x_back, y_back = full_to_compact(
        x_full,
        y_full,
        channel,  # type: ignore[arg-type]
    )

    assert np.allclose(x_back, x_compact)
    assert np.allclose(y_back, y_compact)


def test_compact_detection_config_scales_spatial_quantities() -> None:
    base = DetectionConfig(
        threshold_sigma=3.0,
        fwhm_px=3.0,
        min_separation_px=4.0,
        min_pixels=5,
        use_provisional_field_mask=True,
        bright_mask_sigma=40.0,
    )

    compact = compact_detection_config(base)

    assert compact.threshold_sigma == 3.0
    assert compact.fwhm_px == 1.5
    assert compact.min_separation_px == 2.0
    assert compact.min_pixels == 2
    assert compact.use_provisional_field_mask is False
    assert compact.bright_mask_sigma is None


def _item(
    channel: str,
    x: float,
    y: float,
    *,
    uncertainty: float = 1.0,
    snr: float = 10.0,
) -> ChannelDetection:
    return ChannelDetection(
        channel=channel,  # type: ignore[arg-type]
        x=x,
        y=y,
        x_uncertainty=uncertainty,
        y_uncertainty=uncertainty,
        signal_to_noise=snr,
        flux=100.0,
    )


def test_soft_fusion_retains_single_channel_candidates() -> None:
    by_channel = {
        "blue": (
            _item("blue", 100.0, 200.0, snr=20.0),
            _item("blue", 300.0, 400.0, snr=8.0),
        ),
        "green1": (_item("green1", 101.0, 199.5, snr=18.0),),
        "green2": (_item("green2", 99.5, 201.0, snr=16.0),),
        "red": (_item("red", 100.5, 200.5, snr=14.0),),
    }

    fused = fuse_channel_detections(
        by_channel,  # type: ignore[arg-type]
        match_radius_px=6.0,
        min_channels=1,
    )

    assert len(fused) == 2
    assert fused[0].channel_support == 4
    assert fused[0].channels == ("blue", "green1", "green2", "red")
    assert fused[1].channel_support == 1
    assert fused[1].channels == ("blue",)


def test_soft_fusion_can_require_channel_multiplicity_for_diagnostics() -> None:
    by_channel = {
        "blue": (
            _item("blue", 100.0, 200.0),
            _item("blue", 300.0, 400.0),
        ),
        "green1": (_item("green1", 101.0, 200.0),),
        "green2": (_item("green2", 100.0, 201.0),),
        "red": (),
    }

    fused = fuse_channel_detections(
        by_channel,  # type: ignore[arg-type]
        match_radius_px=6.0,
        min_channels=3,
    )

    assert len(fused) == 1
    assert fused[0].channel_support == 3


def test_weighted_coordinate_uses_inverse_variance() -> None:
    value, uncertainty = weighted_coordinate(
        np.asarray([10.0, 14.0]),
        np.asarray([1.0, 2.0]),
    )

    assert value == pytest.approx(10.8)
    assert uncertainty == pytest.approx(np.sqrt(0.8))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"field_edge_keep_margin_px": -1.0}, "field_edge_keep_margin_px"),
        ({"grid_search_radius_deg": 0.0}, "grid_search_radius_deg"),
        ({"grid_acceptance_radius_deg": 0.0}, "grid_acceptance_radius_deg"),
        (
            {"grid_search_radius_deg": 70.0, "grid_acceptance_radius_deg": 75.0},
            "grid_acceptance_radius_deg",
        ),
        ({"grid_contour_samples": 100}, "grid_contour_samples"),
        ({"channel_match_radius_px": 0.0}, "channel_match_radius_px"),
        ({"background_box_size_compact_px": 2}, "background_box_size"),
        ({"minimum_channel_support": 0}, "minimum_channel_support"),
    ],
)
def test_multichannel_config_rejects_invalid_values(
    kwargs: dict[str, float | int | None],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        MultiChannelSEPConfig(**kwargs)


class _IdentityEvaluator:
    sensor_width_px = 120
    sensor_height_px = 120
    image_coordinate_convention = (
        "x=column,y=row;origin=upper-left;+x=right;+y=down;"
        "pixel-centers-at-integer-coordinates"
    )
    calibrated_angular_range_deg = (0.0, 179.0)

    def angle_to_pixel(self, r_deg, theta_deg):
        return np.asarray(r_deg, dtype=float), np.asarray(theta_deg, dtype=float)

    def pixel_to_angle(self, x, y, **kwargs):
        del kwargs
        return np.asarray(x, dtype=float), np.asarray(y, dtype=float)


def _synthetic_frame(data: np.ndarray):
    from datetime import datetime, timezone

    from gonet_astrometry.models.frame import (
        ImageFrame,
        ImageMetadata,
        ObserverLocation,
    )

    return ImageFrame(
        data=data,
        metadata=ImageMetadata(
            exposure_start=datetime(2026, 8, 28, tzinfo=timezone.utc),
            exposure_duration_s=10.0,
            location=ObserverLocation(41.88, -87.63, 180.0),
        ),
    )


def _synthetic_calibration():
    from gonet_astrometry.adapters.grid_calibration import PortableGridTransform
    from gonet_astrometry.models.grid import GridCalibration

    evaluator = _IdentityEvaluator()
    return GridCalibration(
        transform=PortableGridTransform(evaluator),
        image_shape=(120, 120),
        coordinate_convention=evaluator.image_coordinate_convention,
        source="synthetic-grid.npz",
    )


def test_automatic_field_edge_drives_search_and_acceptance_masks() -> None:
    from gonet_astrometry.detection.multichannel import IndependentChannelSEPDetector

    yy, xx = np.indices((120, 120), dtype=float)
    illuminated = np.hypot(xx - 60.0, yy - 60.0) <= 48.0
    data = np.where(illuminated, 100.0, 0.0)
    detector = IndependentChannelSEPDetector(
        _synthetic_calibration(),
        DetectionConfig(
            threshold_sigma=3.0,
            footprint_threshold_fraction=0.2,
            footprint_smoothing_px=0.0,
            footprint_erosion_px=0.0,
        ),
        MultiChannelSEPConfig(field_edge_keep_margin_px=8.0),
    )

    search, acceptance = detector._frame_field_masks(_synthetic_frame(data))

    assert search[60, 60]
    assert acceptance[60, 60]

    assert not search[0, 0]
    assert not acceptance[0, 0]

    # Near the automatically inferred edge: still searchable,
    # but excluded by the additional 8 px keep margin.
    assert search[60, 101]
    assert not acceptance[60, 101]

    assert np.count_nonzero(acceptance) < np.count_nonzero(search)


def test_static_field_mask_is_combined_with_automatic_field_edge() -> None:
    from gonet_astrometry.detection.field_mask import FieldMask
    from gonet_astrometry.detection.multichannel import IndependentChannelSEPDetector

    yy, xx = np.indices((120, 120), dtype=float)
    illuminated = np.hypot(xx - 60.0, yy - 60.0) <= 48.0
    data = np.where(illuminated, 100.0, 0.0)
    excluded = np.zeros((120, 120), dtype=bool)
    excluded[55:66, 55:66] = True
    field_mask = FieldMask(
        excluded=excluded,
        coordinate_convention=_IdentityEvaluator.image_coordinate_convention,
    )
    detector = IndependentChannelSEPDetector(
        _synthetic_calibration(),
        DetectionConfig(
            threshold_sigma=3.0,
            footprint_threshold_fraction=0.2,
            footprint_smoothing_px=0.0,
            footprint_erosion_px=0.0,
        ),
        MultiChannelSEPConfig(field_edge_keep_margin_px=0.0),
        field_mask=field_mask,
    )

    search, acceptance = detector._frame_field_masks(_synthetic_frame(data))

    assert not search[60, 60]
    assert not acceptance[60, 60]
    assert search[60, 80]
    assert acceptance[60, 80]


class _RadialEvaluator:
    sensor_width_px = 200
    sensor_height_px = 200
    image_coordinate_convention = (
        "x=column,y=row;origin=upper-left;+x=right;+y=down;"
        "pixel-centers-at-integer-coordinates"
    )
    calibrated_angular_range_deg = (0.0, 95.0)

    def angle_to_pixel(self, r_deg, theta_deg):
        radius = np.asarray(r_deg, dtype=float)
        theta = np.deg2rad(np.asarray(theta_deg, dtype=float))
        return 100.0 + radius * np.cos(theta), 100.0 + radius * np.sin(theta)

    def pixel_to_angle(self, x, y, **kwargs):
        del kwargs
        dx = np.asarray(x, dtype=float) - 100.0
        dy = np.asarray(y, dtype=float) - 100.0
        return np.hypot(dx, dy), np.mod(np.rad2deg(np.arctan2(dy, dx)), 360.0)


def _radial_calibration():
    from gonet_astrometry.adapters.grid_calibration import PortableGridTransform
    from gonet_astrometry.models.grid import GridCalibration

    evaluator = _RadialEvaluator()
    return GridCalibration(
        transform=PortableGridTransform(evaluator),
        image_shape=(200, 200),
        coordinate_convention=evaluator.image_coordinate_convention,
        source="synthetic-radial-grid.npz",
    )


def test_grid_radius_caps_reproduce_search_and_acceptance_without_auto_edge() -> None:
    from gonet_astrometry.detection.multichannel import IndependentChannelSEPDetector

    data = np.ones((200, 200), dtype=float)
    detector = IndependentChannelSEPDetector(
        _radial_calibration(),
        DetectionConfig(
            threshold_sigma=3.0,
            use_provisional_field_mask=False,
        ),
        MultiChannelSEPConfig(
            field_edge_keep_margin_px=50.0,
            grid_search_radius_deg=75.0,
            grid_acceptance_radius_deg=70.0,
        ),
    )

    search, acceptance = detector._frame_field_masks(_synthetic_frame(data))

    assert search[100, 100]
    assert acceptance[100, 100]
    assert search[100, 172]  # r = 72: searchable guard band
    assert not acceptance[100, 172]
    assert not search[100, 178]  # r = 78: outside the 75-degree cap
    assert not acceptance[100, 178]


def test_grid_radius_caps_are_optional_when_auto_edge_is_disabled() -> None:
    from gonet_astrometry.detection.multichannel import IndependentChannelSEPDetector

    data = np.ones((200, 200), dtype=float)
    detector = IndependentChannelSEPDetector(
        _radial_calibration(),
        DetectionConfig(
            threshold_sigma=3.0,
            use_provisional_field_mask=False,
        ),
        MultiChannelSEPConfig(field_edge_keep_margin_px=50.0),
    )

    search, acceptance = detector._frame_field_masks(_synthetic_frame(data))

    assert np.all(search)
    assert np.all(acceptance)
