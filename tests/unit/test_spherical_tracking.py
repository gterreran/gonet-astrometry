from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from gonet_astrometry.adapters.grid_calibration import PortableGridTransform
from gonet_astrometry.models.detection import Detection, DetectionCatalog
from gonet_astrometry.models.frame import ObserverLocation
from gonet_astrometry.models.grid import GridCalibration
from gonet_astrometry.tracking.sequence import DetectionEpoch, DetectionSequence
from gonet_astrometry.tracking.spherical import (
    SphericalTracker,
    SphericalTrackingConfig,
)


class PolarEvaluator:
    sensor_width_px = 360
    sensor_height_px = 180
    image_coordinate_convention = (
        "x=column,y=row;origin=upper-left;+x=right;+y=down;"
        "pixel-centers-at-integer-coordinates"
    )
    calibrated_angular_range_deg = (0.0, 179.0)

    def pixel_to_angle(self, x, y, **kwargs):
        del kwargs
        return np.asarray(x, dtype=float), np.asarray(y, dtype=float)

    def angle_to_pixel(self, r_deg, theta_deg):
        return np.asarray(r_deg, dtype=float), np.asarray(theta_deg, dtype=float)


def _calibration() -> GridCalibration:
    evaluator = PolarEvaluator()
    return GridCalibration(
        transform=PortableGridTransform(evaluator),
        image_shape=(180, 360),
        coordinate_convention=evaluator.image_coordinate_convention,
        source="synthetic-grid.npz",
    )


def _sequence(
    frame_seconds: list[float],
    radii_by_frame: list[tuple[float, ...]],
) -> DetectionSequence:
    start = datetime(2026, 8, 22, tzinfo=timezone.utc)
    location = ObserverLocation(41.88, -87.63, 180.0)
    epochs = []

    for frame_index, (seconds, radii) in enumerate(
        zip(frame_seconds, radii_by_frame, strict=True)
    ):
        frame_id = f"frame-{frame_index}"
        detections = tuple(
            Detection(
                identifier=index,
                x=float(radius),
                y=40.0 + 80.0 * index,
                flux=100.0,
                signal_to_noise=20.0,
                x_uncertainty=0.2,
                y_uncertainty=0.2,
                flags=("channel-support:4",),
            )
            for index, radius in enumerate(radii)
        )
        epochs.append(
            DetectionEpoch(
                frame_identifier=frame_id,
                source_path=Path(frame_id),
                exposure_midpoint=start + timedelta(seconds=seconds),
                location=location,
                image_shape=(180, 360),
                sensor_orientation="native",
                catalog=DetectionCatalog(
                    frame_id,
                    detections,
                    "synthetic-multichannel",
                ),
            )
        )
    return DetectionSequence.from_epochs(epochs)


def test_spherical_tracker_links_sidereal_scale_motion() -> None:
    # 0.20 deg/min = 0.10 deg every 30 seconds. This is inside both the
    # one-point bootstrap gate and the established-track prediction gate.
    sequence = _sequence(
        [0.0, 30.0, 60.0],
        [
            (40.000, 55.000),
            (40.100, 55.100),
            (40.200, 55.200),
        ],
    )

    result = SphericalTracker().track(sequence, _calibration())

    assert len(result.tracks) == 2
    assert [len(track.points) for track in result.tracks] == [3, 3]
    assert all(track.diagnostic_class == "candidate" for track in result.tracks)


def test_spherical_tracker_does_not_bootstrap_across_long_first_gap() -> None:
    sequence = _sequence(
        [0.0, 120.0, 150.0],
        [
            (40.000,),
            (40.500,),
            (40.625,),
        ],
    )
    config = SphericalTrackingConfig(min_track_length=2)

    result = SphericalTracker(config).track(sequence, _calibration())

    # The first candidate expires before the second frame. The second and third
    # candidates form a new two-point track.
    assert len(result.tracks) == 1
    assert [point.frame_identifier for point in result.tracks[0].points] == [
        "frame-1",
        "frame-2",
    ]


def test_spherical_tracking_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="velocity_fit_points"):
        SphericalTrackingConfig(velocity_fit_points=1)
    with pytest.raises(ValueError, match="prediction_tolerance_arcmin"):
        SphericalTrackingConfig(prediction_tolerance_arcmin=0.0)
