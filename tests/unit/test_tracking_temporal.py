from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from gonet_astrometry.models.detection import Detection, DetectionCatalog
from gonet_astrometry.models.frame import ObserverLocation
from gonet_astrometry.tracking.config import TrackingConfig
from gonet_astrometry.tracking.image_plane import ImagePlaneTracker
from gonet_astrometry.tracking.sequence import DetectionEpoch, DetectionSequence
from gonet_astrometry.tracking.temporal import (
    sequence_timing_diagnostics,
    track_population_diagnostics,
)


def _epoch(index: int, seconds: float, x: float) -> DetectionEpoch:
    frame_id = f"frame-{index}.jpg"
    detection = Detection(index, x, 20.0, 10.0, 10.0, 0.2, 0.2)
    return DetectionEpoch(
        frame_identifier=frame_id,
        source_path=Path(frame_id),
        exposure_midpoint=datetime(2026, 8, 8, tzinfo=timezone.utc)
        + timedelta(seconds=seconds),
        location=ObserverLocation(41.0, -87.0, 180.0),
        image_shape=(100, 100),
        sensor_orientation="native",
        catalog=DetectionCatalog(frame_id, (detection,), "synthetic"),
    )


def test_sequence_timing_diagnostics_preserve_irregular_cadence() -> None:
    sequence = DetectionSequence.from_epochs(
        [_epoch(0, 0.0, 10.0), _epoch(1, 2.0, 10.1), _epoch(2, 302.0, 15.0)]
    )
    diagnostics = sequence_timing_diagnostics(sequence)

    assert diagnostics.epoch_count == 3
    assert diagnostics.duration_s == pytest.approx(302.0)
    assert diagnostics.min_interval_s == pytest.approx(2.0)
    assert diagnostics.median_interval_s == pytest.approx(151.0)
    assert diagnostics.max_interval_s == pytest.approx(300.0)


def test_track_population_diagnostics_measure_fragmentation() -> None:
    sequence = DetectionSequence.from_epochs(
        [_epoch(0, 0.0, 10.0), _epoch(1, 60.0, 11.0), _epoch(2, 120.0, 12.0)]
    )
    result = ImagePlaneTracker(
        TrackingConfig(
            max_speed_px_per_minute=5.0,
            prediction_tolerance_px=2.0,
            min_track_length=3,
        )
    ).track(sequence)
    diagnostics = track_population_diagnostics(result)

    assert diagnostics.track_count == 1
    assert diagnostics.min_length == 3
    assert diagnostics.median_length == pytest.approx(3.0)
    assert diagnostics.max_length == 3
    assert diagnostics.max_duration_s == pytest.approx(120.0)
    assert diagnostics.median_coverage_fraction == pytest.approx(1.0)


def test_empty_track_population_has_zero_summary() -> None:
    sequence = DetectionSequence.from_epochs(
        [_epoch(0, 0.0, 10.0), _epoch(1, 60.0, 50.0)]
    )
    result = ImagePlaneTracker(
        TrackingConfig(max_speed_px_per_minute=1.0, min_track_length=2)
    ).track(sequence)
    assert track_population_diagnostics(result).track_count == 0


def test_sequence_timing_diagnostics_support_single_epoch() -> None:
    sequence = DetectionSequence.from_epochs([_epoch(0, 0.0, 10.0)])
    diagnostics = sequence_timing_diagnostics(sequence)

    assert diagnostics.epoch_count == 1
    assert diagnostics.duration_s == pytest.approx(0.0)
    assert diagnostics.min_interval_s == pytest.approx(0.0)
    assert diagnostics.median_interval_s == pytest.approx(0.0)
    assert diagnostics.p90_interval_s == pytest.approx(0.0)
    assert diagnostics.max_interval_s == pytest.approx(0.0)
