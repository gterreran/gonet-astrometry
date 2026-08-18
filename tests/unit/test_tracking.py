from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from gonet_astrometry.models.detection import Detection, DetectionCatalog
from gonet_astrometry.models.frame import ObserverLocation
from gonet_astrometry.tracking.config import TrackingConfig
from gonet_astrometry.tracking.image_plane import ImagePlaneTracker
from gonet_astrometry.tracking.sequence import (
    DetectionEpoch,
    DetectionSequence,
    location_separation_m,
)


def _detection(identifier: int, x: float, y: float) -> Detection:
    return Detection(identifier, x, y, 10.0, 10.0, 0.2, 0.2)


def _epoch(
    index: int,
    seconds: float,
    detections: tuple[Detection, ...],
    *,
    location: ObserverLocation | None = None,
) -> DetectionEpoch:
    frame_id = f"frame-{index}.jpg"
    catalog = DetectionCatalog(frame_id, detections, "synthetic")
    return DetectionEpoch(
        frame_identifier=frame_id,
        source_path=Path(frame_id),
        exposure_midpoint=datetime(2026, 8, 8, tzinfo=timezone.utc)
        + timedelta(seconds=seconds),
        location=location or ObserverLocation(41.0, -87.0, 180.0),
        image_shape=(3040, 4056),
        sensor_orientation="native",
        catalog=catalog,
    )


def test_detection_sequence_sorts_and_validates_epochs() -> None:
    first = _epoch(0, 0.0, (_detection(0, 1.0, 2.0),))
    second = _epoch(1, 60.0, (_detection(0, 2.0, 3.0),))
    sequence = DetectionSequence.from_epochs([second, first])

    assert sequence.epochs == (first, second)
    assert sequence.duration_seconds == pytest.approx(60.0)
    assert sequence.total_detections == 2
    assert sequence.epoch_intervals_seconds == (60.0,)
    assert sequence.catalog_for(first.frame_identifier) is first.catalog
    assert sequence.catalog_for("missing") is None


def test_detection_sequence_rejects_geometry_and_duplicate_times() -> None:
    first = _epoch(0, 0.0, ())
    duplicate_time = _epoch(1, 0.0, ())
    with pytest.raises(ValueError, match="timestamps"):
        DetectionSequence((first, duplicate_time))

    different_shape = DetectionEpoch(
        frame_identifier="different.jpg",
        source_path=Path("different.jpg"),
        exposure_midpoint=first.exposure_midpoint + timedelta(seconds=1),
        location=first.location,
        image_shape=(100, 100),
        sensor_orientation="native",
        catalog=DetectionCatalog("different.jpg", (), "synthetic"),
    )
    with pytest.raises(ValueError, match="shapes"):
        DetectionSequence((first, different_shape))


def test_location_validation_allows_small_gps_jitter_and_rejects_moves() -> None:
    first = ObserverLocation(41.0, -87.0, 180.0)
    nearby = ObserverLocation(41.0001, -87.0001, 181.0)
    far = ObserverLocation(42.0, -87.0, 180.0)

    assert location_separation_m(first, nearby) < 20.0
    sequence = DetectionSequence.from_epochs(
        [_epoch(0, 0.0, (), location=first), _epoch(1, 60.0, (), location=nearby)]
    )
    sequence.validate_locations(50.0)

    moved = DetectionSequence.from_epochs(
        [_epoch(0, 0.0, (), location=first), _epoch(1, 60.0, (), location=far)]
    )
    with pytest.raises(ValueError, match="observing locations differ"):
        moved.validate_locations(50.0)


def test_tracker_links_smooth_motion_and_labels_low_motion() -> None:
    sequence = DetectionSequence.from_epochs(
        [
            _epoch(
                0,
                0.0,
                (
                    _detection(0, 100.0, 100.0),
                    _detection(1, 300.0, 100.0),
                    _detection(2, 500.0, 500.0),
                ),
            ),
            _epoch(
                1,
                60.0,
                (
                    _detection(0, 105.0, 102.0),
                    _detection(1, 297.0, 104.0),
                    _detection(2, 500.1, 500.0),
                    _detection(3, 900.0, 900.0),
                ),
            ),
            _epoch(
                2,
                120.0,
                (
                    _detection(0, 110.0, 104.0),
                    _detection(1, 294.0, 108.0),
                    _detection(2, 500.2, 500.0),
                ),
            ),
        ]
    )
    result = ImagePlaneTracker(
        TrackingConfig(
            max_speed_px_per_minute=15.0,
            prediction_tolerance_px=2.0,
            min_track_length=3,
            low_motion_threshold_px_per_minute=0.5,
        )
    ).track(sequence)

    assert len(result.tracks) == 3
    assert result.diagnostic_counts() == {
        "candidate": 2,
        "low-motion": 1,
        "poor-fit": 0,
    }
    assert result.assigned_detection_count == 9
    assert result.unassigned_detection_count == 1
    assert all(track.quality == pytest.approx(1.0) for track in result.tracks)

    moving = next(track for track in result.tracks if track.identifier == 0)
    resolved = result.resolve(moving)
    assert [point.detection.x for point in resolved] == [100.0, 105.0, 110.0]
    assert moving.diagnostics is not None
    assert moving.diagnostics.mean_speed_px_per_minute == pytest.approx(
        (5.0**2 + 2.0**2) ** 0.5
    )


def test_tracker_bridges_configured_missing_epoch() -> None:
    sequence = DetectionSequence.from_epochs(
        [
            _epoch(0, 0.0, (_detection(0, 100.0, 100.0),)),
            _epoch(1, 60.0, ()),
            _epoch(2, 120.0, (_detection(0, 110.0, 100.0),)),
        ]
    )

    bridged = ImagePlaneTracker(
        TrackingConfig(
            max_speed_px_per_minute=10.0,
            prediction_tolerance_px=2.0,
            max_gap_frames=1,
            min_track_length=2,
        )
    ).track(sequence)
    assert len(bridged.tracks) == 1
    assert bridged.tracks[0].diagnostics is not None
    assert bridged.tracks[0].diagnostics.missed_frames == 1

    strict = ImagePlaneTracker(
        TrackingConfig(
            max_speed_px_per_minute=10.0,
            prediction_tolerance_px=2.0,
            max_gap_frames=0,
            min_track_length=2,
        )
    ).track(sequence)
    assert strict.tracks == ()


def test_tracker_uses_elapsed_time_instead_of_epoch_count_for_gaps() -> None:
    sequence = DetectionSequence.from_epochs(
        [
            _epoch(0, 0.0, (_detection(0, 100.0, 100.0),)),
            _epoch(1, 1.0, ()),
            _epoch(2, 2.0, ()),
            _epoch(3, 300.0, (_detection(0, 105.0, 100.0),)),
        ]
    )
    result = ImagePlaneTracker(
        TrackingConfig(
            max_speed_px_per_minute=10.0,
            prediction_tolerance_px=2.0,
            max_gap_minutes=6.0,
            max_gap_frames=None,
            min_track_length=2,
        )
    ).track(sequence)

    assert len(result.tracks) == 1
    diagnostics = result.tracks[0].diagnostics
    assert diagnostics is not None
    assert diagnostics.detection_count == 2
    assert diagnostics.span_epoch_count == 4
    assert diagnostics.coverage_fraction == pytest.approx(0.5)
    assert diagnostics.max_interval_s == pytest.approx(300.0)


def test_tracker_expires_track_after_elapsed_time_limit() -> None:
    sequence = DetectionSequence.from_epochs(
        [
            _epoch(0, 0.0, (_detection(0, 100.0, 100.0),)),
            _epoch(1, 901.0, (_detection(0, 101.0, 100.0),)),
        ]
    )
    result = ImagePlaneTracker(
        TrackingConfig(max_gap_minutes=15.0, min_track_length=2)
    ).track(sequence)
    assert result.tracks == ()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_speed_px_per_minute": 0.0}, "max_speed"),
        ({"prediction_tolerance_px": 0.0}, "prediction_tolerance"),
        ({"max_gap_minutes": 0.0}, "max_gap_minutes"),
        ({"max_gap_frames": -1}, "max_gap_frames"),
        ({"max_prediction_gap_scale": 0.9}, "max_prediction_gap_scale"),
        ({"min_track_length": 1}, "min_track_length"),
        ({"low_motion_threshold_px_per_minute": -1.0}, "low_motion"),
        ({"poor_fit_rms_px": 0.0}, "poor_fit"),
        ({"location_tolerance_m": -1.0}, "location_tolerance"),
    ],
)
def test_tracking_config_rejects_invalid_values(
    kwargs: dict[str, float | int],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        TrackingConfig(**kwargs)


def test_detection_epoch_validates_metadata_and_uses_identifier_as_path() -> None:
    from datetime import datetime

    catalog = DetectionCatalog("frame.jpg", (), "synthetic")
    with pytest.raises(ValueError, match="timezone-aware"):
        DetectionEpoch(
            "frame.jpg",
            Path("frame.jpg"),
            datetime(2026, 8, 8),
            ObserverLocation(0.0, 0.0),
            (10, 10),
            "native",
            catalog,
        )

    with pytest.raises(ValueError, match="frame_identifier"):
        DetectionEpoch(
            "other.jpg",
            Path("other.jpg"),
            datetime(2026, 8, 8, tzinfo=timezone.utc),
            ObserverLocation(0.0, 0.0),
            (10, 10),
            "native",
            catalog,
        )

    with pytest.raises(ValueError, match="image_shape"):
        DetectionEpoch(
            "frame.jpg",
            Path("frame.jpg"),
            datetime(2026, 8, 8, tzinfo=timezone.utc),
            ObserverLocation(0.0, 0.0),
            (0, 10),
            "native",
            catalog,
        )


def test_detection_epoch_from_frame_falls_back_to_identifier_path() -> None:
    import numpy as np

    from gonet_astrometry.models.frame import ImageFrame, ImageMetadata

    metadata = ImageMetadata(
        datetime(2026, 8, 8, tzinfo=timezone.utc),
        10.0,
        ObserverLocation(0.0, 0.0),
    )
    frame = ImageFrame(np.ones((4, 6)), metadata)
    catalog = DetectionCatalog("fallback.jpg", (), "synthetic")

    epoch = DetectionEpoch.from_frame("fallback.jpg", frame, catalog)
    assert epoch.source_path == Path("fallback.jpg")
    assert epoch.image_shape == (4, 6)


def test_detection_sequence_rejects_too_few_duplicate_ids_and_orientation() -> None:
    first = _epoch(0, 0.0, ())
    with pytest.raises(ValueError, match="at least two"):
        DetectionSequence((first,))

    duplicate_id = DetectionEpoch(
        frame_identifier=first.frame_identifier,
        source_path=Path("duplicate.jpg"),
        exposure_midpoint=first.exposure_midpoint + timedelta(seconds=1),
        location=first.location,
        image_shape=first.image_shape,
        sensor_orientation="native",
        catalog=DetectionCatalog(first.frame_identifier, (), "synthetic"),
    )
    with pytest.raises(ValueError, match="identifiers"):
        DetectionSequence((first, duplicate_id))

    different_orientation = DetectionEpoch(
        frame_identifier="orientation.jpg",
        source_path=Path("orientation.jpg"),
        exposure_midpoint=first.exposure_midpoint + timedelta(seconds=1),
        location=first.location,
        image_shape=first.image_shape,
        sensor_orientation="rotated",
        catalog=DetectionCatalog("orientation.jpg", (), "synthetic"),
    )
    with pytest.raises(ValueError, match="orientations"):
        DetectionSequence((first, different_orientation))


def test_detection_sequence_can_validate_location_during_construction() -> None:
    first = _epoch(0, 0.0, (), location=ObserverLocation(0.0, 0.0))
    second = _epoch(1, 60.0, (), location=ObserverLocation(1.0, 0.0))
    with pytest.raises(ValueError, match="observing locations differ"):
        DetectionSequence.from_epochs(
            [first, second],
            location_tolerance_m=100.0,
        )
    with pytest.raises(ValueError, match="cannot be negative"):
        DetectionSequence.from_epochs(
            [first, second],
            location_tolerance_m=-1.0,
        )


def test_tracker_marks_irregular_four_point_track_as_poor_fit() -> None:
    sequence = DetectionSequence.from_epochs(
        [
            _epoch(0, 0.0, (_detection(0, 100.0, 100.0),)),
            _epoch(1, 60.0, (_detection(0, 105.0, 100.0),)),
            _epoch(2, 120.0, (_detection(0, 110.0, 100.0),)),
            _epoch(3, 180.0, (_detection(0, 125.0, 100.0),)),
        ]
    )
    result = ImagePlaneTracker(
        TrackingConfig(
            max_speed_px_per_minute=30.0,
            prediction_tolerance_px=20.0,
            min_track_length=4,
            poor_fit_rms_px=0.1,
        )
    ).track(sequence)

    assert len(result.tracks) == 1
    assert result.tracks[0].diagnostic_class == "poor-fit"
    assert result.tracks[0].quality is not None
    assert result.tracks[0].quality < 1.0


def test_location_validation_uses_best_reference_not_first_epoch() -> None:
    center = ObserverLocation(41.0, -87.0, 180.0)
    west = ObserverLocation(41.0, -87.0010, 180.0)
    east = ObserverLocation(41.0, -86.9990, 180.0)
    sequence = DetectionSequence.from_epochs(
        [
            _epoch(0, 0.0, (), location=west),
            _epoch(1, 60.0, (), location=center),
            _epoch(2, 120.0, (), location=east),
        ]
    )

    assert location_separation_m(west, east) > 150.0
    sequence.validate_locations(100.0)
