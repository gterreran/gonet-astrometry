from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from gonet_astrometry.adapters.grid_calibration import PortableGridTransform
from gonet_astrometry.models.detection import Detection, DetectionCatalog
from gonet_astrometry.models.frame import ObserverLocation
from gonet_astrometry.models.grid import GridCalibration
from gonet_astrometry.models.track import StarTrack, TrackDiagnostics, TrackPoint
from gonet_astrometry.solving.sidereal import (
    SIDEREAL_RATE_RAD_PER_SECOND,
    SiderealFitConfig,
)
from gonet_astrometry.solving.stellar_tracks import (
    SiderealTrackMerger,
    StellarTrackMergeConfig,
)
from gonet_astrometry.tracking.image_plane import ImagePlaneTrackingResult
from gonet_astrometry.tracking.sequence import DetectionEpoch, DetectionSequence


class IdentityPolarEvaluator:
    sensor_width_px = 200
    sensor_height_px = 400
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


def _unit(value: np.ndarray) -> np.ndarray:
    return value / np.linalg.norm(value)


def _rotate(
    vector: np.ndarray,
    axis: np.ndarray,
    angle: float,
) -> np.ndarray:
    return (
        vector * np.cos(angle)
        + np.cross(axis, vector) * np.sin(angle)
        + axis * np.dot(vector, axis) * (1.0 - np.cos(angle))
    )


def _pixel_from_ray(ray: np.ndarray) -> tuple[float, float]:
    radius = np.rad2deg(np.arccos(np.clip(ray[2], -1.0, 1.0)))
    theta = np.mod(
        np.rad2deg(np.arctan2(ray[1], ray[0])),
        360.0,
    )
    return float(radius), float(theta)


def _diagnostics(
    point_count: int,
    duration_s: float,
) -> TrackDiagnostics:
    return TrackDiagnostics(
        duration_s=duration_s,
        displacement_px=10.0,
        mean_speed_px_per_minute=1.0,
        fit_rms_px=0.1,
        missed_frames=0,
        diagnostic_class="candidate",
        detection_count=point_count,
        span_epoch_count=point_count,
        coverage_fraction=1.0,
        median_interval_s=duration_s / max(point_count - 1, 1),
        max_interval_s=duration_s / max(point_count - 1, 1),
    )


def _synthetic_tracking(
    *,
    include_overlapping_fragment: bool = False,
) -> tuple[
    ImagePlaneTrackingResult,
    np.ndarray,
    GridCalibration,
]:
    axis = _unit(np.asarray([0.25, -0.35, 0.9027735], dtype=float))
    reference_stars = [
        _unit(np.asarray([0.70, 0.10, 0.70], dtype=float)),
        _unit(np.asarray([-0.30, 0.80, 0.52], dtype=float)),
        _unit(np.asarray([0.55, -0.70, 0.45], dtype=float)),
        _unit(np.asarray([-0.65, -0.30, 0.70], dtype=float)),
    ]

    start = datetime(2026, 8, 22, tzinfo=timezone.utc)
    elapsed = np.asarray(
        [0.0, 360.0, 720.0, 1080.0, 1440.0, 1800.0],
        dtype=float,
    )
    location = ObserverLocation(41.88, -87.63, 180.0)

    epochs = []
    for epoch_index, seconds in enumerate(elapsed):
        detections = []
        for star_index, star in enumerate(reference_stars):
            ray = _rotate(
                star,
                axis,
                SIDEREAL_RATE_RAD_PER_SECOND * float(seconds),
            )
            x, y = _pixel_from_ray(ray)
            detections.append(
                Detection(
                    identifier=star_index,
                    x=x,
                    y=y,
                    flux=100.0,
                    signal_to_noise=20.0,
                    x_uncertainty=0.05,
                    y_uncertainty=0.05,
                    flags=("channel-support:4",),
                )
            )

        if include_overlapping_fragment and 2 <= epoch_index <= 4:
            target = reference_stars[3]
            ray = _rotate(
                target,
                axis,
                SIDEREAL_RATE_RAD_PER_SECOND * float(seconds),
            )
            x, y = _pixel_from_ray(ray)
            detections.append(
                Detection(
                    identifier=99,
                    x=x,
                    y=y,
                    flux=50.0,
                    signal_to_noise=10.0,
                    x_uncertainty=0.05,
                    y_uncertainty=0.05,
                    flags=("channel-support:2",),
                )
            )

        frame_id = f"frame-{epoch_index}"
        epochs.append(
            DetectionEpoch(
                frame_identifier=frame_id,
                source_path=Path(frame_id),
                exposure_midpoint=start + timedelta(seconds=float(seconds)),
                location=location,
                image_shape=(400, 200),
                sensor_orientation="native",
                catalog=DetectionCatalog(
                    frame_id,
                    tuple(detections),
                    "synthetic-multichannel",
                ),
            )
        )

    sequence = DetectionSequence(tuple(epochs))
    tracks = []

    # Three long informative seed tracks establish the common sidereal pole.
    for star_index in range(3):
        tracks.append(
            StarTrack(
                identifier=10 + star_index,
                points=tuple(
                    TrackPoint(epoch.frame_identifier, star_index) for epoch in epochs
                ),
                diagnostics=_diagnostics(
                    len(epochs),
                    float(elapsed[-1]),
                ),
            )
        )

    # The fourth physical star is deliberately split into two temporal
    # fragments. Each fragment is too short for the preliminary pole fit, but
    # together they become a strong full stellar track.
    tracks.append(
        StarTrack(
            identifier=20,
            points=tuple(
                TrackPoint(epochs[index].frame_identifier, 3) for index in range(3)
            ),
            diagnostics=_diagnostics(3, float(elapsed[2])),
        )
    )
    tracks.append(
        StarTrack(
            identifier=21,
            points=tuple(
                TrackPoint(epochs[index].frame_identifier, 3) for index in range(3, 6)
            ),
            diagnostics=_diagnostics(
                3,
                float(elapsed[5] - elapsed[3]),
            ),
        )
    )

    if include_overlapping_fragment:
        tracks.append(
            StarTrack(
                identifier=22,
                points=tuple(
                    TrackPoint(epochs[index].frame_identifier, 99)
                    for index in range(2, 5)
                ),
                diagnostics=_diagnostics(
                    3,
                    float(elapsed[4] - elapsed[2]),
                ),
            )
        )

    calibration = GridCalibration(
        transform=PortableGridTransform(IdentityPolarEvaluator()),
        image_shape=(400, 200),
        coordinate_convention=(IdentityPolarEvaluator.image_coordinate_convention),
        source="synthetic-grid.npz",
    )
    return (
        ImagePlaneTrackingResult(sequence, tuple(tracks)),
        axis,
        calibration,
    )


def _merger() -> SiderealTrackMerger:
    return SiderealTrackMerger(
        StellarTrackMergeConfig(
            min_fragment_points=3,
            merge_radius_deg=10.0 / 60.0,
            merged_rms_deg=15.0 / 60.0,
        ),
        SiderealFitConfig(
            min_track_points=5,
            min_track_duration_minutes=5.0,
            min_track_span_deg=0.1,
            robust_scale_deg=0.02,
            consistent_rms_deg=0.05,
        ),
    )


def test_merger_recovers_split_physical_star() -> None:
    tracking, expected_axis, calibration = _synthetic_tracking()

    result = _merger().fit_and_merge(
        tracking,
        calibration,
    )

    assert len(tracking.tracks) == 5
    assert len(result.tracking.tracks) == 4
    assert any(set(source_ids) == {20, 21} for source_ids in result.source_fragment_ids)

    merged_index = next(
        index
        for index, source_ids in enumerate(result.source_fragment_ids)
        if set(source_ids) == {20, 21}
    )
    assert len(result.tracking.tracks[merged_index].points) == 6
    assert (
        abs(
            float(
                np.dot(
                    result.final_solution.axis_grid,
                    expected_axis,
                )
            )
        )
        > 0.999
    )


def test_merger_never_combines_fragments_that_share_an_epoch() -> None:
    tracking, _, calibration = _synthetic_tracking(include_overlapping_fragment=True)

    result = _merger().fit_and_merge(
        tracking,
        calibration,
    )

    assert any(set(source_ids) == {20, 21} for source_ids in result.source_fragment_ids)
    assert all(22 not in source_ids for source_ids in result.source_fragment_ids)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"min_fragment_points": 1}, "min_fragment_points"),
        ({"merge_radius_deg": 0.0}, "merge_radius_deg"),
        ({"merged_rms_deg": 0.0}, "merged_rms_deg"),
    ],
)
def test_merge_config_rejects_invalid_values(
    kwargs: dict[str, float | int],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        StellarTrackMergeConfig(**kwargs)
