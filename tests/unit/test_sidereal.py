from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from gonet_astrometry.adapters.grid_calibration import PortableGridTransform
from gonet_astrometry.models.detection import Detection, DetectionCatalog
from gonet_astrometry.models.frame import ObserverLocation
from gonet_astrometry.models.grid import GridCalibration
from gonet_astrometry.models.track import (
    StarTrack,
    TrackDiagnostics,
    TrackPoint,
)
from gonet_astrometry.solving.sidereal import (
    SIDEREAL_RATE_RAD_PER_SECOND,
    SiderealAxisFitter,
    SiderealFitConfig,
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


def _rotate(vector: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    return (
        vector * np.cos(angle)
        + np.cross(axis, vector) * np.sin(angle)
        + axis * np.dot(vector, axis) * (1.0 - np.cos(angle))
    )


def _pixel_from_ray(ray: np.ndarray) -> tuple[float, float]:
    r = np.rad2deg(np.arccos(np.clip(ray[2], -1.0, 1.0)))
    theta = np.mod(np.rad2deg(np.arctan2(ray[1], ray[0])), 360.0)
    return float(r), float(theta)


def _tracking_result() -> tuple[ImagePlaneTrackingResult, np.ndarray, GridCalibration]:
    axis = np.asarray([0.25, -0.35, 0.9027735], dtype=float)
    axis /= np.linalg.norm(axis)
    stars = [
        np.asarray([0.7, 0.1, 0.70710678]),
        np.asarray([-0.3, 0.8, 0.51961524]),
        np.asarray([0.4, -0.8, 0.4472136]),
        np.asarray([-0.75, -0.2, 0.630476]),
        np.asarray([0.1, 0.95, 0.295804]),
        np.asarray([0.85, -0.35, 0.3937]),
    ]
    stars = [star / np.linalg.norm(star) for star in stars]
    elapsed = np.linspace(0.0, 2.5 * 3600.0, 18)
    start = datetime(2026, 8, 18, 0, 0, tzinfo=timezone.utc)
    epochs = []
    for epoch_index, seconds in enumerate(elapsed):
        detections = []
        for star_index, star in enumerate(stars):
            ray = _rotate(star, axis, SIDEREAL_RATE_RAD_PER_SECOND * seconds)
            x, y = _pixel_from_ray(ray)
            detections.append(
                Detection(
                    identifier=star_index,
                    x=x,
                    y=y,
                    flux=100.0,
                    signal_to_noise=20.0,
                    x_uncertainty=0.1,
                    y_uncertainty=0.1,
                )
            )
        frame_id = f"frame-{epoch_index}"
        catalog = DetectionCatalog(frame_id, tuple(detections), "synthetic")
        epochs.append(
            DetectionEpoch(
                frame_identifier=frame_id,
                source_path=Path(frame_id),
                exposure_midpoint=start + timedelta(seconds=float(seconds)),
                location=ObserverLocation(42.0, -88.0, 200.0),
                image_shape=(400, 200),
                sensor_orientation="native",
                catalog=catalog,
            )
        )
    sequence = DetectionSequence(tuple(epochs))
    tracks = tuple(
        StarTrack(
            identifier=star_index,
            points=tuple(
                TrackPoint(epoch.frame_identifier, star_index) for epoch in epochs
            ),
        )
        for star_index in range(len(stars))
    )
    calibration = GridCalibration(
        transform=PortableGridTransform(IdentityPolarEvaluator()),
        image_shape=(400, 200),
        coordinate_convention=IdentityPolarEvaluator.image_coordinate_convention,
        source="synthetic.npz",
    )
    return ImagePlaneTrackingResult(sequence, tracks), axis, calibration


def _tracking_result_with_conflicting_tracks() -> (
    tuple[ImagePlaneTrackingResult, np.ndarray, GridCalibration]
):
    base, expected_axis, calibration = _tracking_result()
    elapsed = np.asarray(
        [
            (
                epoch.exposure_midpoint - base.sequence.epochs[0].exposure_midpoint
            ).total_seconds()
            for epoch in base.sequence.epochs
        ],
        dtype=float,
    )
    false_axes = [
        np.asarray([np.cos(phi), np.sin(phi), 0.25], dtype=float)
        for phi in np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False)
    ]
    false_axes = [axis / np.linalg.norm(axis) for axis in false_axes]
    false_stars = [
        np.asarray([0.35 + 0.02 * index, -0.55, 0.75], dtype=float)
        for index in range(len(false_axes))
    ]
    false_stars = [star / np.linalg.norm(star) for star in false_stars]

    epochs = []
    for epoch_index, epoch in enumerate(base.sequence.epochs):
        detections = list(epoch.catalog.detections)
        for false_index, (axis, star) in enumerate(
            zip(false_axes, false_stars, strict=True)
        ):
            ray = _rotate(
                star,
                axis,
                SIDEREAL_RATE_RAD_PER_SECOND * float(elapsed[epoch_index]),
            )
            x, y = _pixel_from_ray(ray)
            detections.append(
                Detection(
                    identifier=100 + false_index,
                    x=x,
                    y=y,
                    flux=80.0,
                    signal_to_noise=15.0,
                    x_uncertainty=0.1,
                    y_uncertainty=0.1,
                )
            )
        epochs.append(
            DetectionEpoch(
                frame_identifier=epoch.frame_identifier,
                source_path=epoch.source_path,
                exposure_midpoint=epoch.exposure_midpoint,
                location=epoch.location,
                image_shape=epoch.image_shape,
                sensor_orientation=epoch.sensor_orientation,
                catalog=DetectionCatalog(
                    epoch.frame_identifier, tuple(detections), "synthetic"
                ),
            )
        )

    tracks = list(base.tracks)
    for false_index in range(len(false_axes)):
        tracks.append(
            StarTrack(
                identifier=100 + false_index,
                points=tuple(
                    TrackPoint(epoch.frame_identifier, 100 + false_index)
                    for epoch in epochs
                ),
                diagnostics=TrackDiagnostics(
                    duration_s=float(elapsed[-1]),
                    displacement_px=10.0,
                    mean_speed_px_per_minute=1.0,
                    fit_rms_px=1.0,
                    missed_frames=0,
                    diagnostic_class="candidate",
                    detection_count=len(epochs),
                    span_epoch_count=len(epochs),
                    coverage_fraction=1.0,
                    median_interval_s=float(np.median(np.diff(elapsed))),
                    max_interval_s=float(np.max(np.diff(elapsed))),
                ),
            )
        )
    return (
        ImagePlaneTrackingResult(DetectionSequence(tuple(epochs)), tuple(tracks)),
        expected_axis,
        calibration,
    )


def test_sidereal_axis_fitter_recovers_known_rotation_axis() -> None:
    tracking, expected_axis, calibration = _tracking_result()
    solution = SiderealAxisFitter(
        SiderealFitConfig(
            min_track_points=5,
            min_track_duration_minutes=10.0,
            min_track_span_deg=0.1,
            robust_scale_deg=0.01,
            consistent_rms_deg=0.02,
        )
    ).fit(tracking, calibration)

    assert np.dot(solution.axis_grid, expected_axis) > 0.99999
    assert solution.fit_rms_deg < 1e-4
    counts = solution.diagnostic_counts()
    assert counts["sidereal-consistent"] == 6
    assert counts["sidereal-rejected"] == 0


def test_sidereal_axis_fitter_uses_common_axis_consensus_with_conflicting_tracks() -> (
    None
):
    tracking, expected_axis, calibration = _tracking_result_with_conflicting_tracks()
    solution = SiderealAxisFitter(
        SiderealFitConfig(
            min_track_points=5,
            min_track_duration_minutes=10.0,
            min_track_span_deg=0.1,
            robust_scale_deg=0.02,
            consistent_rms_deg=0.05,
        )
    ).fit(tracking, calibration)

    assert abs(float(np.dot(solution.axis_grid, expected_axis))) > 0.999
    counts = solution.diagnostic_counts()
    assert counts["sidereal-consistent"] >= 6
    assert counts["sidereal-rejected"] >= 10


def test_sidereal_axis_fitter_validates_sensor_shape() -> None:
    tracking, _, calibration = _tracking_result()
    bad = GridCalibration(
        transform=calibration.transform,
        image_shape=(10, 10),
        coordinate_convention=calibration.coordinate_convention,
    )
    with pytest.raises(ValueError, match="sensor shape"):
        SiderealAxisFitter().fit(tracking, bad)


def test_sidereal_fit_config_validates_limits() -> None:
    bad_values = [
        {"min_track_points": 2},
        {"min_track_duration_minutes": 0.0},
        {"min_track_span_deg": 0.0},
        {"robust_scale_deg": 0.0},
        {"consistent_rms_deg": 0.0},
        {"max_fit_tracks": 2},
        {"max_fit_points_per_track": 2},
        {"inverse_reprojection_tolerance_px": 0.0},
    ]
    for kwargs in bad_values:
        with pytest.raises(ValueError):
            SiderealFitConfig(**kwargs)
