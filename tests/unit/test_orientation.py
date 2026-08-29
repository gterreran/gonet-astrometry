from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from gonet_astrometry.adapters.grid_calibration import PortableGridTransform
from gonet_astrometry.catalogs.base import CatalogStar
from gonet_astrometry.geometry.horizon import altaz_to_enu, enu_to_altaz, ncp_enu_vector
from gonet_astrometry.models.detection import Detection, DetectionCatalog
from gonet_astrometry.models.frame import ObserverLocation
from gonet_astrometry.models.grid import GridCalibration
from gonet_astrometry.models.track import StarTrack, TrackPoint
from gonet_astrometry.solving.orientation import (
    AbsoluteOrientationSolver,
    OrientationFitConfig,
)
from gonet_astrometry.solving.sidereal import (
    SIDEREAL_RATE_RAD_PER_SECOND,
    SiderealRotationSolution,
    SiderealTrackDiagnostics,
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


def _rotate(vector: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    return (
        vector * np.cos(angle)
        + np.cross(axis, vector) * np.sin(angle)
        + axis * np.dot(vector, axis) * (1.0 - np.cos(angle))
    )


def _pixel_from_ray(ray: np.ndarray) -> tuple[float, float]:
    radius = np.rad2deg(np.arccos(np.clip(ray[2], -1.0, 1.0)))
    theta = np.mod(np.rad2deg(np.arctan2(ray[1], ray[0])), 360.0)
    return float(radius), float(theta)


def _basis(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    preferred = np.array([0.0, 0.0, 1.0])
    first = preferred - axis * np.dot(preferred, axis)
    if np.linalg.norm(first) < 1e-8:
        preferred = np.array([1.0, 0.0, 0.0])
        first = preferred - axis * np.dot(preferred, axis)
    first = _unit(first)
    second = _unit(np.cross(axis, first))
    return first, second


def _orientation_matrix(ncp_grid: np.ndarray, ncp_enu: np.ndarray, twist: float):
    g1, g2 = _basis(ncp_grid)
    e1, e2 = _basis(ncp_enu)
    c = np.cos(twist)
    s = np.sin(twist)
    rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return (
        np.column_stack((e1, e2, ncp_enu)) @ rz @ np.column_stack((g1, g2, ncp_grid)).T
    )


def _synthetic_orientation_case():
    signed_axis = _unit(np.array([0.20, -0.35, 0.915]))
    ncp_grid = -signed_axis
    location = ObserverLocation(42.0, -88.0, 200.0)
    ncp_enu = ncp_enu_vector(location.latitude_deg)
    true_twist = np.deg2rad(73.0)
    true_matrix = _orientation_matrix(ncp_grid, ncp_enu, true_twist)

    g1, g2 = _basis(ncp_grid)
    declinations = np.linspace(15.0, 70.0, 12)
    phases = np.deg2rad(np.linspace(5.0, 325.0, 12))
    reference_rays = []
    for declination, phase in zip(declinations, phases, strict=True):
        dec = np.deg2rad(declination)
        reference_rays.append(
            np.cos(dec) * np.cos(phase) * g1
            + np.cos(dec) * np.sin(phase) * g2
            + np.sin(dec) * ncp_grid
        )
    reference_rays = np.asarray(reference_rays)

    start = datetime(2026, 8, 18, 0, 0, tzinfo=timezone.utc)
    elapsed = np.linspace(0.0, 4.0 * 3600.0, 24)
    reference_seconds = float(elapsed[-1] / 2.0)
    epochs = []
    for epoch_index, seconds in enumerate(elapsed):
        detections = []
        for star_index, reference_ray in enumerate(reference_rays):
            ray = _rotate(
                reference_ray,
                signed_axis,
                SIDEREAL_RATE_RAD_PER_SECOND * (seconds - reference_seconds),
            )
            x, y = _pixel_from_ray(ray)
            detections.append(Detection(star_index, x, y, 100.0, 20.0, 0.1, 0.1))
        frame_id = f"frame-{epoch_index}"
        epochs.append(
            DetectionEpoch(
                frame_identifier=frame_id,
                source_path=Path(frame_id),
                exposure_midpoint=start + timedelta(seconds=float(seconds)),
                location=location,
                image_shape=(400, 200),
                sensor_orientation="native",
                catalog=DetectionCatalog(frame_id, tuple(detections), "synthetic"),
            )
        )
    sequence = DetectionSequence(tuple(epochs))
    tracks = tuple(
        StarTrack(
            identifier=index,
            points=tuple(TrackPoint(epoch.frame_identifier, index) for epoch in epochs),
        )
        for index in range(len(reference_rays))
    )
    tracking = ImagePlaneTrackingResult(sequence, tracks)
    diagnostics = tuple(
        SiderealTrackDiagnostics(
            track_identifier=index,
            diagnostic_class="sidereal-consistent",
            total_point_count=len(epochs),
            valid_point_count=len(epochs),
            duration_s=float(elapsed[-1]),
            angular_span_deg=40.0,
            rms_residual_deg=0.01,
            median_residual_deg=0.005,
            max_residual_deg=0.02,
        )
        for index in range(len(reference_rays))
    )
    sidereal = SiderealRotationSolution(
        axis_grid=signed_axis,
        fit_rms_deg=0.01,
        fit_median_deg=0.005,
        fit_p95_deg=0.02,
        fitted_track_count=len(tracks),
        fitted_point_count=len(tracks) * len(epochs),
        track_diagnostics=diagnostics,
        rotation_sign=1,
    )
    calibration = GridCalibration(
        transform=PortableGridTransform(IdentityPolarEvaluator()),
        image_shape=(400, 200),
        coordinate_convention=IdentityPolarEvaluator.image_coordinate_convention,
    )

    true_catalog_rays = (true_matrix @ reference_rays.T).T
    true_stars = []
    for index, ray in enumerate(true_catalog_rays):
        dec = np.rad2deg(np.arcsin(np.clip(np.dot(ray, ncp_enu), -1.0, 1.0)))
        true_stars.append(CatalogStar(f"true-{index}", index * 10.0, dec, 3.0))

    rng = np.random.default_rng(1234)
    false_rays = []
    false_stars = []
    for index in range(40):
        vector = _unit(rng.normal(size=3))
        if vector[2] < 0.2:
            vector[2] = abs(vector[2]) + 0.3
            vector = _unit(vector)
        false_rays.append(vector)
        dec = np.rad2deg(np.arcsin(np.clip(np.dot(vector, ncp_enu), -1.0, 1.0)))
        false_stars.append(CatalogStar(f"false-{index}", index * 7.0, dec, 5.5))
    catalog = tuple(true_stars + false_stars)
    catalog_rays = np.vstack((true_catalog_rays, np.asarray(false_rays)))

    def provider(stars, epoch, observer):
        del epoch, observer
        indices = [catalog.index(star) for star in stars]
        rays = catalog_rays[indices]
        altitude = np.rad2deg(np.arcsin(np.clip(rays[:, 2], -1.0, 1.0)))
        return rays, altitude

    return tracking, sidereal, calibration, catalog, provider, true_matrix


def test_absolute_orientation_solver_recovers_known_attitude() -> None:
    tracking, sidereal, calibration, catalog, provider, expected = (
        _synthetic_orientation_case()
    )
    solution = AbsoluteOrientationSolver(
        OrientationFitConfig(
            limiting_magnitude=6.5,
            min_catalog_altitude_deg=-90.0,
            min_track_points=5,
            min_track_duration_minutes=10.0,
            deduplication_radius_deg=0.05,
            declination_tolerance_deg=0.5,
            consensus_bin_deg=0.5,
            consensus_tolerance_deg=1.0,
            match_radius_deg=0.2,
            min_matches=8,
        )
    ).fit(
        tracking,
        sidereal,
        calibration,
        catalog,
        catalog_ray_provider=provider,
    )

    assert len(solution.matches) >= 10
    assert np.allclose(solution.grid_to_enu, expected, atol=1e-6)
    assert solution.fit_median_deg < 1e-6
    assert np.allclose(
        solution.grid_to_enu @ solution.ncp_grid,
        solution.ncp_enu,
        atol=1e-8,
    )


def test_horizon_coordinate_helpers_round_trip() -> None:
    rays = altaz_to_enu(np.array([0.0, 90.0]), np.array([42.0, 0.0]))
    assert np.allclose(rays[0], ncp_enu_vector(42.0))
    azimuth, altitude = enu_to_altaz(rays[1])
    assert azimuth == pytest.approx(90.0)
    assert altitude == pytest.approx(0.0)


def test_orientation_config_validates_values() -> None:
    bad = [
        {"bootstrap_limiting_magnitude": 7.0},
        {"min_track_points": 2},
        {"min_track_duration_minutes": 0.0},
        {"deduplication_radius_deg": 0.0},
        {"declination_tolerance_deg": 0.0},
        {"consensus_bin_deg": 0.0},
        {"consensus_tolerance_deg": 0.0},
        {"match_radius_deg": 0.0},
        {"min_matches": 2},
        {"max_anchors": 2},
        {"inverse_reprojection_tolerance_px": 0.0},
        {"track_validation_rms_deg": 0.0},
        {"max_declination_robust_sigma_deg": 0.0},
        {"max_declination_p95_deg": 0.0},
    ]
    for kwargs in bad:
        with pytest.raises(ValueError):
            OrientationFitConfig(**kwargs)


def test_global_twist_search_registers_entire_declination_pattern() -> None:
    observed_phase = np.deg2rad(np.asarray([8.0, 71.0, 149.0, 236.0, 311.0]))
    observed_dec = np.asarray([12.0, 24.0, 37.0, 51.0, 68.0])
    expected_twist_deg = 83.4

    true_phase = np.mod(observed_phase + np.deg2rad(expected_twist_deg), 2.0 * np.pi)
    catalog_phase = list(true_phase)
    catalog_dec = list(observed_dec)

    # Add several individually plausible stars near each declination, but with
    # phases that cannot all be explained by one common camera twist.
    distractor_offsets = (17.0, 121.0, 247.0)
    for declination, phase in zip(observed_dec, observed_phase, strict=True):
        for offset in distractor_offsets:
            catalog_phase.append(float(np.mod(phase + np.deg2rad(offset), 2.0 * np.pi)))
            catalog_dec.append(float(declination + 0.03))

    solver = AbsoluteOrientationSolver(
        OrientationFitConfig(
            declination_tolerance_deg=0.1,
            match_radius_deg=0.1,
            consensus_bin_deg=0.25,
            consensus_tolerance_deg=1.0,
            min_matches=4,
            max_anchors=10,
        )
    )
    twist = solver._consensus_twist(  # noqa: SLF001 - focused solver regression
        observed_phase,
        observed_dec,
        np.asarray(catalog_phase, dtype=np.float64),
        np.asarray(catalog_dec, dtype=np.float64),
    )

    error = (np.rad2deg(twist) - expected_twist_deg + 180.0) % 360.0 - 180.0
    assert error == pytest.approx(0.0, abs=0.03)
