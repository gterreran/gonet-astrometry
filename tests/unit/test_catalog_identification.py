from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from gonet_astrometry.adapters.grid_calibration import PortableGridTransform
from gonet_astrometry.catalogs.base import CatalogStar
from gonet_astrometry.geometry.horizon import altaz_to_enu
from gonet_astrometry.models.detection import Detection, DetectionCatalog
from gonet_astrometry.models.frame import ObserverLocation
from gonet_astrometry.models.grid import GridCalibration
from gonet_astrometry.models.track import StarTrack, TrackPoint
from gonet_astrometry.products.stellar_identifications import (
    StellarIdentificationProduct,
    load_stellar_identification_product,
    save_stellar_identification_product,
)
from gonet_astrometry.tracking.catalog_identification import (
    CatalogSequenceMatcher,
    StellarIdentificationConfig,
)
from gonet_astrometry.tracking.image_plane import ImagePlaneTrackingResult
from gonet_astrometry.tracking.modes import (
    combine_catalog_and_fallback_tracks,
    unmatched_detection_sequence,
)
from gonet_astrometry.tracking.sequence import DetectionEpoch, DetectionSequence


class PolarEvaluator:
    sensor_width_px = 180
    sensor_height_px = 360
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


def _rotation_z(angle_deg: float) -> np.ndarray:
    angle = np.deg2rad(angle_deg)
    c = np.cos(angle)
    s = np.sin(angle)
    return np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _true_attitude() -> np.ndarray:
    base = np.asarray([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]])
    return base @ _rotation_z(-58.0)


def _pixel_from_grid_ray(ray: np.ndarray) -> tuple[float, float]:
    radius = np.rad2deg(np.arccos(np.clip(ray[2], -1.0, 1.0)))
    theta = np.mod(np.rad2deg(np.arctan2(ray[1], ray[0])), 360.0)
    return float(radius), float(theta)


def _synthetic_case():
    start = datetime(2026, 8, 29, 2, 0, tzinfo=timezone.utc)
    location = ObserverLocation(41.9, -87.6, 195.0)
    stars = tuple(
        CatalogStar(f"HR {100 + index}", index * 10.0, 10.0 + index, 1.0 + 0.15 * index)
        for index in range(10)
    )
    base_azimuth = np.linspace(15.0, 330.0, len(stars))
    altitude = np.linspace(35.0, 75.0, len(stars))
    true_matrix = _true_attitude()

    epochs = []
    ray_lookup: dict[tuple[str, datetime], np.ndarray] = {}
    for epoch_index in range(4):
        timestamp = start + timedelta(minutes=5 * epoch_index)
        detections = []
        detection_identifier = 0
        for star_index, star in enumerate(stars):
            azimuth = base_azimuth[star_index] + 1.25 * epoch_index
            ray_enu = altaz_to_enu(float(azimuth), float(altitude[star_index]))
            ray_lookup[(star.identifier, timestamp)] = ray_enu
            # Deliberately omit one expected star from alternate frames.
            if star_index == 9 and epoch_index % 2 == 0:
                continue
            ray_grid = true_matrix.T @ ray_enu
            x, y = _pixel_from_grid_ray(ray_grid)
            detections.append(
                Detection(
                    detection_identifier,
                    x,
                    y,
                    100.0,
                    20.0,
                    0.1,
                    0.1,
                    flags=("channel-support:4",),
                )
            )
            detection_identifier += 1
        # One unmatched non-stellar detection.
        detections.append(
            Detection(detection_identifier, 85.0, 5.0, 10.0, 4.0, 0.2, 0.2)
        )
        frame_id = f"frame-{epoch_index}"
        epochs.append(
            DetectionEpoch(
                frame_identifier=frame_id,
                source_path=Path(f"{frame_id}.jpg"),
                exposure_midpoint=timestamp,
                location=location,
                image_shape=(360, 180),
                sensor_orientation="native",
                catalog=DetectionCatalog(frame_id, tuple(detections), "synthetic"),
            )
        )

    def provider(selected, epoch, observer):
        del observer
        rays = np.asarray(
            [ray_lookup[(star.identifier, epoch)] for star in selected], dtype=float
        )
        return rays, np.rad2deg(np.arcsin(rays[:, 2]))

    calibration = GridCalibration(
        transform=PortableGridTransform(PolarEvaluator()),
        image_shape=(360, 180),
        coordinate_convention=PolarEvaluator.image_coordinate_convention,
    )
    return DetectionSequence(tuple(epochs)), calibration, stars, provider, true_matrix


def test_catalog_sequence_matcher_recovers_attitude_and_star_labels() -> None:
    sequence, calibration, stars, provider, expected_matrix = _synthetic_case()
    config = StellarIdentificationConfig(
        limiting_magnitude=4.5,
        bootstrap_limiting_magnitude=4.5,
        min_catalog_altitude_deg=0.0,
        bootstrap_rotation_step_deg=2.0,
        bootstrap_match_radius_px=30.0,
        refinement_radii_px=(20.0, 10.0, 5.0),
        sequence_refinement_radius_px=5.0,
        final_match_radius_px=2.0,
        bright_rescue_radius_px=4.0,
        robust_clip_floor_arcmin=1.0,
    )
    result = CatalogSequenceMatcher(config).fit_and_match(
        sequence,
        calibration,
        stars,
        catalog_ray_provider=provider,
    )

    assert np.allclose(result.grid_to_enu, expected_matrix, atol=1e-8)
    assert result.visible_star_count == 40
    assert result.matched_star_count == 38
    assert result.fit_median_residual_arcmin < 1e-5

    tracks = result.catalog_tracks(sequence)
    assert len(tracks.tracks) == 10
    assert {track.catalog_identifier for track in tracks.tracks} == {
        star.identifier for star in stars
    }
    short = next(
        track for track in tracks.tracks if track.catalog_identifier == "HR 109"
    )
    assert len(short.points) == 2



def test_hybrid_helpers_reserve_catalog_matches_for_catalog_tracks() -> None:
    sequence, calibration, stars, provider, _ = _synthetic_case()
    config = StellarIdentificationConfig(
        bootstrap_limiting_magnitude=4.5,
        bootstrap_rotation_step_deg=2.0,
        bootstrap_match_radius_px=30.0,
        refinement_radii_px=(20.0, 10.0, 5.0),
        sequence_refinement_radius_px=5.0,
        final_match_radius_px=2.0,
        bright_rescue_radius_px=4.0,
        robust_clip_floor_arcmin=1.0,
    )
    identifications = CatalogSequenceMatcher(config).fit_and_match(
        sequence,
        calibration,
        stars,
        catalog_ray_provider=provider,
    )
    catalog_tracks = identifications.catalog_tracks(sequence)
    unmatched = unmatched_detection_sequence(sequence, identifications)

    assert unmatched.total_detections == 4
    assert all(len(epoch.catalog.detections) == 1 for epoch in unmatched.epochs)

    fallback = ImagePlaneTrackingResult(
        unmatched,
        (
            StarTrack(
                identifier=0,
                points=tuple(
                    TrackPoint(
                        epoch.frame_identifier,
                        epoch.catalog.detections[0].identifier,
                    )
                    for epoch in unmatched.epochs
                ),
            ),
        ),
    )
    combined = combine_catalog_and_fallback_tracks(
        sequence,
        catalog_tracks,
        fallback,
    )

    assert len(combined.tracks) == 11
    assert [track.identifier for track in combined.tracks] == list(range(11))
    assert sum(track.catalog_identifier is not None for track in combined.tracks) == 10
    assert combined.tracks[-1].catalog_identifier is None
    assert combined.assigned_detection_count == sequence.total_detections

def test_stellar_identification_product_round_trip(tmp_path: Path) -> None:
    sequence, calibration, stars, provider, _ = _synthetic_case()
    config = StellarIdentificationConfig(
        bootstrap_limiting_magnitude=4.5,
        bootstrap_rotation_step_deg=2.0,
        bootstrap_match_radius_px=30.0,
        refinement_radii_px=(20.0, 10.0, 5.0),
        sequence_refinement_radius_px=5.0,
        final_match_radius_px=2.0,
        bright_rescue_radius_px=4.0,
        robust_clip_floor_arcmin=1.0,
    )
    result = CatalogSequenceMatcher(config).fit_and_match(
        sequence,
        calibration,
        stars,
        catalog_ray_provider=provider,
    )
    product = StellarIdentificationProduct(
        product_id="identification-id",
        detection_product_id="detection-id",
        grid_calibration_path=tmp_path / "grid.npz",
        catalog_path=tmp_path / "catalog.npz",
        config=config,
        result=result,
    )
    path = save_stellar_identification_product(
        tmp_path / "stellar_identifications.npz", product
    )

    with np.load(path, allow_pickle=False) as data:
        assert data["format"].item() == "gonet-astrometry-stellar-identifications"
        assert data["version"].item() == 1
        assert data["catalog_identifier"].dtype.kind == "U"

    loaded = load_stellar_identification_product(
        path,
        sequence,
        expected_product_id="identification-id",
        expected_detection_product_id="detection-id",
    )
    assert loaded.product_id == product.product_id
    assert np.allclose(loaded.result.grid_to_enu, result.grid_to_enu)
    assert loaded.result.matched_star_count == result.matched_star_count
    assert loaded.config == config


def test_catalog_sequence_matcher_skips_contaminated_dense_bootstrap_epoch() -> None:
    sequence, calibration, stars, provider, expected_matrix = _synthetic_case()
    first = sequence.epochs[0]
    contaminated_detections = tuple(
        Detection(
            identifier=index,
            x=150.0 + 0.1 * (index % 20),
            y=float((17 * index) % 360),
            flux=50.0,
            signal_to_noise=8.0,
            x_uncertainty=0.2,
            y_uncertainty=0.2,
        )
        for index in range(120)
    )
    contaminated_epoch = DetectionEpoch(
        frame_identifier=first.frame_identifier,
        source_path=first.source_path,
        exposure_midpoint=first.exposure_midpoint,
        location=first.location,
        image_shape=first.image_shape,
        sensor_orientation=first.sensor_orientation,
        catalog=DetectionCatalog(
            first.frame_identifier,
            contaminated_detections,
            "synthetic-contaminated",
        ),
    )
    contaminated_sequence = DetectionSequence(
        (contaminated_epoch, *sequence.epochs[1:])
    )
    config = StellarIdentificationConfig(
        limiting_magnitude=4.5,
        bootstrap_limiting_magnitude=4.5,
        min_catalog_altitude_deg=0.0,
        bootstrap_rotation_step_deg=2.0,
        bootstrap_match_radius_px=30.0,
        refinement_radii_px=(20.0, 10.0, 5.0),
        bootstrap_candidate_frame_count=4,
        bootstrap_validation_frame_count=4,
        bootstrap_validation_radius_px=5.0,
        sequence_refinement_radius_px=5.0,
        final_match_radius_px=2.0,
        bright_rescue_radius_px=4.0,
        robust_clip_floor_arcmin=1.0,
    )

    result = CatalogSequenceMatcher(config).fit_and_match(
        contaminated_sequence,
        calibration,
        stars,
        catalog_ray_provider=provider,
    )

    assert result.bootstrap_frame_identifier != contaminated_epoch.frame_identifier
    assert np.allclose(result.grid_to_enu, expected_matrix, atol=1e-8)
    assert result.matched_star_count == 29
