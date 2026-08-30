from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from gonet_astrometry.calibration.stellar_camera import (
    StellarCameraCalibration,
    StellarCameraCalibrationConfig,
    StellarCameraCalibrationFit,
    StellarCameraCalibrator,
)
from gonet_astrometry.products.stellar_camera_calibration import (
    StellarCameraCalibrationProduct,
    load_stellar_camera_calibration_product,
    save_stellar_camera_calibration_product,
)


def _calibration() -> StellarCameraCalibration:
    rotation = Rotation.from_euler("xyz", [0.7, -1.1, 57.4], degrees=True).as_matrix()
    return StellarCameraCalibration(
        image_shape=(3040, 4056),
        camera_to_enu=rotation,
        center_x_px=2042.3,
        center_y_px=1518.7,
        radial_c1_px=1538.0,
        radial_c3_px=-85.0,
        calibrated_theta_max_deg=62.0,
    )


def test_stellar_camera_calibration_round_trip() -> None:
    calibration = _calibration()
    theta = np.deg2rad(np.asarray([0.0, 10.0, 30.0, 55.0]))
    phi = np.deg2rad(np.asarray([0.0, 35.0, 120.0, -70.0]))
    camera = np.column_stack(
        (
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta),
        )
    )
    enu = camera @ calibration.camera_to_enu.T

    x, y = calibration.enu_to_pixel(enu)
    recovered = calibration.pixel_to_enu(x, y)

    np.testing.assert_allclose(recovered, enu, atol=2e-10)


def test_stellar_camera_calibration_rejects_extrapolation_by_default() -> None:
    calibration = _calibration()
    theta = np.deg2rad(70.0)
    camera = np.asarray([[np.sin(theta), 0.0, np.cos(theta)]])
    enu = camera @ calibration.camera_to_enu.T

    with pytest.raises(ValueError, match="outside the stellar-calibrated"):
        calibration.enu_to_pixel(enu)

    x, y = calibration.enu_to_pixel(enu, extrapolate=True)
    recovered = calibration.pixel_to_enu(x, y, extrapolate=True)
    np.testing.assert_allclose(recovered, enu, atol=2e-10)


def test_stellar_camera_calibration_rejects_nonmonotonic_calibrated_range() -> None:
    with pytest.raises(ValueError, match="positive and monotonic"):
        StellarCameraCalibration(
            image_shape=(100, 100),
            camera_to_enu=np.eye(3),
            center_x_px=50.0,
            center_y_px=50.0,
            radial_c1_px=100.0,
            radial_c3_px=-100.0,
            calibrated_theta_max_deg=60.0,
        )


def test_stellar_camera_calibration_product_round_trip(tmp_path: Path) -> None:
    calibration = _calibration()
    config = StellarCameraCalibrationConfig()
    fit = StellarCameraCalibrationFit(
        calibration=calibration,
        seed_measurement_count=12832,
        seed_star_count=80,
        trusted_seed_star_count=76,
        rejected_seed_star_ids=("HR 1087", "HR 1149", "HR 1612", "HR 2854"),
        direct_match_count=13213,
        direct_star_count=143,
        direct_match_median_px=0.853,
        direct_match_p90_px=1.959,
        calibration_measurement_count=10942,
        calibration_star_count=73,
        cv_median_arcmin=2.795,
        cv_p90_arcmin=6.417,
        cv_p90_px=1.854,
    )
    product = StellarCameraCalibrationProduct(
        product_id="stellar-calibration-id",
        detection_product_id="detections-id",
        bootstrap_identification_product_id="identifications-id",
        catalog_path=tmp_path / "bright_star_catalog.npz",
        config=config,
        fit=fit,
    )

    path = save_stellar_camera_calibration_product(
        tmp_path / "stellar_camera_calibration.npz", product
    )
    with np.load(path, allow_pickle=False) as data:
        assert data["format"].item() == "gonet-astrometry-stellar-camera-calibration"
        assert data["version"].item() == 1
        assert data["intrinsic_model"].item() == "radial-poly3"
        assert data["rejected_seed_star_ids"].dtype.kind == "U"

    loaded = load_stellar_camera_calibration_product(
        path,
        expected_product_id="stellar-calibration-id",
        expected_detection_product_id="detections-id",
        expected_bootstrap_identification_product_id="identifications-id",
    )
    assert loaded.config == config
    assert loaded.fit.rejected_seed_star_ids == fit.rejected_seed_star_ids
    assert loaded.fit.cv_p90_arcmin == fit.cv_p90_arcmin
    np.testing.assert_allclose(
        loaded.fit.calibration.camera_to_enu, calibration.camera_to_enu
    )


def test_stellar_camera_calibrator_recovers_direct_poly3_geometry() -> None:
    from datetime import datetime, timedelta, timezone

    from gonet_astrometry.catalogs.base import CatalogStar
    from gonet_astrometry.detection.field_mask import FieldMask
    from gonet_astrometry.geometry.horizon import enu_to_altaz
    from gonet_astrometry.models.detection import Detection, DetectionCatalog
    from gonet_astrometry.models.frame import ObserverLocation
    from gonet_astrometry.tracking.catalog_identification import (
        StellarIdentification,
        StellarIdentificationEpoch,
        StellarIdentificationResult,
    )
    from gonet_astrometry.tracking.sequence import DetectionEpoch, DetectionSequence

    image_shape = (220, 240)
    true = StellarCameraCalibration(
        image_shape=image_shape,
        camera_to_enu=np.eye(3),
        center_x_px=121.4,
        center_y_px=108.7,
        radial_c1_px=142.0,
        radial_c3_px=-8.0,
        calibrated_theta_max_deg=70.0,
    )
    stars = tuple(
        CatalogStar(f"HR {100 + index}", 10.0 * index, 10.0, 1.0 + 0.2 * index)
        for index in range(8)
    )
    start = datetime(2026, 8, 30, 3, 0, tzinfo=timezone.utc)
    location = ObserverLocation(41.9, -87.6, 195.0)
    lookup: dict[tuple[str, datetime], np.ndarray] = {}
    epochs = []
    identification_epochs = []
    rng = np.random.default_rng(5)

    for epoch_index in range(6):
        timestamp = start + timedelta(minutes=5 * epoch_index)
        detections = []
        identifications = []
        for star_index, star in enumerate(stars):
            theta = np.deg2rad(12.0 + 4.5 * star_index)
            phi = np.deg2rad(20.0 + 38.0 * star_index + 1.5 * epoch_index)
            camera = np.asarray(
                [
                    np.sin(theta) * np.cos(phi),
                    np.sin(theta) * np.sin(phi),
                    np.cos(theta),
                ]
            )
            enu = camera.copy()
            lookup[(star.identifier, timestamp)] = enu
            x, y = true.enu_to_pixel(enu)
            x_value = float(x) + float(rng.normal(0.0, 0.03))
            y_value = float(y) + float(rng.normal(0.0, 0.03))
            detection_id = star_index + 1
            detections.append(
                Detection(
                    detection_id,
                    x_value,
                    y_value,
                    100.0,
                    30.0,
                    0.05,
                    0.05,
                )
            )
            azimuth, altitude = enu_to_altaz(enu)
            identifications.append(
                StellarIdentification(
                    catalog_identifier=star.identifier,
                    catalog_magnitude=star.magnitude,
                    azimuth_deg=azimuth,
                    altitude_deg=altitude,
                    predicted_x=x_value + 0.5,
                    predicted_y=y_value - 0.5,
                    detection_identifier=detection_id,
                    residual_px=0.7,
                    residual_arcmin=2.0,
                    match_kind="primary",
                )
            )
        detections.append(
            Detection(99, 5.0, 5.0, 5.0, 3.0, 0.2, 0.2)
        )
        frame_id = f"frame-{epoch_index}"
        epochs.append(
            DetectionEpoch(
                frame_identifier=frame_id,
                source_path=Path(f"{frame_id}.jpg"),
                exposure_midpoint=timestamp,
                location=location,
                image_shape=image_shape,
                sensor_orientation="native",
                catalog=DetectionCatalog(frame_id, tuple(detections), "synthetic"),
            )
        )
        identification_epochs.append(
            StellarIdentificationEpoch(frame_id, tuple(identifications))
        )

    sequence = DetectionSequence(tuple(epochs))
    grid_frame_to_enu = Rotation.from_euler("z", 57.4, degrees=True).as_matrix()
    bootstrap = StellarIdentificationResult(
        grid_to_enu=grid_frame_to_enu,
        bootstrap_frame_identifier="frame-0",
        epochs=tuple(identification_epochs),
        fit_pair_count=48,
        fit_median_residual_arcmin=2.0,
        fit_p90_residual_arcmin=3.0,
    )

    def catalog_provider(selected, epoch, observer):
        del observer
        rays = np.asarray([lookup[(star.identifier, epoch)] for star in selected])
        altitude = np.rad2deg(np.arcsin(rays[:, 2]))
        return rays, altitude

    config = StellarCameraCalibrationConfig(
        seed_min_altitude_deg=20.0,
        fit_min_altitude_deg=30.0,
        min_observations_per_star=3,
        trusted_seed_p90_arcmin=5.0,
        initial_match_radius_px=8.0,
        final_match_radius_px=3.0,
        max_refit_residual_px=3.0,
        refinement_iterations=1,
        star_fold_count=3,
        robust_scale_px=1.0,
        max_nfev=250,
    )
    field_mask = FieldMask(
        excluded=np.zeros(image_shape, dtype=np.bool_),
        coordinate_convention="synthetic",
    )
    result = StellarCameraCalibrator(
        config,
        solar_altitude_provider=lambda seq: {
            epoch.frame_identifier: -30.0 for epoch in seq.epochs
        },
        catalog_ray_provider=catalog_provider,
    ).fit(
        sequence,
        bootstrap,
        stars,
        field_mask=field_mask,
        field_mask_keep_margin_px=1.0,
    )

    assert result.direct_match_count == 48
    assert result.direct_star_count == 8
    assert result.calibration_star_count == 8
    assert result.trusted_seed_star_count == 8
    assert result.rejected_seed_star_ids == ()
    assert result.cv_p90_px < 0.2
    assert result.cv_p90_arcmin < 3.0
    assert abs(result.calibration.center_x_px - true.center_x_px) < 0.2
    assert abs(result.calibration.center_y_px - true.center_y_px) < 0.2
    assert abs(result.calibration.radial_c1_px - true.radial_c1_px) < 0.5
    assert abs(result.calibration.radial_c3_px - true.radial_c3_px) < 1.0
    np.testing.assert_allclose(
        result.calibration.camera_to_enu,
        true.camera_to_enu,
        atol=2e-3,
    )
