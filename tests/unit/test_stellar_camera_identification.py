from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from gonet_astrometry.calibration.stellar_camera import StellarCameraCalibration
from gonet_astrometry.catalogs.base import CatalogStar
from gonet_astrometry.geometry.horizon import altaz_to_enu
from gonet_astrometry.models.detection import Detection, DetectionCatalog
from gonet_astrometry.models.frame import ObserverLocation
from gonet_astrometry.products.stellar_camera_identifications import (
    StellarCameraIdentificationProduct,
    load_stellar_camera_identification_product,
    save_stellar_camera_identification_product,
)
from gonet_astrometry.tracking.sequence import DetectionEpoch, DetectionSequence
from gonet_astrometry.tracking.stellar_camera_identification import (
    StellarCameraIdentificationConfig,
    StellarCameraSequenceMatcher,
)


def test_stellar_camera_matcher_identifies_catalog_without_grid(tmp_path: Path) -> None:
    calibration = StellarCameraCalibration(
        image_shape=(400, 400),
        camera_to_enu=np.eye(3),
        center_x_px=200.0,
        center_y_px=200.0,
        radial_c1_px=160.0,
        radial_c3_px=5.0,
        calibrated_theta_max_deg=70.0,
    )
    timestamp = datetime(2026, 8, 30, tzinfo=timezone.utc)
    location = ObserverLocation(41.9, -87.6, 195.0)
    stars = (
        CatalogStar("HR 1", 0.0, 0.0, 1.0),
        CatalogStar("HR 2", 10.0, 10.0, 2.0),
    )
    rays = np.asarray(
        [altaz_to_enu(30.0, 60.0), altaz_to_enu(120.0, 50.0)],
        dtype=float,
    )
    x, y = calibration.enu_to_pixel(rays)
    detections = tuple(
        Detection(
            index,
            float(x[index] + 0.2),
            float(y[index] - 0.1),
            100.0,
            20.0,
            0.1,
            0.1,
        )
        for index in range(2)
    )
    frame_id = "frame"
    sequence = DetectionSequence(
        (
            DetectionEpoch(
                frame_identifier=frame_id,
                source_path=tmp_path / "frame.jpg",
                exposure_midpoint=timestamp,
                location=location,
                image_shape=(400, 400),
                sensor_orientation="native",
                catalog=DetectionCatalog(frame_id, detections, "synthetic"),
            ),
        )
    )

    def provider(catalog_stars, epoch, observer):
        del catalog_stars, epoch, observer
        return rays, np.asarray([60.0, 50.0])

    config = StellarCameraIdentificationConfig(match_radius_px=2.0)
    result = StellarCameraSequenceMatcher(config).match(
        sequence,
        calibration,
        stars,
        catalog_ray_provider=provider,
    )

    assert result.matched_star_count == 2
    assert result.visible_star_count == 2
    assert {item.catalog_identifier for item in result.epochs[0].matched_stars} == {
        "HR 1",
        "HR 2",
    }
    # This intentionally coarse synthetic camera has only about 100 px/rad
    # near the optical axis, so the injected 0.22 px displacement is roughly
    # 7 arcmin.  The real Adler stellar calibration is closer to 1,000 px/rad.
    # Keep this test focused on Grid-free identification rather than imposing
    # the real-camera angular scale on the toy calibration.
    assert result.fit_p90_residual_arcmin < 8.0

    product = StellarCameraIdentificationProduct(
        product_id="identification-id",
        detection_product_id="detection-id",
        stellar_camera_calibration_path=tmp_path / "stellar-camera.npz",
        catalog_path=tmp_path / "catalog.npz",
        config=config,
        result=result,
    )
    path = save_stellar_camera_identification_product(
        tmp_path / "stellar_camera_identifications.npz",
        product,
    )
    with np.load(path, allow_pickle=False) as data:
        assert (
            data["format"].item()
            == "gonet-astrometry-stellar-camera-identifications"
        )
        assert data["version"].item() == 1
    loaded = load_stellar_camera_identification_product(
        path,
        sequence,
        expected_product_id="identification-id",
        expected_detection_product_id="detection-id",
    )
    assert loaded.result.matched_star_count == 2
    assert loaded.config.match_radius_px == 2.0
