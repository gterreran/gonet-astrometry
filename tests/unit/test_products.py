from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from gonet_astrometry.detection.config import DetectionConfig
from gonet_astrometry.models.detection import (
    Detection,
    DetectionCatalog,
    DetectionDiagnostics,
)
from gonet_astrometry.models.frame import ObserverLocation
from gonet_astrometry.models.track import StarTrack, TrackDiagnostics, TrackPoint
from gonet_astrometry.products import (
    DetectionProduct,
    ProductMismatchError,
    SiderealProduct,
    TrackingProduct,
    detection_product_id,
    load_detection_product,
    load_sidereal_product,
    load_tracking_product,
    save_detection_product,
    save_sidereal_product,
    save_tracking_product,
    sidereal_product_id,
    tracking_product_id,
)
from gonet_astrometry.solving.sidereal import (
    SiderealFitConfig,
    SiderealRotationSolution,
    SiderealTrackDiagnostics,
)
from gonet_astrometry.tracking.config import TrackingConfig
from gonet_astrometry.tracking.image_plane import ImagePlaneTrackingResult
from gonet_astrometry.tracking.sequence import DetectionEpoch, DetectionSequence


def _sequence(tmp_path: Path) -> DetectionSequence:
    epochs = []
    for index in range(3):
        source = (tmp_path / f"frame-{index}.jpg").resolve()
        source.write_bytes(b"raw" + bytes([index]))
        diagnostics = DetectionDiagnostics(
            peak_value=12.0 + index,
            area_pixels=5 + index,
            semimajor_sigma_px=1.5,
            semiminor_sigma_px=1.0,
            orientation_deg=20.0,
            ellipticity=1.0 / 3.0,
            backend_flags=2 if index == 1 else None,
            sharpness=0.7,
            roundness1=0.1,
            roundness2=-0.1,
        )
        detection = Detection(
            identifier=index,
            x=100.0 + index,
            y=200.0 + index,
            flux=500.0 + index,
            signal_to_noise=20.0,
            x_uncertainty=0.25,
            y_uncertainty=0.3,
            elongation=1.5,
            flags=("extended",) if index == 2 else (),
            diagnostics=diagnostics,
        )
        catalog = DetectionCatalog(str(source), (detection,), "sep")
        epochs.append(
            DetectionEpoch(
                frame_identifier=str(source),
                source_path=source,
                exposure_midpoint=datetime(2026, 8, 18, tzinfo=timezone.utc)
                + timedelta(minutes=index),
                location=ObserverLocation(41.0, -87.0, 180.0),
                image_shape=(3040, 4056),
                sensor_orientation="native sensor order",
                catalog=catalog,
            )
        )
    return DetectionSequence(tuple(epochs))


def test_detection_product_round_trip_is_pickle_free(tmp_path: Path) -> None:
    sequence = _sequence(tmp_path)
    config = DetectionConfig()
    product_id = detection_product_id(
        (epoch.source_path for epoch in sequence.epochs),
        "sep",
        config,
        location_tolerance_m=250.0,
    )
    product = DetectionProduct(
        product_id=product_id,
        algorithm="sep",
        detection_config=config,
        location_tolerance_m=250.0,
        sequence=sequence,
        reference_path=sequence.epochs[0].source_path,
        discovered_files=tuple(epoch.source_path for epoch in sequence.epochs),
        zero_gps_files=(tmp_path / "zero.jpg",),
        location_outlier_files=(tmp_path / "outlier.jpg",),
        metadata_errors=((tmp_path / "bad.jpg", "bad metadata"),),
    )
    path = save_detection_product(tmp_path / "products" / "detections.npz", product)

    with np.load(path, allow_pickle=False) as data:
        assert data["format"].item() == "gonet-astrometry-detections"
        assert data["version"].item() == 1

    loaded = load_detection_product(path, expected_product_id=product_id)

    assert loaded.product_id == product.product_id
    assert loaded.algorithm == "sep"
    assert loaded.detection_config == config
    assert loaded.location_tolerance_m == 250.0
    assert loaded.reference_path == product.reference_path
    assert loaded.discovered_files == product.discovered_files
    assert loaded.skipped_file_count == 3
    assert loaded.metadata_errors == product.metadata_errors
    assert loaded.sequence.epochs == product.sequence.epochs


def test_detection_product_rejects_different_provenance(tmp_path: Path) -> None:
    sequence = _sequence(tmp_path)
    product = DetectionProduct(
        product_id="expected",
        algorithm="sep",
        detection_config=DetectionConfig(),
        location_tolerance_m=250.0,
        sequence=sequence,
        reference_path=sequence.epochs[0].source_path,
        discovered_files=tuple(epoch.source_path for epoch in sequence.epochs),
    )
    path = save_detection_product(tmp_path / "detections.npz", product)

    with pytest.raises(ProductMismatchError, match="does not match"):
        load_detection_product(path, expected_product_id="different")


def test_tracking_product_round_trip_is_pickle_free(tmp_path: Path) -> None:
    sequence = _sequence(tmp_path)
    points = tuple(
        TrackPoint(epoch.frame_identifier, epoch.catalog.detections[0].identifier)
        for epoch in sequence.epochs
    )
    diagnostics = TrackDiagnostics(
        duration_s=120.0,
        displacement_px=4.0,
        mean_speed_px_per_minute=2.0,
        fit_rms_px=0.4,
        missed_frames=0,
        diagnostic_class="candidate",
        detection_count=3,
        span_epoch_count=3,
        coverage_fraction=1.0,
        median_interval_s=60.0,
        max_interval_s=60.0,
    )
    result = ImagePlaneTrackingResult(
        sequence,
        (
            StarTrack(
                identifier=4,
                points=points,
                catalog_identifier="catalog-star",
                quality=0.9,
                diagnostics=diagnostics,
            ),
        ),
    )
    detection_id = "detections-id"
    track_id = tracking_product_id(detection_id, TrackingConfig())
    config = TrackingConfig()
    product = TrackingProduct(track_id, detection_id, config, result)
    path = save_tracking_product(tmp_path / "tracks.npz", product)

    with np.load(path, allow_pickle=False) as data:
        assert data["format"].item() == "gonet-astrometry-tracks"
        assert data["version"].item() == 1

    loaded = load_tracking_product(
        path,
        sequence,
        expected_product_id=track_id,
        expected_detection_product_id=detection_id,
    )

    assert loaded.product_id == track_id
    assert loaded.detection_product_id == detection_id
    assert loaded.tracking_config == config
    assert loaded.result.tracks == result.tracks


def test_detection_product_id_changes_when_input_changes(tmp_path: Path) -> None:
    path = tmp_path / "frame.jpg"
    path.write_bytes(b"first")
    config = DetectionConfig()
    first = detection_product_id(
        [path], "sep", config, location_tolerance_m=250.0
    )
    path.write_bytes(b"second version")
    second = detection_product_id(
        [path], "sep", config, location_tolerance_m=250.0
    )

    assert first != second


def test_sidereal_product_round_trip_is_pickle_free(tmp_path: Path) -> None:
    grid_path = tmp_path / "camera_calibration.npz"
    grid_path.write_bytes(b"grid calibration")
    config = SiderealFitConfig()
    product_id = sidereal_product_id("tracks-id", grid_path, config)
    solution = SiderealRotationSolution(
        axis_grid=np.asarray([0.0, 0.6, 0.8], dtype=float),
        fit_rms_deg=0.02,
        fit_median_deg=0.01,
        fit_p95_deg=0.04,
        fitted_track_count=12,
        fitted_point_count=240,
        track_diagnostics=(
            SiderealTrackDiagnostics(
                track_identifier=4,
                diagnostic_class="sidereal-consistent",
                total_point_count=20,
                valid_point_count=19,
                duration_s=1200.0,
                angular_span_deg=2.5,
                rms_residual_deg=0.02,
                median_residual_deg=0.01,
                max_residual_deg=0.05,
            ),
            SiderealTrackDiagnostics(
                track_identifier=5,
                diagnostic_class="insufficient",
                total_point_count=3,
                valid_point_count=2,
                duration_s=20.0,
                angular_span_deg=0.01,
            ),
        ),
    )
    product = SiderealProduct(
        product_id=product_id,
        tracking_product_id="tracks-id",
        grid_calibration_path=grid_path.resolve(),
        fit_config=config,
        solution=solution,
    )
    path = save_sidereal_product(tmp_path / "sidereal_rotation.npz", product)

    with np.load(path, allow_pickle=False) as data:
        assert data["format"].item() == "gonet-astrometry-sidereal-rotation"
        assert data["version"].item() == 1

    loaded = load_sidereal_product(
        path,
        expected_product_id=product_id,
        expected_tracking_product_id="tracks-id",
    )
    assert loaded.product_id == product_id
    assert loaded.fit_config == config
    assert np.allclose(loaded.solution.axis_grid, solution.axis_grid)
    assert loaded.solution.track_diagnostics == solution.track_diagnostics


def test_sidereal_product_id_changes_when_grid_artifact_changes(tmp_path: Path) -> None:
    grid_path = tmp_path / "camera_calibration.npz"
    grid_path.write_bytes(b"first")
    config = SiderealFitConfig()
    first = sidereal_product_id("tracks-id", grid_path, config)
    grid_path.write_bytes(b"second calibration")
    second = sidereal_product_id("tracks-id", grid_path, config)
    assert first != second
