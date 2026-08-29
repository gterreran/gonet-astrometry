from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from gonet_astrometry.models.detection import Detection, DetectionCatalog
from gonet_astrometry.models.frame import ObserverLocation
from gonet_astrometry.models.track import StarTrack, TrackDiagnostics, TrackPoint
from gonet_astrometry.products.provenance import (
    multichannel_detection_product_id,
    spherical_tracking_product_id,
    stellar_tracking_product_id,
)
from gonet_astrometry.products.stellar_tracking import (
    StellarTrackingProduct,
    TemporalTrackingProduct,
    load_stellar_tracking_product,
    load_temporal_tracking_product,
    save_stellar_tracking_product,
    save_temporal_tracking_product,
)
from gonet_astrometry.solving.sidereal import SiderealFitConfig
from gonet_astrometry.solving.stellar_tracks import StellarTrackMergeConfig
from gonet_astrometry.tracking.image_plane import ImagePlaneTrackingResult
from gonet_astrometry.tracking.sequence import DetectionEpoch, DetectionSequence
from gonet_astrometry.tracking.spherical import SphericalTrackingConfig


def _sequence() -> DetectionSequence:
    start = datetime(2026, 8, 22, tzinfo=timezone.utc)
    location = ObserverLocation(41.88, -87.63, 180.0)
    epochs = []
    for index in range(3):
        frame_id = f"frame-{index}"
        catalog = DetectionCatalog(
            frame_id,
            (
                Detection(
                    identifier=5,
                    x=10.0 + index,
                    y=20.0 + index,
                    flux=100.0,
                    signal_to_noise=20.0,
                    x_uncertainty=0.2,
                    y_uncertainty=0.2,
                    flags=("channel-support:4",),
                ),
            ),
            "synthetic",
        )
        epochs.append(
            DetectionEpoch(
                frame_identifier=frame_id,
                source_path=Path(frame_id),
                exposure_midpoint=start + timedelta(seconds=30 * index),
                location=location,
                image_shape=(100, 100),
                sensor_orientation="native",
                catalog=catalog,
            )
        )
    return DetectionSequence(tuple(epochs))


def _tracking(sequence: DetectionSequence) -> ImagePlaneTrackingResult:
    track = StarTrack(
        identifier=7,
        points=tuple(
            TrackPoint(epoch.frame_identifier, 5) for epoch in sequence.epochs
        ),
        diagnostics=TrackDiagnostics(
            duration_s=60.0,
            displacement_px=2.0,
            mean_speed_px_per_minute=2.0,
            fit_rms_px=0.1,
            missed_frames=0,
            diagnostic_class="candidate",
            detection_count=3,
            span_epoch_count=3,
            coverage_fraction=1.0,
            median_interval_s=30.0,
            max_interval_s=30.0,
        ),
    )
    return ImagePlaneTrackingResult(sequence, (track,))


def test_temporal_tracking_product_round_trip_is_pickle_free(
    tmp_path: Path,
) -> None:
    sequence = _sequence()
    product = TemporalTrackingProduct(
        product_id="temporal-id",
        detection_product_id="detections-id",
        grid_calibration_path=tmp_path / "grid.npz",
        tracking_config=SphericalTrackingConfig(),
        result=_tracking(sequence),
    )

    path = save_temporal_tracking_product(
        tmp_path / "temporal_tracks.npz",
        product,
    )
    with np.load(path, allow_pickle=False) as data:
        assert str(data["format"]) == "gonet-astrometry-temporal-tracks"

    loaded = load_temporal_tracking_product(
        path,
        sequence,
        expected_product_id="temporal-id",
        expected_detection_product_id="detections-id",
    )

    assert loaded.tracking_config == product.tracking_config
    assert len(loaded.result.tracks) == 1
    assert loaded.result.tracks[0].identifier == 7
    assert len(loaded.result.tracks[0].points) == 3


def test_stellar_tracking_product_round_trip_preserves_fragment_mapping(
    tmp_path: Path,
) -> None:
    sequence = _sequence()
    reference_time = sequence.epochs[1].exposure_midpoint
    product = StellarTrackingProduct(
        product_id="stellar-id",
        temporal_tracking_product_id="temporal-id",
        grid_calibration_path=tmp_path / "grid.npz",
        merge_config=StellarTrackMergeConfig(),
        sidereal_config=SiderealFitConfig(),
        reference_time=reference_time,
        source_fragment_ids=((12, 19),),
        result=_tracking(sequence),
    )

    path = save_stellar_tracking_product(
        tmp_path / "stellar_tracks.npz",
        product,
    )
    with np.load(path, allow_pickle=False) as data:
        assert str(data["format"]) == "gonet-astrometry-stellar-tracks"

    loaded = load_stellar_tracking_product(
        path,
        sequence,
        expected_product_id="stellar-id",
        expected_temporal_tracking_product_id="temporal-id",
    )

    assert loaded.source_fragment_ids == ((12, 19),)
    assert loaded.reference_time == reference_time
    assert loaded.merge_config == product.merge_config
    assert loaded.sidereal_config == product.sidereal_config


def test_new_pipeline_product_ids_follow_their_dependencies(
    tmp_path: Path,
) -> None:
    image = tmp_path / "frame.jpg"
    image.write_bytes(b"image")
    grid = tmp_path / "grid.npz"
    grid.write_bytes(b"grid")

    from gonet_astrometry.detection.config import DetectionConfig
    from gonet_astrometry.detection.multichannel import MultiChannelSEPConfig

    detection_id = multichannel_detection_product_id(
        (image,),
        DetectionConfig(threshold_sigma=3.0),
        MultiChannelSEPConfig(),
        grid,
        250.0,
    )
    temporal_id = spherical_tracking_product_id(
        detection_id,
        grid,
        SphericalTrackingConfig(),
    )
    stellar_id = stellar_tracking_product_id(
        temporal_id,
        StellarTrackMergeConfig(),
        SiderealFitConfig(),
    )

    changed_temporal_id = spherical_tracking_product_id(
        detection_id,
        grid,
        SphericalTrackingConfig(prediction_tolerance_arcmin=6.0),
    )
    changed_stellar_id = stellar_tracking_product_id(
        temporal_id,
        StellarTrackMergeConfig(merge_radius_deg=0.2),
        SiderealFitConfig(),
    )

    assert temporal_id != changed_temporal_id
    assert stellar_id != changed_stellar_id
