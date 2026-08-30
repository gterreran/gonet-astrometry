from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from gonet_astrometry.detection.config import DetectionConfig
from gonet_astrometry.detection.multichannel import MultiChannelSEPConfig
from gonet_astrometry.detection.preprocessing import PreparedDetectionImage
from gonet_astrometry.models.detection import Detection, DetectionCatalog
from gonet_astrometry.models.frame import ImageFrame, ImageMetadata, ObserverLocation
from gonet_astrometry.runner import (
    DetectionRunArtifacts,
    RunSummary,
    _prepare_report_reference,
    _resolve_report_reference_path,
    execute_multichannel_detection_run,
    execute_tracking_run,
    preflight_tracking_locations,
    run_cli_workflow,
)
from gonet_astrometry.tracking.config import TrackingConfig


def _prepared(shape: tuple[int, int] = (20, 20)) -> PreparedDetectionImage:
    field = np.ones(shape, dtype=bool)
    field[:2, :] = False
    dynamic = np.zeros(shape, dtype=bool)
    dynamic[8:10, 8:10] = True
    return PreparedDetectionImage(
        data=np.zeros(shape, dtype=float),
        mask=~field | dynamic,
        field_mask=field,
        dynamic_mask=dynamic,
        backgrounds=(1.0, 1.0, 1.0, 1.0),
        noises=(1.0, 1.0, 1.0, 1.0),
    )


def _frame(path: Path, index: int) -> ImageFrame:
    metadata = ImageMetadata(
        exposure_start=datetime(2026, 8, 8, tzinfo=timezone.utc)
        + timedelta(minutes=index),
        exposure_duration_s=10.0,
        location=ObserverLocation(41.0, -87.0, 180.0),
        source_path=path,
        sensor_orientation="native",
    )
    return ImageFrame(np.zeros((20, 20), dtype=float), metadata)


class _Detector:
    name = "synthetic"

    def detect(self, frame_identifier: str, frame: ImageFrame) -> DetectionCatalog:
        raise AssertionError("prepared detection path should be used")

    def detect_prepared(
        self,
        frame_identifier: str,
        prepared: PreparedDetectionImage,
    ) -> DetectionCatalog:
        index = int(Path(frame_identifier).stem.split("-")[-1])
        detection = Detection(
            identifier=0,
            x=5.0 + index,
            y=6.0,
            flux=10.0,
            signal_to_noise=10.0,
            x_uncertainty=0.5,
            y_uncertainty=0.5,
        )
        return DetectionCatalog(frame_identifier, (detection,), self.name)


def test_execute_tracking_run_processes_sequence_without_retaining_all_frames(
    tmp_path: Path, monkeypatch
) -> None:
    paths = tuple(tmp_path / f"frame-{index}.jpg" for index in range(3))
    frames = {
        path.resolve(): _frame(path.resolve(), index)
        for index, path in enumerate(paths)
    }
    calls: list[Path] = []

    def frame_loader(path: Path) -> ImageFrame:
        calls.append(path)
        return frames[path]

    monkeypatch.setattr(
        "gonet_astrometry.runner.prepare_bayer_detection_image",
        lambda frame, config: _prepared(frame.shape),
    )
    monkeypatch.setattr(
        "gonet_astrometry.runner.enrich_detection_catalog",
        lambda catalog, prepared, config: catalog,
    )

    artifacts = execute_tracking_run(
        reversed(paths),
        "synthetic",
        DetectionConfig(),
        TrackingConfig(
            max_speed_px_per_minute=5.0,
            prediction_tolerance_px=2.0,
            min_track_length=3,
        ),
        frame_loader=frame_loader,
        detector_factory=lambda identifier, config: _Detector(),
    )

    assert calls == sorted((path.resolve() for path in paths), key=str)
    assert artifacts.reference_path == paths[0].resolve()
    assert len(artifacts.sequence.epochs) == 3
    assert len(artifacts.tracking_result.tracks) == 1
    assert artifacts.tracking_result.assigned_detection_count == 3


def test_execute_tracking_run_requires_two_images(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least two"):
        execute_tracking_run(
            [tmp_path / "one.jpg"],
            "sep",
            DetectionConfig(),
            TrackingConfig(),
        )


def test_resolve_report_reference_path_accepts_retained_epoch(tmp_path: Path) -> None:
    paths = tuple((tmp_path / f"frame-{index}.jpg").resolve() for index in range(2))
    frames = [_frame(path, index) for index, path in enumerate(paths)]
    catalogs = [DetectionCatalog(str(path), (), "synthetic") for path in paths]
    from gonet_astrometry.tracking.sequence import DetectionEpoch, DetectionSequence

    sequence = DetectionSequence.from_epochs(
        [
            DetectionEpoch.from_frame(str(path), frame, catalog)
            for path, frame, catalog in zip(paths, frames, catalogs, strict=False)
        ]
    )

    selected = _resolve_report_reference_path(sequence, paths[0], paths[1])

    assert selected == paths[1]


def test_resolve_report_reference_path_rejects_unretained_image(tmp_path: Path) -> None:
    paths = tuple((tmp_path / f"frame-{index}.jpg").resolve() for index in range(2))
    frames = [_frame(path, index) for index, path in enumerate(paths)]
    catalogs = [DetectionCatalog(str(path), (), "synthetic") for path in paths]
    from gonet_astrometry.tracking.sequence import DetectionEpoch, DetectionSequence

    sequence = DetectionSequence.from_epochs(
        [
            DetectionEpoch.from_frame(str(path), frame, catalog)
            for path, frame, catalog in zip(paths, frames, catalogs, strict=False)
        ]
    )

    with pytest.raises(ValueError, match="not part of the retained detection sequence"):
        _resolve_report_reference_path(sequence, paths[0], tmp_path / "other.jpg")


def test_prepare_report_reference_uses_selected_catalog(
    tmp_path: Path, monkeypatch
) -> None:
    paths = tuple((tmp_path / f"frame-{index}.jpg").resolve() for index in range(2))
    frames = [_frame(path, index) for index, path in enumerate(paths)]
    catalogs = [
        DetectionCatalog(
            str(path),
            (Detection(index, 5.0, 6.0, 10.0, 10.0, 0.5, 0.5),),
            "synthetic",
        )
        for index, path in enumerate(paths)
    ]
    from gonet_astrometry.tracking.sequence import DetectionEpoch, DetectionSequence

    sequence = DetectionSequence.from_epochs(
        [
            DetectionEpoch.from_frame(str(path), frame, catalog)
            for path, frame, catalog in zip(paths, frames, catalogs, strict=False)
        ]
    )
    monkeypatch.setattr(
        "gonet_astrometry.runner.prepare_bayer_detection_image",
        lambda frame, config: _prepared(frame.shape),
    )

    prepared, catalog = _prepare_report_reference(
        sequence,
        paths[1],
        DetectionConfig(),
        frame_loader=lambda path: frames[paths.index(path)],
    )

    assert prepared.data.shape == frames[1].shape
    assert catalog is catalogs[1]


def test_run_cli_workflow_discovers_writes_and_summarizes(
    tmp_path: Path, monkeypatch
) -> None:
    paths = tuple((tmp_path / f"frame-{index}.jpg").resolve() for index in range(2))
    for path in paths:
        path.touch()

    prepared = _prepared()
    frames = [_frame(path, index) for index, path in enumerate(paths)]
    catalogs = []
    for index, path in enumerate(paths):
        detection = Detection(index, 5.0 + index, 6.0, 10.0, 10.0, 0.5, 0.5)
        catalogs.append(DetectionCatalog(str(path), (detection,), "synthetic"))

    from gonet_astrometry.tracking.image_plane import ImagePlaneTracker
    from gonet_astrometry.tracking.sequence import DetectionEpoch, DetectionSequence

    sequence = DetectionSequence.from_epochs(
        [
            DetectionEpoch.from_frame(str(path), frame, catalog)
            for path, frame, catalog in zip(paths, frames, catalogs, strict=False)
        ]
    )
    tracking = ImagePlaneTracker(
        TrackingConfig(min_track_length=2, prediction_tolerance_px=3.0)
    ).track(sequence)
    detected = DetectionRunArtifacts(
        files=paths,
        algorithm="sep",
        sequence=sequence,
        reference_path=paths[0],
        reference_catalog=catalogs[0],
        reference_prepared=prepared,
    )

    monkeypatch.setattr(
        "gonet_astrometry.runner.execute_detection_run",
        lambda *args, **kwargs: detected,
    )
    monkeypatch.setattr(
        "gonet_astrometry.runner._track_sequence",
        lambda sequence, config: tracking,
    )
    monkeypatch.setattr(
        "gonet_astrometry.runner.load_gonet_file_raw",
        lambda path, parse_metadata=False: object(),
    )
    monkeypatch.setattr(
        "gonet_astrometry.runner.get_raw_channel",
        lambda raw, channel: np.zeros((10, 10), dtype=float),
    )
    output = tmp_path / "reports" / "tracking.pdf"
    monkeypatch.setattr(
        "gonet_astrometry.runner.write_tracking_report_pdf",
        lambda path, **kwargs: path.resolve(),
    )

    summary = run_cli_workflow(
        [tmp_path],
        recursive=True,
        algorithm="sep",
        detection_config=DetectionConfig(),
        tracking_config=TrackingConfig(min_track_length=2),
        channel="green1",
        output_path=output,
        output_dir=tmp_path / "products",
        metadata_loader=lambda path: frames[paths.index(path)].metadata,
    )

    assert summary == RunSummary(
        output_path=output.resolve(),
        file_count=2,
        detection_count=2,
        track_count=1,
        assigned_detection_count=2,
        unassigned_detection_count=0,
        detection_product_path=(tmp_path / "products" / "detections.npz").resolve(),
        tracking_product_path=(tmp_path / "products" / "tracks.npz").resolve(),
    )


def test_run_cli_workflow_requires_two_discovered_images(tmp_path: Path) -> None:
    one = tmp_path / "one.jpg"
    one.touch()
    with pytest.raises(ValueError, match="at least two"):
        run_cli_workflow(
            [one],
            recursive=True,
            algorithm="sep",
            detection_config=DetectionConfig(),
            tracking_config=TrackingConfig(),
            channel="green1",
            output_path=tmp_path / "report.pdf",
        )


def test_preflight_tracking_locations_filters_before_detection(tmp_path: Path) -> None:
    paths = tuple((tmp_path / f"frame-{index}.jpg").resolve() for index in range(5))
    locations = {
        paths[0]: ObserverLocation(41.0, -87.0, 180.0),
        paths[1]: ObserverLocation(41.0001, -87.0001, 181.0),
        paths[2]: ObserverLocation(41.0002, -87.0001, 180.0),
        paths[3]: ObserverLocation(0.0, 0.0, 0.0),
    }

    def loader(path: Path) -> ImageMetadata:
        if path == paths[4]:
            raise ValueError("broken metadata")
        return ImageMetadata(
            exposure_start=datetime(2026, 8, 8, tzinfo=timezone.utc),
            exposure_duration_s=10.0,
            location=locations[path],
            source_path=path,
        )

    result = preflight_tracking_locations(paths, 100.0, metadata_loader=loader)

    assert set(result.files) == set(paths[:3])
    assert tuple(item.path for item in result.selection.zero_gps) == (paths[3],)
    assert result.selection.outliers == ()
    assert result.metadata_errors == ((paths[4], "broken metadata"),)


def test_run_cli_workflow_skips_wrong_location_before_tracking(
    tmp_path: Path, monkeypatch
) -> None:
    paths = tuple((tmp_path / f"frame-{index}.jpg").resolve() for index in range(4))
    for path in paths:
        path.touch()

    metadata = {
        paths[0]: _frame(paths[0], 0).metadata,
        paths[1]: _frame(paths[1], 1).metadata,
        paths[2]: ImageMetadata(
            exposure_start=datetime(2026, 8, 8, tzinfo=timezone.utc),
            exposure_duration_s=10.0,
            location=ObserverLocation(0.0, 0.0, 0.0),
            source_path=paths[2],
        ),
        paths[3]: ImageMetadata(
            exposure_start=datetime(2026, 8, 8, tzinfo=timezone.utc),
            exposure_duration_s=10.0,
            location=ObserverLocation(-30.0, 150.0, 20.0),
            source_path=paths[3],
        ),
    }
    captured: list[tuple[Path, ...]] = []

    prepared = _prepared()
    frames = [_frame(path, index) for index, path in enumerate(paths[:2])]
    catalogs = [
        DetectionCatalog(
            str(path),
            (Detection(index, 5.0 + index, 6.0, 10.0, 10.0, 0.5, 0.5),),
            "synthetic",
        )
        for index, path in enumerate(paths[:2])
    ]
    from gonet_astrometry.tracking.image_plane import ImagePlaneTracker
    from gonet_astrometry.tracking.sequence import DetectionEpoch, DetectionSequence

    sequence = DetectionSequence.from_epochs(
        [
            DetectionEpoch.from_frame(str(path), frame, catalog)
            for path, frame, catalog in zip(paths[:2], frames, catalogs, strict=False)
        ]
    )
    tracking = ImagePlaneTracker(
        TrackingConfig(min_track_length=2, prediction_tolerance_px=3.0)
    ).track(sequence)
    detected = DetectionRunArtifacts(
        files=paths[:2],
        algorithm="sep",
        sequence=sequence,
        reference_path=paths[0],
        reference_catalog=catalogs[0],
        reference_prepared=prepared,
    )

    def fake_execute(selected, *args, **kwargs):
        captured.append(tuple(selected))
        return detected

    monkeypatch.setattr("gonet_astrometry.runner.execute_detection_run", fake_execute)
    monkeypatch.setattr(
        "gonet_astrometry.runner._track_sequence",
        lambda sequence, config: tracking,
    )
    monkeypatch.setattr(
        "gonet_astrometry.runner.load_gonet_file_raw",
        lambda path, parse_metadata=False: object(),
    )
    monkeypatch.setattr(
        "gonet_astrometry.runner.get_raw_channel",
        lambda raw, channel: np.zeros((10, 10), dtype=float),
    )
    monkeypatch.setattr(
        "gonet_astrometry.runner.write_tracking_report_pdf",
        lambda path, **kwargs: path.resolve(),
    )

    summary = run_cli_workflow(
        [tmp_path],
        recursive=True,
        algorithm="sep",
        detection_config=DetectionConfig(),
        tracking_config=TrackingConfig(min_track_length=2, location_tolerance_m=250.0),
        channel="green1",
        output_path=tmp_path / "report.pdf",
        output_dir=tmp_path / "products",
        metadata_loader=lambda path: metadata[path],
    )

    assert captured == [paths[:2]]
    assert summary.file_count == 2
    assert summary.skipped_file_count == 2
    assert summary.zero_gps_count == 1
    assert summary.location_outlier_count == 1
    assert summary.metadata_error_count == 0


def test_run_cli_workflow_reuses_products_before_expensive_steps(
    tmp_path: Path, monkeypatch
) -> None:
    paths = tuple((tmp_path / f"cached-{index}.jpg").resolve() for index in range(2))
    for index, path in enumerate(paths):
        path.write_bytes(b"raw" + bytes([index]))

    prepared = _prepared()
    frames = [_frame(path, index) for index, path in enumerate(paths)]
    catalogs = tuple(
        DetectionCatalog(
            str(path),
            (Detection(index, 5.0 + index, 6.0, 10.0, 10.0, 0.5, 0.5),),
            "sep",
        )
        for index, path in enumerate(paths)
    )
    from gonet_astrometry.tracking.image_plane import ImagePlaneTracker
    from gonet_astrometry.tracking.sequence import DetectionEpoch, DetectionSequence

    sequence = DetectionSequence.from_epochs(
        [
            DetectionEpoch.from_frame(str(path), frame, catalog)
            for path, frame, catalog in zip(paths, frames, catalogs, strict=False)
        ]
    )
    tracking_config = TrackingConfig(
        min_track_length=2,
        prediction_tolerance_px=3.0,
    )
    tracking = ImagePlaneTracker(tracking_config).track(sequence)
    detected = DetectionRunArtifacts(
        files=paths,
        algorithm="sep",
        sequence=sequence,
        reference_path=paths[0],
        reference_catalog=catalogs[0],
        reference_prepared=prepared,
    )
    monkeypatch.setattr(
        "gonet_astrometry.runner.execute_detection_run",
        lambda *args, **kwargs: detected,
    )
    monkeypatch.setattr(
        "gonet_astrometry.runner._track_sequence",
        lambda sequence, config: tracking,
    )
    monkeypatch.setattr(
        "gonet_astrometry.runner.load_gonet_file_raw",
        lambda path, parse_metadata=False: object(),
    )
    monkeypatch.setattr(
        "gonet_astrometry.runner.get_raw_channel",
        lambda raw, channel: np.zeros((10, 10), dtype=float),
    )
    monkeypatch.setattr(
        "gonet_astrometry.runner.write_tracking_report_pdf",
        lambda path, **kwargs: path.resolve(),
    )
    product_dir = tmp_path / "products"
    detection_config = DetectionConfig()

    first = run_cli_workflow(
        [tmp_path],
        recursive=False,
        algorithm="sep",
        detection_config=detection_config,
        tracking_config=tracking_config,
        channel="green1",
        output_path=tmp_path / "first.pdf",
        output_dir=product_dir,
        metadata_loader=lambda path: frames[paths.index(path)].metadata,
    )
    assert first.reused_detection_product is False
    assert first.reused_tracking_product is False

    def unexpected(*args, **kwargs):
        raise AssertionError("expensive stage should have been skipped")

    monkeypatch.setattr("gonet_astrometry.runner.execute_detection_run", unexpected)
    monkeypatch.setattr("gonet_astrometry.runner._track_sequence", unexpected)
    monkeypatch.setattr(
        "gonet_astrometry.runner._prepare_cached_reference",
        lambda product, config: (prepared, catalogs[0]),
    )

    second = run_cli_workflow(
        [tmp_path],
        recursive=False,
        algorithm="sep",
        detection_config=detection_config,
        tracking_config=tracking_config,
        channel="green1",
        output_path=tmp_path / "second.pdf",
        output_dir=product_dir,
        metadata_loader=unexpected,
    )

    assert second.reused_detection_product is True
    assert second.reused_tracking_product is True
    assert second.detection_count == 2
    assert second.track_count == 1

    tracking_calls: list[TrackingConfig] = []

    def retrack(sequence, config):
        tracking_calls.append(config)
        return tracking

    monkeypatch.setattr("gonet_astrometry.runner._track_sequence", retrack)
    changed_tracking = TrackingConfig(
        min_track_length=2,
        prediction_tolerance_px=4.0,
    )
    third = run_cli_workflow(
        [tmp_path],
        recursive=False,
        algorithm="sep",
        detection_config=detection_config,
        tracking_config=changed_tracking,
        channel="green1",
        output_path=tmp_path / "third.pdf",
        output_dir=product_dir,
        metadata_loader=unexpected,
    )

    assert third.reused_detection_product is True
    assert third.reused_tracking_product is False
    assert tracking_calls == [changed_tracking]


def test_run_cli_workflow_uses_and_reuses_grid_stellar_pipeline(
    tmp_path: Path, monkeypatch
) -> None:
    from gonet_astrometry.models.grid import GridCalibration
    from gonet_astrometry.solving.sidereal import (
        SiderealFitConfig,
        SiderealRotationSolution,
        SiderealTrackDiagnostics,
    )
    from gonet_astrometry.solving.stellar_tracks import StellarTrackMergeResult
    from gonet_astrometry.tracking.image_plane import ImagePlaneTracker
    from gonet_astrometry.tracking.sequence import DetectionEpoch, DetectionSequence
    from gonet_astrometry.tracking.spherical import SphericalTrackingConfig

    paths = tuple((tmp_path / f"sidereal-{index}.jpg").resolve() for index in range(2))
    for index, path in enumerate(paths):
        path.write_bytes(b"raw" + bytes([index]))
    grid_path = tmp_path / "camera_calibration.npz"
    grid_path.write_bytes(b"portable grid")

    prepared = _prepared()
    frames = [_frame(path, index) for index, path in enumerate(paths)]
    catalogs = tuple(
        DetectionCatalog(
            str(path),
            (Detection(index, 5.0 + index, 6.0, 10.0, 10.0, 0.5, 0.5),),
            "sep-independent-channels",
        )
        for index, path in enumerate(paths)
    )
    sequence = DetectionSequence.from_epochs(
        [
            DetectionEpoch.from_frame(str(path), frame, catalog)
            for path, frame, catalog in zip(paths, frames, catalogs, strict=False)
        ]
    )
    tracking_config = TrackingConfig(
        min_track_length=2,
        prediction_tolerance_px=3.0,
    )
    tracking = ImagePlaneTracker(tracking_config).track(sequence)
    detected = DetectionRunArtifacts(
        files=paths,
        algorithm="sep-independent-channels",
        sequence=sequence,
        reference_path=paths[0],
        reference_catalog=catalogs[0],
        reference_prepared=prepared,
    )

    class Transform:
        def pixel_to_ray(self, x, y):
            del y
            return np.zeros(np.asarray(x).shape + (3,), dtype=float)

    calibration = GridCalibration(
        transform=Transform(),
        image_shape=(20, 20),
        coordinate_convention="test",
    )
    monkeypatch.setattr(
        "gonet_astrometry.runner.load_grid_calibration",
        lambda path: calibration,
    )

    class FakeMultichannelDetector:
        name = "sep-independent-channels"

        def __init__(self, *args, **kwargs):
            del args, kwargs

        def prepare_report_image(self, frame):
            del frame
            return prepared

    monkeypatch.setattr(
        "gonet_astrometry.runner.IndependentChannelSEPDetector",
        FakeMultichannelDetector,
    )

    detection_calls: list[int] = []

    def fake_detection(*args, **kwargs):
        del args, kwargs
        detection_calls.append(1)
        return detected

    monkeypatch.setattr(
        "gonet_astrometry.runner.execute_multichannel_detection_run",
        fake_detection,
    )

    temporal_calls: list[int] = []

    def fake_temporal(sequence_arg, grid, config):
        del sequence_arg, grid, config
        temporal_calls.append(1)
        return tracking

    monkeypatch.setattr(
        "gonet_astrometry.runner._track_spherical_sequence",
        fake_temporal,
    )

    track_identifier = tracking.tracks[0].identifier
    solution = SiderealRotationSolution(
        axis_grid=np.asarray([0.0, 0.6, 0.8]),
        fit_rms_deg=0.02,
        fit_median_deg=0.01,
        fit_p95_deg=0.04,
        fitted_track_count=1,
        fitted_point_count=2,
        track_diagnostics=(
            SiderealTrackDiagnostics(
                track_identifier=track_identifier,
                diagnostic_class="sidereal-consistent",
                total_point_count=2,
                valid_point_count=2,
                duration_s=60.0,
                angular_span_deg=1.0,
                rms_residual_deg=0.02,
                median_residual_deg=0.01,
                max_residual_deg=0.04,
            ),
        ),
    )

    merge_calls: list[int] = []

    class FakeMerger:
        def __init__(self, *args, **kwargs):
            del args, kwargs

        def fit_and_merge(self, tracking_result, grid):
            del grid
            merge_calls.append(1)
            return StellarTrackMergeResult(
                tracking=tracking_result,
                source_fragment_ids=((track_identifier,),),
                reference_time=sequence.epochs[0].exposure_midpoint,
                preliminary_solution=solution,
                final_solution=solution,
            )

    monkeypatch.setattr("gonet_astrometry.runner.SiderealTrackMerger", FakeMerger)
    monkeypatch.setattr(
        "gonet_astrometry.runner.SiderealAxisFitter.fit_candidate_count",
        lambda self, tracking_result, grid: 3,
    )
    monkeypatch.setattr(
        "gonet_astrometry.runner.load_gonet_file_raw",
        lambda path, parse_metadata=False: object(),
    )
    monkeypatch.setattr(
        "gonet_astrometry.runner.get_raw_channel",
        lambda raw, channel: np.zeros((10, 10), dtype=float),
    )
    monkeypatch.setattr(
        "gonet_astrometry.runner.write_tracking_report_pdf",
        lambda path, **kwargs: path.resolve(),
    )

    product_dir = tmp_path / "products"
    fit_config = SiderealFitConfig(
        min_track_points=3,
        min_track_duration_minutes=1.0,
    )
    spherical_config = SphericalTrackingConfig(min_track_length=2)

    first = run_cli_workflow(
        [tmp_path],
        recursive=False,
        algorithm="sep",
        detection_config=DetectionConfig(),
        tracking_config=tracking_config,
        channel="green1",
        output_path=tmp_path / "first.pdf",
        output_dir=product_dir,
        grid_calibration_path=grid_path,
        spherical_tracking_config=spherical_config,
        sidereal_config=fit_config,
        metadata_loader=lambda path: frames[paths.index(path)].metadata,
    )

    assert first.reused_detection_product is False
    assert first.reused_temporal_tracking_product is False
    assert first.reused_stellar_tracking_product is False
    assert first.reused_sidereal_product is False
    assert first.temporal_tracking_product_path == (
        product_dir / "temporal_tracks.npz"
    ).resolve()
    assert first.stellar_tracking_product_path == (
        product_dir / "stellar_tracks.npz"
    ).resolve()
    assert first.tracking_product_path == first.stellar_tracking_product_path
    assert detection_calls == [1]
    assert temporal_calls == [1]
    assert merge_calls == [1]

    monkeypatch.setattr(
        "gonet_astrometry.runner._prepare_multichannel_report_reference",
        lambda sequence_arg, reference_path, detector: (prepared, catalogs[0]),
    )

    second = run_cli_workflow(
        [tmp_path],
        recursive=False,
        algorithm="sep",
        detection_config=DetectionConfig(),
        tracking_config=tracking_config,
        channel="green1",
        output_path=tmp_path / "second.pdf",
        output_dir=product_dir,
        grid_calibration_path=grid_path,
        spherical_tracking_config=spherical_config,
        sidereal_config=fit_config,
        metadata_loader=lambda path: frames[paths.index(path)].metadata,
    )

    assert second.reused_detection_product is True
    assert second.reused_temporal_tracking_product is True
    assert second.reused_stellar_tracking_product is True
    assert second.reused_tracking_product is True
    assert second.reused_sidereal_product is True
    assert detection_calls == [1]
    assert temporal_calls == [1]
    assert merge_calls == [1]

    third = run_cli_workflow(
        [tmp_path],
        recursive=False,
        algorithm="sep",
        detection_config=DetectionConfig(),
        tracking_config=tracking_config,
        channel="green1",
        output_path=tmp_path / "third.pdf",
        output_dir=product_dir,
        grid_calibration_path=grid_path,
        spherical_tracking_config=spherical_config,
        sidereal_config=fit_config,
        overwrite_solution=True,
        metadata_loader=lambda path: frames[paths.index(path)].metadata,
    )

    assert third.reused_detection_product is True
    assert third.reused_temporal_tracking_product is True
    assert third.reused_stellar_tracking_product is False
    assert third.reused_tracking_product is False
    assert third.reused_sidereal_product is False
    assert detection_calls == [1]
    assert temporal_calls == [1]
    assert merge_calls == [1, 1]


def test_grid_run_allows_single_image_and_skips_tracking(
    tmp_path: Path, monkeypatch
) -> None:
    from gonet_astrometry.models.grid import GridCalibration
    from gonet_astrometry.tracking.sequence import DetectionEpoch, DetectionSequence

    path = (tmp_path / "single.jpg").resolve()
    path.write_bytes(b"raw")
    grid_path = tmp_path / "camera_calibration.npz"
    grid_path.write_bytes(b"portable grid")

    frame = _frame(path, 0)
    catalog = DetectionCatalog(
        str(path),
        (Detection(0, 5.0, 6.0, 10.0, 10.0, 0.5, 0.5),),
        "sep-independent-channels",
    )
    sequence = DetectionSequence.from_epochs(
        [DetectionEpoch.from_frame(str(path), frame, catalog)]
    )
    detected = DetectionRunArtifacts(
        files=(path,),
        algorithm="sep-independent-channels",
        sequence=sequence,
        reference_path=path,
        reference_catalog=catalog,
        reference_prepared=_prepared(),
    )

    class Transform:
        def pixel_to_ray(self, x, y):
            del y
            return np.zeros(np.asarray(x).shape + (3,), dtype=float)

    calibration = GridCalibration(
        transform=Transform(),
        image_shape=(20, 20),
        coordinate_convention="test",
    )
    monkeypatch.setattr(
        "gonet_astrometry.runner.load_grid_calibration",
        lambda path: calibration,
    )

    class FakeMultichannelDetector:
        name = "sep-independent-channels"

        def __init__(self, *args, **kwargs):
            del args, kwargs

        def prepare_report_image(self, frame):
            del frame
            return _prepared()

    monkeypatch.setattr(
        "gonet_astrometry.runner.IndependentChannelSEPDetector",
        FakeMultichannelDetector,
    )
    monkeypatch.setattr(
        "gonet_astrometry.runner.execute_multichannel_detection_run",
        lambda *args, **kwargs: detected,
    )
    monkeypatch.setattr(
        "gonet_astrometry.runner._track_spherical_sequence",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("single-image run must not start temporal tracking")
        ),
    )
    monkeypatch.setattr(
        "gonet_astrometry.runner.load_gonet_file_raw",
        lambda path, parse_metadata=False: object(),
    )
    monkeypatch.setattr(
        "gonet_astrometry.runner.get_raw_channel",
        lambda raw, channel: np.zeros((10, 10), dtype=float),
    )
    monkeypatch.setattr(
        "gonet_astrometry.runner.write_tracking_report_pdf",
        lambda path, **kwargs: path.resolve(),
    )

    product_dir = tmp_path / "products"
    summary = run_cli_workflow(
        [path],
        recursive=False,
        algorithm="sep",
        detection_config=DetectionConfig(),
        tracking_config=TrackingConfig(),
        channel="green1",
        output_path=tmp_path / "single.pdf",
        output_dir=product_dir,
        grid_calibration_path=grid_path,
        metadata_loader=lambda source: frame.metadata,
    )

    assert summary.file_count == 1
    assert summary.detection_count == 1
    assert summary.track_count == 0
    assert summary.assigned_detection_count == 0
    assert summary.unassigned_detection_count == 1
    assert summary.detection_product_path == (product_dir / "detections.npz").resolve()
    assert summary.temporal_tracking_product_path is None
    assert summary.stellar_tracking_product_path is None
    assert summary.sidereal_product_path is None


def test_grid_run_short_sequence_stops_after_temporal_tracking(
    tmp_path: Path, monkeypatch
) -> None:
    from gonet_astrometry.models.grid import GridCalibration
    from gonet_astrometry.tracking.image_plane import ImagePlaneTracker
    from gonet_astrometry.tracking.sequence import DetectionEpoch, DetectionSequence
    from gonet_astrometry.tracking.spherical import SphericalTrackingConfig

    paths = tuple((tmp_path / f"short-{index}.jpg").resolve() for index in range(2))
    for index, path in enumerate(paths):
        path.write_bytes(b"raw" + bytes([index]))
    grid_path = tmp_path / "camera_calibration.npz"
    grid_path.write_bytes(b"portable grid")

    frames = [_frame(path, index) for index, path in enumerate(paths)]
    catalogs = tuple(
        DetectionCatalog(
            str(path),
            (Detection(index, 5.0 + index, 6.0, 10.0, 10.0, 0.5, 0.5),),
            "sep-independent-channels",
        )
        for index, path in enumerate(paths)
    )
    sequence = DetectionSequence.from_epochs(
        [
            DetectionEpoch.from_frame(str(path), frame, catalog)
            for path, frame, catalog in zip(paths, frames, catalogs, strict=True)
        ]
    )
    temporal = ImagePlaneTracker(
        TrackingConfig(min_track_length=2, prediction_tolerance_px=3.0)
    ).track(sequence)
    detected = DetectionRunArtifacts(
        files=paths,
        algorithm="sep-independent-channels",
        sequence=sequence,
        reference_path=paths[0],
        reference_catalog=catalogs[0],
        reference_prepared=_prepared(),
    )

    class Transform:
        def pixel_to_ray(self, x, y):
            del y
            rays = np.zeros(np.asarray(x).shape + (3,), dtype=float)
            rays[..., 2] = 1.0
            return rays

    calibration = GridCalibration(
        transform=Transform(),
        image_shape=(20, 20),
        coordinate_convention="test",
    )
    monkeypatch.setattr(
        "gonet_astrometry.runner.load_grid_calibration",
        lambda path: calibration,
    )

    class FakeMultichannelDetector:
        name = "sep-independent-channels"

        def __init__(self, *args, **kwargs):
            del args, kwargs

        def prepare_report_image(self, frame):
            del frame
            return _prepared()

    monkeypatch.setattr(
        "gonet_astrometry.runner.IndependentChannelSEPDetector",
        FakeMultichannelDetector,
    )
    monkeypatch.setattr(
        "gonet_astrometry.runner.execute_multichannel_detection_run",
        lambda *args, **kwargs: detected,
    )
    monkeypatch.setattr(
        "gonet_astrometry.runner._track_spherical_sequence",
        lambda *args, **kwargs: temporal,
    )
    monkeypatch.setattr(
        "gonet_astrometry.runner.SiderealAxisFitter.fit_candidate_count",
        lambda self, tracking_result, grid: 0,
    )
    monkeypatch.setattr(
        "gonet_astrometry.runner.SiderealTrackMerger",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("short sequence must not start stellar optimization")
        ),
    )
    monkeypatch.setattr(
        "gonet_astrometry.runner.load_gonet_file_raw",
        lambda path, parse_metadata=False: object(),
    )
    monkeypatch.setattr(
        "gonet_astrometry.runner.get_raw_channel",
        lambda raw, channel: np.zeros((10, 10), dtype=float),
    )
    monkeypatch.setattr(
        "gonet_astrometry.runner.write_tracking_report_pdf",
        lambda path, **kwargs: path.resolve(),
    )

    product_dir = tmp_path / "products"
    summary = run_cli_workflow(
        [tmp_path],
        recursive=False,
        algorithm="sep",
        detection_config=DetectionConfig(),
        tracking_config=TrackingConfig(min_track_length=2),
        channel="green1",
        output_path=tmp_path / "short.pdf",
        output_dir=product_dir,
        grid_calibration_path=grid_path,
        spherical_tracking_config=SphericalTrackingConfig(min_track_length=2),
        metadata_loader=lambda path: frames[paths.index(path)].metadata,
    )

    assert summary.temporal_tracking_product_path == (
        product_dir / "temporal_tracks.npz"
    ).resolve()
    assert summary.tracking_product_path == summary.temporal_tracking_product_path
    assert summary.stellar_tracking_product_path is None
    assert summary.sidereal_product_path is None
    assert summary.track_count == len(temporal.tracks)


def test_multichannel_detection_parallel_matches_serial(
    tmp_path: Path, monkeypatch
) -> None:
    from concurrent.futures import Future

    from gonet_astrometry.models.grid import GridCalibration

    paths = tuple((tmp_path / f"frame-{index}.jpg").resolve() for index in range(4))
    frames = {path: _frame(path, index) for index, path in enumerate(paths)}
    grid_path = (tmp_path / "camera_calibration.npz").resolve()
    grid_path.write_bytes(b"portable-grid")
    calibration = GridCalibration(
        transform=object(),
        image_shape=(20, 20),
        coordinate_convention="test",
        source=str(grid_path),
    )

    class FakeMultichannelDetector:
        name = "sep-independent-channels"

        def __init__(self, *args, **kwargs):
            del args, kwargs

        def detect(self, frame_identifier: str, frame: ImageFrame) -> DetectionCatalog:
            del frame
            index = int(Path(frame_identifier).stem.split("-")[-1])
            detection = Detection(
                identifier=1,
                x=5.0 + index,
                y=6.0,
                flux=10.0,
                signal_to_noise=10.0,
                x_uncertainty=0.5,
                y_uncertainty=0.5,
            )
            return DetectionCatalog(
                frame_identifier,
                (detection,),
                self.name,
            )

        def prepare_report_image(self, frame: ImageFrame) -> PreparedDetectionImage:
            return _prepared(frame.shape)

    class InlineProcessPoolExecutor:
        def __init__(self, *, max_workers, initializer, initargs):
            assert max_workers == 3
            initializer(*initargs)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            del exc_type, exc, traceback
            return False

        def submit(self, function, *args):
            future = Future()
            try:
                future.set_result(function(*args))
            except Exception as exc:  # pragma: no cover - mirrors executor contract
                future.set_exception(exc)
            return future

    monkeypatch.setattr(
        "gonet_astrometry.runner.IndependentChannelSEPDetector",
        FakeMultichannelDetector,
    )
    monkeypatch.setattr(
        "gonet_astrometry.runner.load_grid_calibration",
        lambda path: calibration,
    )
    monkeypatch.setattr(
        "gonet_astrometry.runner.load_gonet_image",
        lambda path: frames[Path(path).resolve()],
    )
    monkeypatch.setattr(
        "gonet_astrometry.runner.ProcessPoolExecutor",
        InlineProcessPoolExecutor,
    )

    serial_detector = FakeMultichannelDetector()
    serial = execute_multichannel_detection_run(
        reversed(paths),
        DetectionConfig(),
        MultiChannelSEPConfig(),
        calibration,
        location_tolerance_m=250.0,
        workers=1,
        detector=serial_detector,
        frame_loader=lambda path: frames[Path(path).resolve()],
    )
    parallel_detector = FakeMultichannelDetector()
    parallel = execute_multichannel_detection_run(
        reversed(paths),
        DetectionConfig(),
        MultiChannelSEPConfig(),
        calibration,
        location_tolerance_m=250.0,
        workers=3,
        grid_calibration_path=grid_path,
        detector=parallel_detector,
        frame_loader=lambda path: frames[Path(path).resolve()],
    )

    assert parallel.files == serial.files
    assert parallel.sequence == serial.sequence
    assert parallel.reference_path == serial.reference_path
    assert parallel.reference_catalog == serial.reference_catalog


def test_multichannel_detection_rejects_nonpositive_workers(tmp_path: Path) -> None:
    from gonet_astrometry.models.grid import GridCalibration

    path = (tmp_path / "frame-0.jpg").resolve()
    calibration = GridCalibration(
        transform=object(),
        image_shape=(20, 20),
        coordinate_convention="test",
    )

    with pytest.raises(ValueError, match="workers must be at least one"):
        execute_multichannel_detection_run(
            [path],
            DetectionConfig(),
            MultiChannelSEPConfig(),
            calibration,
            location_tolerance_m=250.0,
            workers=0,
        )


def test_grid_detection_only_can_write_and_reuse_stellar_identifications(
    tmp_path: Path, monkeypatch
) -> None:
    from types import SimpleNamespace

    from gonet_astrometry.calibration.stellar_camera import (
        StellarCameraCalibration,
        StellarCameraCalibrationFit,
    )
    from gonet_astrometry.models.grid import GridCalibration
    from gonet_astrometry.tracking.catalog_identification import (
        StellarIdentification,
        StellarIdentificationEpoch,
        StellarIdentificationResult,
    )
    from gonet_astrometry.tracking.sequence import DetectionEpoch, DetectionSequence

    path = (tmp_path / "single-identification.jpg").resolve()
    path.write_bytes(b"raw")
    grid_path = tmp_path / "camera_calibration.npz"
    grid_path.write_bytes(b"portable grid")
    catalog_path = tmp_path / "bright_star_catalog.npz"
    catalog_path.write_bytes(b"catalog fingerprint")

    frame = _frame(path, 0)
    catalog = DetectionCatalog(
        str(path),
        (Detection(0, 5.0, 6.0, 10.0, 10.0, 0.5, 0.5),),
        "sep-independent-channels",
    )
    sequence = DetectionSequence.from_epochs(
        [DetectionEpoch.from_frame(str(path), frame, catalog)]
    )
    detected = DetectionRunArtifacts(
        files=(path,),
        algorithm="sep-independent-channels",
        sequence=sequence,
        reference_path=path,
        reference_catalog=catalog,
        reference_prepared=_prepared(),
    )

    class Transform:
        def pixel_to_ray(self, x, y):
            del y
            rays = np.zeros(np.asarray(x).shape + (3,), dtype=float)
            rays[..., 2] = 1.0
            return rays

    calibration = GridCalibration(
        transform=Transform(),
        image_shape=(20, 20),
        coordinate_convention="test",
    )
    monkeypatch.setattr(
        "gonet_astrometry.runner.load_grid_calibration", lambda path: calibration
    )

    class FakeMultichannelDetector:
        name = "sep-independent-channels"

        def __init__(self, *args, **kwargs):
            del args, kwargs

        def prepare_report_image(self, frame):
            del frame
            return _prepared()

    monkeypatch.setattr(
        "gonet_astrometry.runner.IndependentChannelSEPDetector",
        FakeMultichannelDetector,
    )
    detection_calls: list[int] = []

    def fake_detection(*args, **kwargs):
        del args, kwargs
        detection_calls.append(1)
        return detected

    monkeypatch.setattr(
        "gonet_astrometry.runner.execute_multichannel_detection_run",
        fake_detection,
    )
    monkeypatch.setattr(
        "gonet_astrometry.runner.load_or_fetch_bright_star_catalog",
        lambda cache_path: SimpleNamespace(
            stars=(object(),),
            query_bright_stars=lambda epoch, limiting_magnitude: (object(),),
        ),
    )

    identification_result = StellarIdentificationResult(
        grid_to_enu=np.eye(3),
        bootstrap_frame_identifier=str(path),
        epochs=(
            StellarIdentificationEpoch(
                frame_identifier=str(path),
                visible_stars=(
                    StellarIdentification(
                        catalog_identifier="HR 1",
                        catalog_magnitude=1.0,
                        azimuth_deg=0.0,
                        altitude_deg=80.0,
                        predicted_x=5.0,
                        predicted_y=6.0,
                        detection_identifier=0,
                        residual_px=0.1,
                        residual_arcmin=0.2,
                        match_kind="primary",
                    ),
                ),
            ),
        ),
        fit_pair_count=1,
        fit_median_residual_arcmin=0.2,
        fit_p90_residual_arcmin=0.2,
    )
    matcher_calls: list[int] = []

    class FakeMatcher:
        def __init__(self, config):
            del config

        def fit_and_match(self, *args, **kwargs):
            del args, kwargs
            matcher_calls.append(1)
            return identification_result

    monkeypatch.setattr("gonet_astrometry.runner.CatalogSequenceMatcher", FakeMatcher)

    stellar_calibration_fit = StellarCameraCalibrationFit(
        calibration=StellarCameraCalibration(
            image_shape=(20, 20),
            camera_to_enu=np.eye(3),
            center_x_px=10.0,
            center_y_px=10.0,
            radial_c1_px=20.0,
            radial_c3_px=-1.0,
            calibrated_theta_max_deg=60.0,
        ),
        seed_measurement_count=10,
        seed_star_count=4,
        trusted_seed_star_count=4,
        rejected_seed_star_ids=(),
        direct_match_count=9,
        direct_star_count=4,
        direct_match_median_px=0.5,
        direct_match_p90_px=1.0,
        calibration_measurement_count=8,
        calibration_star_count=4,
        cv_median_arcmin=2.0,
        cv_p90_arcmin=5.0,
        cv_p90_px=1.5,
    )
    calibrator_calls: list[int] = []

    class FakeStellarCameraCalibrator:
        def __init__(self, config):
            del config

        def fit(self, *args, **kwargs):
            del args, kwargs
            calibrator_calls.append(1)
            return stellar_calibration_fit

    monkeypatch.setattr(
        "gonet_astrometry.runner.StellarCameraCalibrator",
        FakeStellarCameraCalibrator,
    )
    monkeypatch.setattr(
        "gonet_astrometry.runner.load_gonet_file_raw",
        lambda path, parse_metadata=False: object(),
    )
    monkeypatch.setattr(
        "gonet_astrometry.runner.get_raw_channel",
        lambda raw, channel: np.zeros((10, 10), dtype=float),
    )
    monkeypatch.setattr(
        "gonet_astrometry.runner.write_tracking_report_pdf",
        lambda path, **kwargs: path.resolve(),
    )

    product_dir = tmp_path / "products"
    kwargs = dict(
        inputs=[path],
        recursive=False,
        algorithm="sep",
        detection_config=DetectionConfig(),
        tracking_config=TrackingConfig(),
        channel="green1",
        output_dir=product_dir,
        grid_calibration_path=grid_path,
        catalog_cache_path=catalog_path,
        identify_stars=True,
        fit_stellar_calibration=True,
        detection_only=True,
        metadata_loader=lambda source: frame.metadata,
    )

    first = run_cli_workflow(output_path=tmp_path / "first-ident.pdf", **kwargs)
    assert first.reused_stellar_identification_product is False
    assert first.stellar_identification_product_path == (
        product_dir / "stellar_identifications.npz"
    ).resolve()
    assert first.reused_stellar_camera_calibration_product is False
    assert first.stellar_camera_calibration_product_path == (
        product_dir / "stellar_camera_calibration.npz"
    ).resolve()
    assert detection_calls == [1]
    assert matcher_calls == [1]
    assert calibrator_calls == [1]

    monkeypatch.setattr(
        "gonet_astrometry.runner._prepare_multichannel_report_reference",
        lambda sequence_arg, reference_path, detector: (_prepared(), catalog),
    )
    second = run_cli_workflow(output_path=tmp_path / "second-ident.pdf", **kwargs)
    assert second.reused_detection_product is True
    assert second.reused_stellar_identification_product is True
    assert second.reused_stellar_camera_calibration_product is True
    assert detection_calls == [1]
    assert matcher_calls == [1]
    assert calibrator_calls == [1]


def test_grid_hybrid_tracking_groups_catalog_labels_and_tracks_only_unmatched(
    tmp_path: Path, monkeypatch
) -> None:
    from types import SimpleNamespace

    from gonet_astrometry.models.grid import GridCalibration
    from gonet_astrometry.models.track import StarTrack, TrackPoint
    from gonet_astrometry.tracking.catalog_identification import (
        StellarIdentification,
        StellarIdentificationEpoch,
        StellarIdentificationResult,
    )
    from gonet_astrometry.tracking.image_plane import ImagePlaneTrackingResult
    from gonet_astrometry.tracking.sequence import DetectionEpoch, DetectionSequence
    from gonet_astrometry.tracking.spherical import SphericalTrackingConfig

    paths = tuple((tmp_path / f"hybrid-{index}.jpg").resolve() for index in range(3))
    for index, path in enumerate(paths):
        path.write_bytes(b"raw" + bytes([index]))
    grid_path = tmp_path / "camera_calibration.npz"
    grid_path.write_bytes(b"portable grid")
    catalog_path = tmp_path / "bright_star_catalog.npz"
    catalog_path.write_bytes(b"catalog fingerprint")

    frames = [_frame(path, index) for index, path in enumerate(paths)]
    catalogs = tuple(
        DetectionCatalog(
            str(path),
            (
                Detection(0, 5.0 + index, 6.0, 10.0, 10.0, 0.5, 0.5),
                Detection(1, 12.0 + index, 14.0, 8.0, 8.0, 0.5, 0.5),
            ),
            "sep-independent-channels",
        )
        for index, path in enumerate(paths)
    )
    sequence = DetectionSequence.from_epochs(
        [
            DetectionEpoch.from_frame(str(path), frame, catalog)
            for path, frame, catalog in zip(paths, frames, catalogs, strict=True)
        ]
    )
    detected = DetectionRunArtifacts(
        files=paths,
        algorithm="sep-independent-channels",
        sequence=sequence,
        reference_path=paths[0],
        reference_catalog=catalogs[0],
        reference_prepared=_prepared(),
    )

    class Transform:
        def pixel_to_ray(self, x, y):
            del y
            rays = np.zeros(np.asarray(x).shape + (3,), dtype=float)
            rays[..., 2] = 1.0
            return rays

    calibration = GridCalibration(
        transform=Transform(),
        image_shape=(20, 20),
        coordinate_convention="test",
    )
    monkeypatch.setattr(
        "gonet_astrometry.runner.load_grid_calibration", lambda path: calibration
    )

    class FakeMultichannelDetector:
        name = "sep-independent-channels"

        def __init__(self, *args, **kwargs):
            del args, kwargs

        def prepare_report_image(self, frame):
            del frame
            return _prepared()

    monkeypatch.setattr(
        "gonet_astrometry.runner.IndependentChannelSEPDetector",
        FakeMultichannelDetector,
    )
    detection_calls: list[int] = []

    def fake_detection(*args, **kwargs):
        del args, kwargs
        detection_calls.append(1)
        return detected

    monkeypatch.setattr(
        "gonet_astrometry.runner.execute_multichannel_detection_run",
        fake_detection,
    )
    monkeypatch.setattr(
        "gonet_astrometry.runner.load_or_fetch_bright_star_catalog",
        lambda cache_path: SimpleNamespace(
            stars=(object(),),
            query_bright_stars=lambda epoch, limiting_magnitude: (object(),),
        ),
    )

    identification_result = StellarIdentificationResult(
        grid_to_enu=np.eye(3),
        bootstrap_frame_identifier=str(paths[0]),
        epochs=tuple(
            StellarIdentificationEpoch(
                frame_identifier=str(path),
                visible_stars=(
                    StellarIdentification(
                        catalog_identifier="HR 1",
                        catalog_magnitude=1.0,
                        azimuth_deg=0.0,
                        altitude_deg=80.0,
                        predicted_x=5.0 + index,
                        predicted_y=6.0,
                        detection_identifier=0,
                        residual_px=0.1,
                        residual_arcmin=0.2,
                        match_kind="primary",
                    ),
                ),
            )
            for index, path in enumerate(paths)
        ),
        fit_pair_count=3,
        fit_median_residual_arcmin=0.2,
        fit_p90_residual_arcmin=0.2,
    )
    matcher_calls: list[int] = []

    class FakeMatcher:
        def __init__(self, config):
            del config

        def fit_and_match(self, *args, **kwargs):
            del args, kwargs
            matcher_calls.append(1)
            return identification_result

    monkeypatch.setattr("gonet_astrometry.runner.CatalogSequenceMatcher", FakeMatcher)

    fallback_calls: list[int] = []

    def fake_fallback(unmatched_sequence, grid, config):
        del grid, config
        fallback_calls.append(unmatched_sequence.total_detections)
        assert unmatched_sequence.total_detections == 3
        assert all(
            [detection.identifier for detection in epoch.catalog.detections] == [1]
            for epoch in unmatched_sequence.epochs
        )
        return ImagePlaneTrackingResult(
            unmatched_sequence,
            (
                StarTrack(
                    identifier=0,
                    points=tuple(
                        TrackPoint(epoch.frame_identifier, 1)
                        for epoch in unmatched_sequence.epochs
                    ),
                ),
            ),
        )

    monkeypatch.setattr(
        "gonet_astrometry.runner._track_spherical_sequence",
        fake_fallback,
    )
    monkeypatch.setattr(
        "gonet_astrometry.runner.load_gonet_file_raw",
        lambda path, parse_metadata=False: object(),
    )
    monkeypatch.setattr(
        "gonet_astrometry.runner.get_raw_channel",
        lambda raw, channel: np.zeros((10, 10), dtype=float),
    )
    monkeypatch.setattr(
        "gonet_astrometry.runner.write_tracking_report_pdf",
        lambda path, **kwargs: path.resolve(),
    )

    product_dir = tmp_path / "products"
    kwargs = dict(
        inputs=[tmp_path],
        recursive=False,
        algorithm="sep",
        detection_config=DetectionConfig(),
        tracking_config=TrackingConfig(),
        tracking_mode="hybrid",
        channel="green1",
        output_dir=product_dir,
        grid_calibration_path=grid_path,
        catalog_cache_path=catalog_path,
        spherical_tracking_config=SphericalTrackingConfig(min_track_length=2),
        metadata_loader=lambda source: frames[paths.index(source)].metadata,
    )

    first = run_cli_workflow(output_path=tmp_path / "hybrid-first.pdf", **kwargs)
    assert first.tracking_mode == "hybrid"
    assert first.track_count == 2
    assert first.catalog_track_count == 1
    assert first.fallback_track_count == 1
    assert first.assigned_detection_count == 6
    assert first.unassigned_detection_count == 0
    assert first.stellar_identification_product_path == (
        product_dir / "stellar_identifications.npz"
    ).resolve()
    assert first.fallback_tracking_product_path == (
        product_dir / "fallback_temporal_tracks.npz"
    ).resolve()
    assert first.reused_fallback_tracking_product is False
    assert detection_calls == [1]
    assert matcher_calls == [1]
    assert fallback_calls == [3]

    monkeypatch.setattr(
        "gonet_astrometry.runner._prepare_multichannel_report_reference",
        lambda sequence_arg, reference_path, detector: (_prepared(), catalogs[0]),
    )
    second = run_cli_workflow(output_path=tmp_path / "hybrid-second.pdf", **kwargs)
    assert second.reused_detection_product is True
    assert second.reused_stellar_identification_product is True
    assert second.reused_fallback_tracking_product is True
    assert second.track_count == 2
    assert detection_calls == [1]
    assert matcher_calls == [1]
    assert fallback_calls == [3]


def test_stellar_calibration_run_identifies_without_grid(
    tmp_path: Path, monkeypatch
) -> None:
    from types import SimpleNamespace

    from gonet_astrometry.calibration.stellar_camera import StellarCameraCalibration
    from gonet_astrometry.tracking.catalog_identification import (
        StellarIdentification,
        StellarIdentificationEpoch,
        StellarIdentificationResult,
    )
    from gonet_astrometry.tracking.sequence import DetectionEpoch, DetectionSequence

    paths = tuple((tmp_path / f"stellar-{index}.jpg").resolve() for index in range(2))
    for path in paths:
        path.write_bytes(b"raw")
    frames = [_frame(path, index) for index, path in enumerate(paths)]
    catalogs = tuple(
        DetectionCatalog(
            str(path),
            (
                Detection(0, 5.0 + index, 6.0, 10.0, 10.0, 0.5, 0.5),
            ),
            "sep-independent-channels",
        )
        for index, path in enumerate(paths)
    )
    sequence = DetectionSequence.from_epochs(
        [
            DetectionEpoch.from_frame(str(path), frame, catalog)
            for path, frame, catalog in zip(paths, frames, catalogs, strict=True)
        ]
    )
    detected = DetectionRunArtifacts(
        files=paths,
        algorithm="sep-independent-channels",
        sequence=sequence,
        reference_path=paths[0],
        reference_catalog=catalogs[0],
        reference_prepared=_prepared(),
    )
    stellar_path = tmp_path / "stellar_camera_calibration.npz"
    stellar_path.write_bytes(b"stellar calibration fingerprint")
    catalog_path = tmp_path / "bright_star_catalog.npz"
    catalog_path.write_bytes(b"catalog fingerprint")
    calibration = StellarCameraCalibration(
        image_shape=(20, 20),
        camera_to_enu=np.eye(3),
        center_x_px=10.0,
        center_y_px=10.0,
        radial_c1_px=8.0,
        radial_c3_px=0.5,
        calibrated_theta_max_deg=60.0,
    )
    monkeypatch.setattr(
        "gonet_astrometry.runner.load_stellar_camera_calibration_product",
        lambda path: SimpleNamespace(
            fit=SimpleNamespace(calibration=calibration),
            config=SimpleNamespace(final_match_radius_px=5.0),
        ),
    )

    class FakeMultichannelDetector:
        name = "sep-independent-channels"

        def __init__(self, *args, **kwargs):
            del args, kwargs

        def prepare_report_image(self, frame):
            del frame
            return _prepared()

    monkeypatch.setattr(
        "gonet_astrometry.runner.IndependentChannelSEPDetector",
        FakeMultichannelDetector,
    )
    monkeypatch.setattr(
        "gonet_astrometry.runner.execute_multichannel_detection_run",
        lambda *args, **kwargs: detected,
    )
    monkeypatch.setattr(
        "gonet_astrometry.runner.load_or_fetch_bright_star_catalog",
        lambda cache_path: SimpleNamespace(
            stars=(object(),),
            query_bright_stars=lambda epoch, limiting_magnitude: (object(),),
        ),
    )
    identification_result = StellarIdentificationResult(
        grid_to_enu=np.eye(3),
        bootstrap_frame_identifier="stellar-camera-calibration",
        epochs=tuple(
            StellarIdentificationEpoch(
                frame_identifier=str(path),
                visible_stars=(
                    StellarIdentification(
                        catalog_identifier="HR 1",
                        catalog_magnitude=1.0,
                        azimuth_deg=0.0,
                        altitude_deg=80.0,
                        predicted_x=5.0 + index,
                        predicted_y=6.0,
                        detection_identifier=0,
                        residual_px=0.1,
                        residual_arcmin=0.2,
                        match_kind="primary",
                    ),
                ),
            )
            for index, path in enumerate(paths)
        ),
        fit_pair_count=2,
        fit_median_residual_arcmin=0.2,
        fit_p90_residual_arcmin=0.2,
    )
    matcher_calls: list[int] = []

    class FakeMatcher:
        def __init__(self, config):
            del config

        def match(self, *args, **kwargs):
            del args, kwargs
            matcher_calls.append(1)
            return identification_result

    monkeypatch.setattr(
        "gonet_astrometry.runner.StellarCameraSequenceMatcher", FakeMatcher
    )
    monkeypatch.setattr(
        "gonet_astrometry.runner.load_gonet_file_raw",
        lambda path, parse_metadata=False: object(),
    )
    monkeypatch.setattr(
        "gonet_astrometry.runner.get_raw_channel",
        lambda raw, channel: np.zeros((10, 10), dtype=float),
    )
    monkeypatch.setattr(
        "gonet_astrometry.runner.write_tracking_report_pdf",
        lambda path, **kwargs: path.resolve(),
    )

    product_dir = tmp_path / "products"
    summary = run_cli_workflow(
        inputs=[tmp_path],
        recursive=False,
        algorithm="sep",
        detection_config=DetectionConfig(),
        tracking_config=TrackingConfig(),
        tracking_mode="catalog",
        channel="green1",
        output_path=tmp_path / "stellar.pdf",
        output_dir=product_dir,
        stellar_camera_calibration_path=stellar_path,
        catalog_cache_path=catalog_path,
        metadata_loader=lambda source: frames[paths.index(source)].metadata,
    )

    assert summary.track_count == 1
    assert summary.catalog_track_count == 1
    assert summary.assigned_detection_count == 2
    assert summary.stellar_identification_product_path == (
        product_dir / "stellar_camera_identifications.npz"
    ).resolve()
    assert summary.stellar_camera_calibration_product_path == stellar_path.resolve()
    assert matcher_calls == [1]
