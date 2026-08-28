from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from gonet_astrometry.detection.config import DetectionConfig
from gonet_astrometry.detection.preprocessing import PreparedDetectionImage
from gonet_astrometry.models.detection import Detection, DetectionCatalog
from gonet_astrometry.models.frame import ImageFrame, ImageMetadata, ObserverLocation
from gonet_astrometry.runner import (
    DetectionRunArtifacts,
    RunSummary,
    _prepare_report_reference,
    _resolve_report_reference_path,
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


def test_run_cli_workflow_reuses_sidereal_solution_independently(
    tmp_path: Path, monkeypatch
) -> None:
    from gonet_astrometry.models.grid import GridCalibration
    from gonet_astrometry.solving.sidereal import (
        SiderealFitConfig,
        SiderealRotationSolution,
        SiderealTrackDiagnostics,
    )
    from gonet_astrometry.tracking.image_plane import ImagePlaneTracker
    from gonet_astrometry.tracking.sequence import DetectionEpoch, DetectionSequence

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
            "sep",
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
    fit_calls: list[int] = []

    def fake_fit(self, tracking_result, grid):
        del self, tracking_result, grid
        fit_calls.append(1)
        return solution

    monkeypatch.setattr("gonet_astrometry.runner.SiderealAxisFitter.fit", fake_fit)
    product_dir = tmp_path / "products"
    fit_config = SiderealFitConfig(
        min_track_points=3,
        min_track_duration_minutes=1.0,
    )

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
        sidereal_config=fit_config,
        metadata_loader=lambda path: frames[paths.index(path)].metadata,
    )
    assert first.reused_sidereal_product is False
    assert first.sidereal_product_path == (
        product_dir / "sidereal_rotation.npz"
    ).resolve()
    assert fit_calls == [1]

    monkeypatch.setattr(
        "gonet_astrometry.runner._prepare_cached_reference",
        lambda product, config: (prepared, catalogs[0]),
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
        sidereal_config=fit_config,
        metadata_loader=lambda path: frames[paths.index(path)].metadata,
    )
    assert second.reused_detection_product is True
    assert second.reused_tracking_product is True
    assert second.reused_sidereal_product is True
    assert fit_calls == [1]

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
        sidereal_config=fit_config,
        overwrite_solution=True,
        metadata_loader=lambda path: frames[paths.index(path)].metadata,
    )
    assert third.reused_detection_product is True
    assert third.reused_tracking_product is True
    assert third.reused_sidereal_product is False
    assert fit_calls == [1, 1]
