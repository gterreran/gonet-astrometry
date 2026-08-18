from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from gonet_astrometry.detection.config import DetectionConfig
from gonet_astrometry.detection.preprocessing import PreparedDetectionImage
from gonet_astrometry.diagnostics.report import write_tracking_report_pdf
from gonet_astrometry.models.detection import (
    Detection,
    DetectionCatalog,
    DetectionDiagnostics,
)
from gonet_astrometry.models.frame import ImageFrame, ImageMetadata, ObserverLocation
from gonet_astrometry.runner import TrackingRunArtifacts
from gonet_astrometry.tracking.config import TrackingConfig
from gonet_astrometry.tracking.image_plane import ImagePlaneTracker
from gonet_astrometry.tracking.sequence import DetectionEpoch, DetectionSequence


def _artifacts(tmp_path: Path) -> TrackingRunArtifacts:
    paths = tuple((tmp_path / f"frame-{index}.jpg").resolve() for index in range(3))
    epochs = []
    reference_catalog = None
    for index, path in enumerate(paths):
        metadata = ImageMetadata(
            datetime(2026, 8, 8, tzinfo=timezone.utc) + timedelta(minutes=index),
            10.0,
            ObserverLocation(41.0, -87.0, 180.0),
            source_path=path,
            sensor_orientation="native",
        )
        frame = ImageFrame(np.zeros((20, 20)), metadata)
        detection = Detection(
            identifier=0,
            x=5.0 + index,
            y=7.0,
            flux=20.0,
            signal_to_noise=10.0,
            x_uncertainty=0.5,
            y_uncertainty=0.5,
            diagnostics=DetectionDiagnostics(peak_value=10.0, area_pixels=4),
        )
        catalog = DetectionCatalog(str(path), (detection,), "synthetic")
        if reference_catalog is None:
            reference_catalog = catalog
        epochs.append(DetectionEpoch.from_frame(str(path), frame, catalog))

    sequence = DetectionSequence.from_epochs(epochs)
    config = TrackingConfig(
        max_speed_px_per_minute=5.0,
        prediction_tolerance_px=2.0,
        min_track_length=3,
    )
    result = ImagePlaneTracker(config).track(sequence)
    field = np.ones((20, 20), dtype=bool)
    field[:2, :] = False
    dynamic = np.zeros((20, 20), dtype=bool)
    dynamic[10:12, 10:12] = True
    prepared = PreparedDetectionImage(
        data=np.zeros((20, 20)),
        mask=~field | dynamic,
        field_mask=field,
        dynamic_mask=dynamic,
        backgrounds=(0.0, 0.0, 0.0, 0.0),
        noises=(1.0, 1.0, 1.0, 1.0),
    )
    assert reference_catalog is not None
    return TrackingRunArtifacts(
        files=paths,
        algorithm="sep",
        sequence=sequence,
        tracking_result=result,
        reference_path=paths[0],
        reference_catalog=reference_catalog,
        reference_prepared=prepared,
    )


def test_write_tracking_report_pdf_creates_pdf(tmp_path: Path) -> None:
    output = write_tracking_report_pdf(
        tmp_path / "nested" / "report.pdf",
        image_data=np.arange(100, dtype=float).reshape(10, 10),
        channel="green1",
        artifacts=_artifacts(tmp_path),
        detection_config=DetectionConfig(),
        tracking_config=TrackingConfig(min_track_length=3),
    )

    assert output == (tmp_path / "nested" / "report.pdf").resolve()
    assert output.read_bytes().startswith(b"%PDF")
