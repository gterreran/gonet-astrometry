import numpy as np

from gonet_astrometry.portal.figures import channel_image_figure, empty_image_figure


def test_empty_image_figure_contains_message() -> None:
    figure = empty_image_figure("Nothing loaded")

    assert figure.layout.annotations[0].text == "Nothing loaded"


def test_channel_image_figure_contains_native_channel_data() -> None:
    data = np.arange(24, dtype=np.float64).reshape(4, 6)

    figure = channel_image_figure(
        data,
        channel="green1",
        source_name="frame.jpg",
    )

    assert len(figure.data) == 1
    assert figure.data[0].type == "heatmap"
    assert np.array_equal(figure.data[0].z, data)
    assert figure.layout.dragmode == "pan"
    assert figure.layout.yaxis.autorange == "reversed"


def test_channel_image_figure_handles_constant_and_nonfinite_data() -> None:
    constant = channel_image_figure(
        np.ones((2, 2), dtype=np.float64),
        channel="red",
        source_name="constant.jpg",
    )
    nonfinite = channel_image_figure(
        np.full((2, 2), np.nan),
        channel="blue",
        source_name="nan.jpg",
    )

    assert constant.data[0].zmax > constant.data[0].zmin
    assert (nonfinite.data[0].zmin, nonfinite.data[0].zmax) == (0.0, 1.0)


def test_channel_image_figure_overlays_full_sensor_detections() -> None:
    from gonet_astrometry.models.detection import Detection, DetectionCatalog

    catalog = DetectionCatalog(
        "frame.jpg",
        (
            Detection(
                identifier=1,
                x=20.0,
                y=10.0,
                flux=15.0,
                signal_to_noise=7.0,
                x_uncertainty=0.1,
                y_uncertainty=0.1,
            ),
        ),
        "test",
    )

    figure = channel_image_figure(
        np.ones((20, 30), dtype=np.float64),
        channel="green1",
        source_name="frame.jpg",
        detections=catalog,
    )

    assert len(figure.data) == 2
    assert figure.data[1].type == "scattergl"
    assert tuple(figure.data[1].x) == (10.0,)
    assert tuple(figure.data[1].y) == (5.0,)


def test_channel_image_figure_overlays_mask_boundaries() -> None:
    field_mask = np.zeros((20, 24), dtype=bool)
    field_mask[4:18, 4:22] = True
    dynamic_mask = np.zeros((20, 24), dtype=bool)
    dynamic_mask[10:16, 14:20] = True

    figure = channel_image_figure(
        np.ones((10, 12), dtype=np.float64),
        channel="green1",
        source_name="frame.jpg",
        field_mask=field_mask,
        dynamic_mask=dynamic_mask,
    )

    assert len(figure.data) == 3
    assert figure.data[1].name == "Usable field boundary"
    assert figure.data[2].name == "Bright-region mask"


def test_channel_image_figure_groups_diagnostic_classes_and_hover_values() -> None:
    from gonet_astrometry.models.detection import (
        Detection,
        DetectionCatalog,
        DetectionDiagnostics,
    )

    compact = Detection(
        1,
        10.0,
        12.0,
        30.0,
        9.0,
        0.1,
        0.1,
        elongation=1.2,
        diagnostics=DetectionDiagnostics(
            peak_value=9.0,
            area_pixels=8,
            ellipticity=0.2,
        ),
    )
    extended = Detection(
        2,
        20.0,
        22.0,
        80.0,
        15.0,
        0.1,
        0.1,
        elongation=2.5,
        flags=("extended", "elongated"),
        diagnostics=DetectionDiagnostics(
            peak_value=15.0,
            area_pixels=45,
            ellipticity=0.6,
        ),
    )

    figure = channel_image_figure(
        np.ones((20, 30), dtype=np.float64),
        channel="green1",
        source_name="frame.jpg",
        detections=DetectionCatalog("frame", (compact, extended), "test"),
    )

    assert len(figure.data) == 3
    assert figure.data[1].name == "Compact (1)"
    assert figure.data[2].name == "Extended (1)"
    assert figure.data[1].customdata[0][1] == "compact"
    assert figure.data[1].customdata[0][4] == "9.00"
    assert figure.data[2].customdata[0][6] == "45"
    assert figure.data[2].customdata[0][10] == "0.600"
    assert figure.data[2].customdata[0][16] == "extended, elongated"


def test_channel_image_figure_overlays_image_plane_tracks() -> None:
    from datetime import datetime, timedelta, timezone
    from pathlib import Path

    from gonet_astrometry.models.detection import Detection, DetectionCatalog
    from gonet_astrometry.models.frame import ObserverLocation
    from gonet_astrometry.tracking.config import TrackingConfig
    from gonet_astrometry.tracking.image_plane import ImagePlaneTracker
    from gonet_astrometry.tracking.sequence import DetectionEpoch, DetectionSequence

    location = ObserverLocation(41.0, -87.0, 180.0)
    start = datetime(2026, 8, 8, tzinfo=timezone.utc)
    epochs = []
    for index, x in enumerate((10.0, 14.0, 18.0)):
        frame_id = f"frame-{index}.jpg"
        catalog = DetectionCatalog(
            frame_id,
            (Detection(0, x, 20.0, 10.0, 8.0, 0.1, 0.1),),
            "synthetic",
        )
        epochs.append(
            DetectionEpoch(
                frame_identifier=frame_id,
                source_path=Path(frame_id),
                exposure_midpoint=start + timedelta(minutes=index),
                location=location,
                image_shape=(40, 60),
                sensor_orientation="native",
                catalog=catalog,
            )
        )
    result = ImagePlaneTracker(
        TrackingConfig(
            max_speed_px_per_minute=10.0,
            prediction_tolerance_px=2.0,
            min_track_length=3,
        )
    ).track(DetectionSequence.from_epochs(epochs))

    figure = channel_image_figure(
        np.ones((20, 30), dtype=np.float64),
        channel="green1",
        source_name="frame-0.jpg",
        tracking_result=result,
    )

    assert len(figure.data) == 2
    assert figure.data[1].name == "Candidate tracks (1)"
    assert tuple(figure.data[1].x[:3]) == (5.0, 7.0, 9.0)
    assert figure.data[1].customdata[0][0] == "0"
