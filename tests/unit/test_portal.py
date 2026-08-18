from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("dash")

from dash import html

from gonet_astrometry.detection.config import DetectionConfig
from gonet_astrometry.models.detection import Detection, DetectionCatalog
from gonet_astrometry.models.frame import ImageFrame, ImageMetadata, ObserverLocation
from gonet_astrometry.portal import ids
from gonet_astrometry.portal.app import create_app
from gonet_astrometry.portal.callbacks import (
    detect_source_view,
    discover_source_paths,
    load_channel_view,
    track_sequence_view,
)
from gonet_astrometry.portal.layout import build_layout
from gonet_astrometry.portal.session import PortalSession


class FakeRawFile:
    filename = "frame.jpg"
    is_bayer_planes = False

    def get_channel(self, channel_name: str) -> np.ndarray:
        values = {"blue": 1.0, "green1": 2.0, "green2": 3.0, "red": 4.0}
        return np.full((4, 6), values[channel_name], dtype=np.float64)


def _raw_loader(_path: Path) -> FakeRawFile:
    return FakeRawFile()


def _frame_loader(_path: Path) -> ImageFrame:
    metadata = ImageMetadata(
        datetime(2026, 8, 6, tzinfo=timezone.utc),
        10.0,
        ObserverLocation(0.0, 0.0),
    )
    return ImageFrame(np.ones((8, 12), dtype=np.float64), metadata)


class FakeDetector:
    name = "fake-detector"

    def detect_prepared(
        self,
        frame_identifier: str,
        _prepared: object,
    ) -> DetectionCatalog:
        detection = Detection(1, 4.0, 6.0, 10.0, 8.0, 0.1, 0.1)
        return DetectionCatalog(frame_identifier, (detection,), self.name)


def _detector_factory(
    _identifier: str,
    _config: DetectionConfig | None,
) -> FakeDetector:
    return FakeDetector()


def _component_ids(component: object) -> set[str]:
    ids_found: set[str] = set()
    component_id = getattr(component, "id", None)
    if isinstance(component_id, str):
        ids_found.add(component_id)

    children = getattr(component, "children", None)
    if isinstance(children, (list, tuple)):
        for child in children:
            ids_found.update(_component_ids(child))
    elif children is not None:
        ids_found.update(_component_ids(children))
    return ids_found


def test_discover_source_paths_registers_candidates_lazily(tmp_path: Path) -> None:
    first = tmp_path / "first.jpg"
    second = tmp_path / "second.jpg"
    first.touch()
    second.touch()
    calls: list[Path] = []

    def loader(path: Path) -> FakeRawFile:
        calls.append(path)
        return FakeRawFile()

    session = PortalSession(loader=loader)
    (
        options,
        selected,
        tracking_options,
        tracking_values,
        status,
    ) = discover_source_paths(
        str(tmp_path),
        ["recursive"],
        session=session,
    )

    assert [option["value"] for option in options] == [str(first), str(second)]
    assert selected == str(first)
    assert tracking_options == options
    assert tracking_values == [str(first), str(second)]
    assert "2 candidate GONet files" in status
    assert calls == []


def test_load_channel_view_handles_empty_invalid_and_unregistered_paths(
    tmp_path: Path,
) -> None:
    session = PortalSession(loader=_raw_loader)
    _, status, rows = load_channel_view("", "green1", session=session)
    assert "Discover and select" in status
    assert rows == []

    _, status, rows = load_channel_view("frame.jpg", "green", session=session)
    assert "valid GONet channel" in status
    assert rows == []

    missing = tmp_path / "missing.jpg"
    _, status, rows = load_channel_view(
        str(missing),
        "green1",
        session=session,
    )
    assert "Unable to load" in status
    assert rows == []


def test_load_channel_view_reuses_cached_file_between_channels(tmp_path: Path) -> None:
    image = tmp_path / "frame.jpg"
    image.touch()
    calls: list[Path] = []

    def loader(path: Path) -> FakeRawFile:
        calls.append(path)
        return FakeRawFile()

    session = PortalSession(loader=loader)
    session.discover((image,))

    figure, status, rows = load_channel_view(
        str(image),
        "green1",
        session=session,
    )
    second_figure, _, _ = load_channel_view(
        str(image),
        "red",
        session=session,
    )

    assert figure.data[0].type == "heatmap"
    assert second_figure.data[0].type == "heatmap"
    assert "green1 channel" in status
    assert "6 × 4 pixels" in status
    assert len(rows) == 5
    assert isinstance(rows[0], html.Tr)
    assert calls == [image.resolve()]


def test_detect_source_view_runs_selected_backend(tmp_path: Path) -> None:
    image = tmp_path / "frame.jpg"
    image.touch()
    session = PortalSession(
        loader=_raw_loader,
        frame_loader=_frame_loader,
        detector_factory=_detector_factory,
    )
    session.discover((image,))

    figure, status = detect_source_view(
        str(image),
        "green1",
        "fake",
        5.0,
        3.0,
        5,
        session=session,
    )

    assert len(figure.data) == 2
    assert "Detected 1 source candidate" in status
    assert "backend time" in status
    assert "candidates/s" in status


def test_detect_source_view_validates_inputs(tmp_path: Path) -> None:
    session = PortalSession(loader=_raw_loader)

    _, status = detect_source_view("", "green1", "fake", 5.0, 3.0, 5, session=session)
    assert "select a GONet image" in status

    image = tmp_path / "frame.jpg"
    image.touch()
    session.discover((image,))
    _, status = detect_source_view(
        str(image),
        "green1",
        "fake",
        None,
        3.0,
        5,
        session=session,
    )
    assert "Invalid detector settings" in status


def test_layout_and_app_include_sidebar_controls(tmp_path: Path) -> None:
    image = tmp_path / "frame.jpg"
    image.touch()
    layout = build_layout(image, (image,))
    app = create_app(initial_path=image, raw_loader=_raw_loader)

    assert isinstance(layout, html.Div)
    assert app.title == "GONet Astrometry Calibrator"
    assert len(app.callback_map) >= 6
    assert f"{ids.LOG_WINDOW}.children" in app.callback_map
    assert f"{ids.BTN_EXIT}.disabled" in app.callback_map
    assert {
        ids.SOURCE_PATHS,
        ids.DISCOVER_FILES,
        ids.FILE_SELECT,
        ids.CHANNEL,
        ids.DETECTOR,
        ids.DETECTION_THRESHOLD,
        ids.DETECTION_FWHM,
        ids.DETECTION_MIN_PIXELS,
        ids.DETECT_SOURCES,
        ids.TRACKING_FILES,
        ids.TRACK_MAX_SPEED,
        ids.TRACK_PREDICTION_TOLERANCE,
        ids.TRACK_MAX_GAP,
        ids.TRACK_MIN_LENGTH,
        ids.BUILD_TRACKS,
        ids.SHOW_DETECTION_MASKS,
        ids.LOAD_IMAGE,
        ids.BTN_EXIT,
    }.issubset(_component_ids(layout))
    assert app.server.config["astrometry_session"].files == (image.resolve(),)


def test_registered_callbacks_delegate_to_session(tmp_path: Path) -> None:
    image = tmp_path / "frame.jpg"
    image.touch()
    app = create_app(
        initial_path=image,
        raw_loader=_raw_loader,
        frame_loader=_frame_loader,
        detector_factory=_detector_factory,
    )

    image_callback_entry = next(
        entry
        for key, entry in app.callback_map.items()
        if f"{ids.IMAGE_GRAPH}.figure" in key
    )
    image_callback = image_callback_entry["callback"].__wrapped__
    figure, status, rows = image_callback(
        1,
        "red",
        ["show"],
        str(image.resolve()),
    )

    assert figure.data[0].type == "heatmap"
    assert "Loaded frame.jpg" in status
    assert "red channel" in status
    assert len(rows) == 5

    discovery_entry = next(
        entry
        for key, entry in app.callback_map.items()
        if f"{ids.FILE_SELECT}.options" in key
    )
    discovery_callback = discovery_entry["callback"].__wrapped__
    (
        options,
        selected,
        tracking_options,
        tracking_values,
        discovery_status,
    ) = discovery_callback(
        1,
        str(tmp_path),
        ["recursive"],
        str(image.resolve()),
        [str(image.resolve())],
    )
    assert selected == str(image.resolve())
    assert options[0]["value"] == str(image.resolve())
    assert tracking_options == options
    assert tracking_values == [str(image.resolve())]
    assert "candidate GONet file" in discovery_status

    detection_entry = next(
        entry
        for key, entry in app.callback_map.items()
        if f"{ids.DETECTION_STATUS}.children" in key
    )
    detection_callback = detection_entry["callback"].__wrapped__
    detection_figure, detection_status = detection_callback(
        1,
        str(image.resolve()),
        "green1",
        "fake",
        5.0,
        3.0,
        5,
        ["show"],
    )
    assert len(detection_figure.data) == 2
    assert "Detected 1 source candidate" in detection_status


def test_track_sequence_view_builds_bootstrap_tracks(tmp_path: Path) -> None:
    from datetime import timedelta

    files = tuple(tmp_path / f"frame-{index}.jpg" for index in range(3))
    for path in files:
        path.touch()

    base_time = datetime(2026, 8, 6, tzinfo=timezone.utc)

    def sequence_frame_loader(path: Path) -> ImageFrame:
        index = int(path.stem.split("-")[-1])
        metadata = ImageMetadata(
            base_time + timedelta(minutes=index),
            10.0,
            ObserverLocation(0.0, 0.0),
            source_path=path,
        )
        return ImageFrame(np.ones((8, 12), dtype=np.float64), metadata)

    session = PortalSession(
        loader=_raw_loader,
        frame_loader=sequence_frame_loader,
        detector_factory=_detector_factory,
    )
    session.discover(files)

    figure, status = track_sequence_view(
        [str(path) for path in files],
        str(files[0]),
        "green1",
        "fake",
        5.0,
        3.0,
        5,
        20.0,
        6.0,
        1,
        3,
        session=session,
    )

    assert "Built 1 tracklets" in status
    assert session.tracking_result is not None
    assert len(figure.data) >= 3
    assert any("Low-motion tracks" in str(trace.name) for trace in figure.data)
