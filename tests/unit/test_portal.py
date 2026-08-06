from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("dash")

from dash import html

from gonet_astrometry.portal import ids
from gonet_astrometry.portal.app import create_app
from gonet_astrometry.portal.callbacks import (
    discover_source_paths,
    load_channel_view,
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
    options, selected, status = discover_source_paths(
        str(tmp_path),
        ["recursive"],
        session=session,
    )

    assert [option["value"] for option in options] == [str(first), str(second)]
    assert selected == str(first)
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


def test_layout_and_app_include_sidebar_controls(tmp_path: Path) -> None:
    image = tmp_path / "frame.jpg"
    image.touch()
    layout = build_layout(image, (image,))
    app = create_app(initial_path=image, raw_loader=_raw_loader)

    assert isinstance(layout, html.Div)
    assert app.title == "GONet Astrometry Calibrator"
    assert len(app.callback_map) >= 5
    assert f"{ids.LOG_WINDOW}.children" in app.callback_map
    assert f"{ids.BTN_EXIT}.disabled" in app.callback_map
    assert {
        ids.SOURCE_PATHS,
        ids.DISCOVER_FILES,
        ids.FILE_SELECT,
        ids.CHANNEL,
        ids.DETECTOR,
        ids.LOAD_IMAGE,
        ids.BTN_EXIT,
    }.issubset(_component_ids(layout))
    assert app.server.config["astrometry_session"].files == (image.resolve(),)


def test_registered_callbacks_delegate_to_session(tmp_path: Path) -> None:
    image = tmp_path / "frame.jpg"
    image.touch()
    app = create_app(initial_path=image, raw_loader=_raw_loader)

    image_callback_entry = next(
        entry
        for key, entry in app.callback_map.items()
        if f"{ids.IMAGE_GRAPH}.figure" in key
    )
    image_callback = image_callback_entry["callback"].__wrapped__
    figure, status, rows = image_callback(1, "red", str(image.resolve()))

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
    options, selected, discovery_status = discovery_callback(
        1,
        str(tmp_path),
        ["recursive"],
        str(image.resolve()),
    )
    assert selected == str(image.resolve())
    assert options[0]["value"] == str(image.resolve())
    assert "candidate GONet file" in discovery_status
