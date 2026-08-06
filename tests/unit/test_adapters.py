import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from gonet_astrometry.adapters import gonet_wizard
from gonet_astrometry.adapters.gonet_wizard import (
    GONetImageLoader,
    get_raw_channel,
    load_gonet_file_raw,
    load_gonet_image,
    load_grid_calibration,
    reconstruct_bayer_mosaic,
)


class FakeRawFile:
    filename = "frame.jpg"

    def __init__(self, *, is_bayer_planes: bool = False) -> None:
        self.is_bayer_planes = is_bayer_planes
        self.meta: dict[str, object] | None = {
            "date_utc": "2025-08-06T06:00:46+00:00",
            "exptime": 1.0,
            "latitude": 41.88,
            "longitude": -87.63,
        }
        compact = {
            "blue": np.full((2, 3), 1.0),
            "green1": np.full((2, 3), 2.0),
            "green2": np.full((2, 3), 3.0),
            "red": np.full((2, 3), 4.0),
        }
        if is_bayer_planes:
            self.channels = self._expand(compact)
        else:
            self.channels = compact

    def get_channel(self, channel_name: str) -> np.ndarray:
        return self.channels[channel_name]

    def to_bayer_planes(
        self,
        fill_value: float | int = np.nan,
    ) -> dict[str, np.ndarray]:
        if self.is_bayer_planes:
            return self.channels
        return self._expand(self.channels, fill_value=fill_value)

    @staticmethod
    def _expand(
        channels: dict[str, np.ndarray],
        *,
        fill_value: float | int = np.nan,
    ) -> dict[str, np.ndarray]:
        rows, columns = channels["blue"].shape
        planes = {
            name: np.full((2 * rows, 2 * columns), fill_value, dtype=np.float64)
            for name in channels
        }
        planes["blue"][0::2, 0::2] = channels["blue"]
        planes["green1"][0::2, 1::2] = channels["green1"]
        planes["green2"][1::2, 0::2] = channels["green2"]
        planes["red"][1::2, 1::2] = channels["red"]
        return planes


def test_load_gonet_file_raw_uses_wizard_native_loader(monkeypatch) -> None:
    calls: list[tuple[Path, bool]] = []
    loaded = FakeRawFile()

    class FakeGONetFileRaw:
        @classmethod
        def from_file(cls, path: Path, *, meta: bool) -> FakeRawFile:
            calls.append((path, meta))
            return loaded

    package = ModuleType("GONet_Wizard")
    utilities = ModuleType("GONet_Wizard.GONet_utils")
    utilities.GONetFileRaw = FakeGONetFileRaw  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "GONet_Wizard", package)
    monkeypatch.setitem(sys.modules, "GONet_Wizard.GONet_utils", utilities)

    assert load_gonet_file_raw(Path("frame.jpg")) is loaded
    assert load_gonet_file_raw(Path("science.jpg"), parse_metadata=True) is loaded
    assert calls == [(Path("frame.jpg"), False), (Path("science.jpg"), True)]


def test_load_gonet_file_raw_reports_missing_wizard(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "GONet_Wizard", None)
    monkeypatch.delitem(sys.modules, "GONet_Wizard.GONet_utils", raising=False)

    with pytest.raises(RuntimeError, match="GONet Wizard is required"):
        load_gonet_file_raw(Path("frame.jpg"))


def test_get_raw_channel_validates_name_and_shape() -> None:
    raw = FakeRawFile()

    assert np.array_equal(get_raw_channel(raw, "green1"), raw.channels["green1"])

    with pytest.raises(ValueError, match="Unsupported"):
        get_raw_channel(raw, "green")  # type: ignore[arg-type]

    raw.channels["red"] = np.zeros((2, 3, 1))
    with pytest.raises(ValueError, match="two-dimensional"):
        get_raw_channel(raw, "red")


def test_reconstruct_bayer_mosaic_from_compact_and_expanded_channels() -> None:
    expected = np.array(
        [
            [1, 2, 1, 2, 1, 2],
            [3, 4, 3, 4, 3, 4],
            [1, 2, 1, 2, 1, 2],
            [3, 4, 3, 4, 3, 4],
        ],
        dtype=np.float64,
    )

    assert np.array_equal(reconstruct_bayer_mosaic(FakeRawFile()), expected)
    assert np.array_equal(
        reconstruct_bayer_mosaic(FakeRawFile(is_bayer_planes=True)),
        expected,
    )


def test_reconstruct_bayer_mosaic_validates_planes() -> None:
    raw = FakeRawFile()
    planes = raw.to_bayer_planes()
    planes.pop("red")
    raw.to_bayer_planes = (  # type: ignore[method-assign]
        lambda fill_value=np.nan: planes
    )
    with pytest.raises(ValueError, match="missing channels"):
        reconstruct_bayer_mosaic(raw)

    raw = FakeRawFile(is_bayer_planes=True)
    raw.channels["red"] = np.zeros((3, 6))
    with pytest.raises(ValueError, match="matching two-dimensional"):
        reconstruct_bayer_mosaic(raw)

    raw = FakeRawFile(is_bayer_planes=True)
    raw.channels["red"][0, 0] = 4.0
    with pytest.raises(ValueError, match="exactly one"):
        reconstruct_bayer_mosaic(raw)


def test_load_gonet_image_builds_scientific_frame(monkeypatch) -> None:
    raw = FakeRawFile()
    calls: list[tuple[Path, bool]] = []

    def fake_loader(path: Path, *, parse_metadata: bool = False) -> FakeRawFile:
        calls.append((path, parse_metadata))
        return raw

    monkeypatch.setattr(gonet_wizard, "load_gonet_file_raw", fake_loader)

    frame = load_gonet_image(Path("frame.jpg"))

    assert frame.shape == (4, 6)
    assert frame.metadata.location.latitude_deg == 41.88
    assert frame.metadata.source_path == Path("frame.jpg")
    assert calls == [(Path("frame.jpg"), True)]
    assert np.array_equal(GONetImageLoader().load(Path("frame.jpg")).data, frame.data)


def test_grid_adapter_is_explicitly_unimplemented() -> None:
    with pytest.raises(NotImplementedError, match="Grid calibration adapter"):
        load_grid_calibration(Path("grid.json"))
