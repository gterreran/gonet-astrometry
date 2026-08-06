import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from gonet_astrometry.adapters.gonet_wizard import (
    get_raw_channel,
    load_gonet_file_raw,
    load_gonet_image,
    load_grid_calibration,
)


class FakeRawFile:
    filename = "frame.jpg"
    is_bayer_planes = False

    def __init__(self) -> None:
        self.channels = {
            "blue": np.ones((2, 3), dtype=np.float64),
            "green1": np.full((2, 3), 2.0),
            "green2": np.full((2, 3), 3.0),
            "red": np.full((2, 3), 4.0),
        }

    def get_channel(self, channel_name: str) -> np.ndarray:
        return self.channels[channel_name]


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

    result = load_gonet_file_raw(Path("frame.jpg"))

    assert result is loaded
    assert calls == [(Path("frame.jpg"), False)]


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


def test_image_adapter_is_explicitly_unimplemented() -> None:
    with pytest.raises(NotImplementedError, match="image adapter"):
        load_gonet_image(Path("frame.raw"))


def test_grid_adapter_is_explicitly_unimplemented() -> None:
    with pytest.raises(NotImplementedError, match="Grid calibration adapter"):
        load_grid_calibration(Path("grid.json"))
