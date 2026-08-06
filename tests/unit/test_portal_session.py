from pathlib import Path

import numpy as np
import pytest

from gonet_astrometry.portal.session import PortalSession


class FakeRawFile:
    filename = "frame.jpg"
    is_bayer_planes = False

    def get_channel(self, _channel_name: str) -> np.ndarray:
        return np.ones((2, 3), dtype=np.float64)


def test_session_discovers_without_loading_and_caches_one_file(tmp_path: Path) -> None:
    first = tmp_path / "first.jpg"
    second = tmp_path / "second.jpg"
    first.touch()
    second.touch()
    calls: list[Path] = []

    def loader(path: Path) -> FakeRawFile:
        calls.append(path)
        return FakeRawFile()

    session = PortalSession(loader=loader)
    result = session.discover((tmp_path,))

    assert result.files == (first.resolve(), second.resolve())
    assert calls == []
    assert session.loaded_path is None

    first_loaded = session.load(first)
    assert session.load(first) is first_loaded
    assert calls == [first.resolve()]
    assert session.loaded_path == first.resolve()

    session.load(second)
    assert calls == [first.resolve(), second.resolve()]
    assert session.loaded_path == second.resolve()


def test_session_rejects_unregistered_files_and_invalidates_removed_cache(
    tmp_path: Path,
) -> None:
    selected = tmp_path / "selected.jpg"
    other = tmp_path / "other.jpg"
    selected.touch()
    other.touch()
    session = PortalSession(loader=lambda _path: FakeRawFile())
    session.discover((selected,))
    session.load(selected)

    with pytest.raises(ValueError, match="not registered"):
        session.load(other)

    session.discover((other,))
    assert session.loaded_path is None


def test_session_preserves_cached_file_when_rediscovered(tmp_path: Path) -> None:
    selected = tmp_path / "selected.jpg"
    selected.touch()
    calls: list[Path] = []

    def loader(path: Path) -> FakeRawFile:
        calls.append(path)
        return FakeRawFile()

    session = PortalSession(loader=loader)
    session.discover((selected,))
    loaded = session.load(selected)
    session.discover((tmp_path,))

    assert session.load(selected) is loaded
    assert calls == [selected.resolve()]
