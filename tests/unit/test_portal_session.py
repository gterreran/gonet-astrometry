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


def test_session_caches_scientific_frame_and_detection_catalog(tmp_path: Path) -> None:
    from datetime import datetime, timezone

    from gonet_astrometry.detection.config import DetectionConfig
    from gonet_astrometry.models.detection import DetectionCatalog
    from gonet_astrometry.models.frame import (
        ImageFrame,
        ImageMetadata,
        ObserverLocation,
    )

    selected = tmp_path / "selected.jpg"
    selected.touch()
    frame_calls: list[Path] = []
    detector_calls: list[tuple[str, str]] = []
    metadata = ImageMetadata(
        datetime(2026, 8, 6, tzinfo=timezone.utc),
        10.0,
        ObserverLocation(0.0, 0.0),
    )
    frame = ImageFrame(np.ones((4, 4)), metadata)

    def frame_loader(path: Path) -> ImageFrame:
        frame_calls.append(path)
        return frame

    class FakeDetector:
        name = "fake"

        def detect_prepared(
            self,
            frame_identifier: str,
            prepared: object,
        ) -> DetectionCatalog:
            assert prepared is not None
            detector_calls.append(("fake", frame_identifier))
            return DetectionCatalog(frame_identifier, (), self.name)

    def detector_factory(
        identifier: str,
        _config: DetectionConfig | None,
    ) -> FakeDetector:
        detector_calls.append(("factory", identifier))
        return FakeDetector()

    session = PortalSession(
        loader=lambda _path: FakeRawFile(),
        frame_loader=frame_loader,
        detector_factory=detector_factory,
    )
    session.discover((selected,))

    config = DetectionConfig()
    first = session.detect(selected, "fake", config)
    second = session.detect(selected, "fake", config)

    assert first.frame_identifier == str(selected.resolve())
    assert second.detector_name == "fake"
    assert session.detection_catalog is second
    assert session.prepared_image is not None
    assert session.frame_path == selected.resolve()
    assert frame_calls == [selected.resolve()]
    assert detector_calls == [
        ("factory", "fake"),
        ("fake", str(selected.resolve())),
        ("factory", "fake"),
        ("fake", str(selected.resolve())),
    ]


def test_session_records_separate_detection_timings(tmp_path: Path) -> None:
    from datetime import datetime, timezone

    from gonet_astrometry.detection.config import DetectionConfig
    from gonet_astrometry.models.detection import Detection, DetectionCatalog
    from gonet_astrometry.models.frame import (
        ImageFrame,
        ImageMetadata,
        ObserverLocation,
    )

    selected = tmp_path / "selected.jpg"
    selected.touch()
    metadata = ImageMetadata(
        datetime(2026, 8, 6, tzinfo=timezone.utc),
        10.0,
        ObserverLocation(0.0, 0.0),
    )
    frame = ImageFrame(np.ones((4, 4)), metadata)

    class FakeDetector:
        name = "timed-fake"

        def detect_prepared(
            self,
            frame_identifier: str,
            _prepared: object,
        ) -> DetectionCatalog:
            detection = Detection(1, 1.0, 1.0, 5.0, 5.0, 0.1, 0.1)
            return DetectionCatalog(frame_identifier, (detection,), self.name)

    times = iter((10.0, 10.1, 10.15, 10.35, 10.65))
    session = PortalSession(
        loader=lambda _path: FakeRawFile(),
        frame_loader=lambda _path: frame,
        detector_factory=lambda _identifier, _config: FakeDetector(),
        clock=lambda: next(times),
    )
    session.discover((selected,))

    session.detect(selected, "fake", DetectionConfig())
    timing = session.detection_timing

    assert timing is not None
    assert timing.frame_load_seconds == pytest.approx(0.1)
    assert timing.detector_setup_seconds == pytest.approx(0.05)
    assert timing.preprocessing_seconds == pytest.approx(0.2)
    assert timing.backend_seconds == pytest.approx(0.3)
    assert timing.detector_seconds == pytest.approx(0.5)
    assert timing.total_seconds == pytest.approx(0.65)
    assert timing.source_count == 1
    assert timing.sources_per_second == pytest.approx(10.0 / 3.0)


def test_session_processes_sequence_lazily_and_reuses_catalogs(tmp_path: Path) -> None:
    from datetime import datetime, timedelta, timezone

    from gonet_astrometry.detection.config import DetectionConfig
    from gonet_astrometry.models.detection import Detection, DetectionCatalog
    from gonet_astrometry.models.frame import (
        ImageFrame,
        ImageMetadata,
        ObserverLocation,
    )
    from gonet_astrometry.tracking.config import TrackingConfig

    files = tuple(tmp_path / f"frame-{index}.jpg" for index in range(3))
    for path in files:
        path.touch()

    frame_calls: list[Path] = []
    detector_calls: list[str] = []
    base_time = datetime(2026, 8, 8, 12, 0, tzinfo=timezone.utc)

    def frame_loader(path: Path) -> ImageFrame:
        frame_calls.append(path)
        index = int(path.stem.split("-")[-1])
        metadata = ImageMetadata(
            base_time + timedelta(minutes=index),
            10.0,
            ObserverLocation(41.0, -87.0, 180.0),
            source_path=path,
        )
        return ImageFrame(np.ones((8, 12), dtype=np.float64), metadata)

    class SequenceDetector:
        name = "sequence-fake"

        def detect_prepared(
            self,
            frame_identifier: str,
            _prepared: object,
        ) -> DetectionCatalog:
            detector_calls.append(frame_identifier)
            index = int(Path(frame_identifier).stem.split("-")[-1])
            detection = Detection(
                0,
                2.0 + index,
                3.0,
                10.0,
                8.0,
                0.1,
                0.1,
            )
            return DetectionCatalog(frame_identifier, (detection,), self.name)

    session = PortalSession(
        loader=lambda _path: FakeRawFile(),
        frame_loader=frame_loader,
        detector_factory=lambda _identifier, _config: SequenceDetector(),
    )
    session.discover(files)
    session.load(files[0])

    detection_config = DetectionConfig()
    first = session.build_tracks(
        files,
        "fake",
        detection_config,
        TrackingConfig(
            max_speed_px_per_minute=10.0,
            prediction_tolerance_px=2.0,
            min_track_length=3,
        ),
    )

    assert len(first.tracks) == 1
    assert session.loaded_path == files[0].resolve()
    assert session.frame_path == files[0].resolve()
    assert frame_calls == [path.resolve() for path in files]
    assert detector_calls == [str(path.resolve()) for path in files]
    assert session.catalog_for_path(files[1]) is not None

    second = session.build_tracks(
        files,
        "fake",
        detection_config,
        TrackingConfig(
            max_speed_px_per_minute=12.0,
            prediction_tolerance_px=3.0,
            min_track_length=3,
        ),
    )

    assert len(second.tracks) == 1
    assert frame_calls == [path.resolve() for path in files]
    assert detector_calls == [str(path.resolve()) for path in files]
