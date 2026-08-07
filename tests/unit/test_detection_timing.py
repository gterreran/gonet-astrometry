import pytest

from gonet_astrometry.detection.timing import DetectionTiming


def test_detection_timing_reports_backend_throughput() -> None:
    timing = DetectionTiming(
        frame_identifier="frame.jpg",
        detector_name="fake",
        frame_load_seconds=0.1,
        detector_setup_seconds=0.05,
        preprocessing_seconds=0.2,
        backend_seconds=0.5,
        total_seconds=0.85,
        source_count=25,
    )

    assert timing.detector_seconds == pytest.approx(0.7)
    assert timing.sources_per_second == pytest.approx(50.0)


def test_detection_timing_handles_zero_duration() -> None:
    timing = DetectionTiming(
        frame_identifier="frame.jpg",
        detector_name="fake",
        frame_load_seconds=0.0,
        detector_setup_seconds=0.0,
        preprocessing_seconds=0.0,
        backend_seconds=0.0,
        total_seconds=0.0,
        source_count=0,
    )

    assert timing.detector_seconds == 0.0
    assert timing.sources_per_second is None


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"backend_seconds": -0.1}, "cannot be negative"),
        ({"source_count": -1}, "cannot be negative"),
        ({"total_seconds": 0.2}, "shorter"),
    ],
)
def test_detection_timing_rejects_invalid_values(
    overrides: dict[str, float | int],
    message: str,
) -> None:
    values: dict[str, str | float | int] = {
        "frame_identifier": "frame.jpg",
        "detector_name": "fake",
        "frame_load_seconds": 0.1,
        "detector_setup_seconds": 0.1,
        "preprocessing_seconds": 0.1,
        "backend_seconds": 0.1,
        "total_seconds": 0.4,
        "source_count": 1,
    }
    values.update(overrides)

    with pytest.raises(ValueError, match=message):
        DetectionTiming(**values)  # type: ignore[arg-type]
