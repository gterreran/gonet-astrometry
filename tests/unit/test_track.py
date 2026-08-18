import pytest

from gonet_astrometry.models.track import StarTrack, TrackPoint


def test_track_accepts_two_points() -> None:
    points = (TrackPoint("a", 1), TrackPoint("b", 2))
    track = StarTrack(8, points, quality=0.75)
    assert track.points == points


def test_track_requires_two_points() -> None:
    with pytest.raises(ValueError, match="at least two"):
        StarTrack(8, (TrackPoint("a", 1),))


@pytest.mark.parametrize("quality", [-0.01, 1.01])
def test_track_rejects_invalid_quality(quality: float) -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        StarTrack(8, (TrackPoint("a", 1), TrackPoint("b", 2)), quality=quality)


def test_track_diagnostics_validate_values_and_classification() -> None:
    from gonet_astrometry.models.track import TrackDiagnostics

    diagnostics = TrackDiagnostics(60.0, 5.0, 5.0, 0.2, 0, "low-motion")
    track = StarTrack(
        9,
        (TrackPoint("a", 1), TrackPoint("b", 2)),
        diagnostics=diagnostics,
    )
    assert track.diagnostic_class == "low-motion"

    default = StarTrack(10, (TrackPoint("a", 1), TrackPoint("b", 2)))
    assert default.diagnostic_class == "candidate"

    with pytest.raises(ValueError, match="negative values"):
        TrackDiagnostics(-1.0, 0.0, 0.0, 0.0, 0)
    with pytest.raises(ValueError, match="missed_frames"):
        TrackDiagnostics(1.0, 0.0, 0.0, 0.0, -1)
