"""Temporal source-track models."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TrackPoint:
    """Reference to one detection participating in a track.

    Parameters
    ----------
    frame_identifier
        Identifier of the image containing the detection.
    detection_identifier
        Identifier of the detection within that image.
    """

    frame_identifier: str
    detection_identifier: int


@dataclass(frozen=True, slots=True)
class StarTrack:
    """Candidate association of a stellar source across multiple images.

    Parameters
    ----------
    identifier
        Stable track identifier.
    points
        Time-ordered references to detections.
    catalog_identifier
        Optional identifier of a matched catalog star.
    quality
        Optional normalized track quality in the closed interval ``[0, 1]``.

    Raises
    ------
    ValueError
        If fewer than two points are supplied or ``quality`` lies outside its
        allowed interval.
    """

    identifier: int
    points: tuple[TrackPoint, ...]
    catalog_identifier: str | None = None
    quality: float | None = None

    def __post_init__(self) -> None:
        if len(self.points) < 2:
            raise ValueError("A StarTrack requires at least two detections")
        if self.quality is not None and not 0.0 <= self.quality <= 1.0:
            raise ValueError("StarTrack.quality must lie in [0, 1]")
