"""Temporal source-track models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

TrackClass = Literal["candidate", "low-motion", "poor-fit"]
"""Non-destructive image-plane classification for one source track."""


@dataclass(frozen=True, slots=True)
class TrackDiagnostics:
    """Motion and fit diagnostics for one candidate track.

    Parameters
    ----------
    duration_s
        Elapsed time between the first and last detections.
    displacement_px
        Straight-line displacement in native full-sensor pixels.
    mean_speed_px_per_minute
        End-to-end apparent speed in native full-sensor pixels per minute.
    fit_rms_px
        RMS residual around a quadratic image-plane fit. This is only a
        bootstrap smoothness diagnostic; it is not a celestial-motion model.
    missed_frames
        Number of epochs missing between the first and last detections.
    diagnostic_class
        Non-destructive classification of the tracklet.
    detection_count
        Number of detections retained in the track.
    span_epoch_count
        Number of sequence epochs from the first through the last detection,
        inclusive.
    coverage_fraction
        Fraction of epochs in the track span that contain a detection.
    median_interval_s
        Median elapsed time between consecutive detections in the track.
    max_interval_s
        Longest elapsed time between consecutive detections in the track.
    """

    duration_s: float
    displacement_px: float
    mean_speed_px_per_minute: float
    fit_rms_px: float
    missed_frames: int
    diagnostic_class: TrackClass = "candidate"
    detection_count: int = 0
    span_epoch_count: int = 0
    coverage_fraction: float = 0.0
    median_interval_s: float = 0.0
    max_interval_s: float = 0.0

    def __post_init__(self) -> None:
        values = (
            self.duration_s,
            self.displacement_px,
            self.mean_speed_px_per_minute,
            self.fit_rms_px,
        )
        if any(value < 0 for value in values):
            raise ValueError("Track diagnostics cannot contain negative values")
        if self.missed_frames < 0:
            raise ValueError("TrackDiagnostics.missed_frames cannot be negative")
        if self.detection_count < 0 or self.span_epoch_count < 0:
            raise ValueError("TrackDiagnostics counts cannot be negative")
        if not 0.0 <= self.coverage_fraction <= 1.0:
            raise ValueError("TrackDiagnostics.coverage_fraction must lie in [0, 1]")
        if self.median_interval_s < 0 or self.max_interval_s < 0:
            raise ValueError("TrackDiagnostics intervals cannot be negative")


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
    diagnostics
        Optional image-plane motion and fit diagnostics.

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
    diagnostics: TrackDiagnostics | None = None

    def __post_init__(self) -> None:
        if len(self.points) < 2:
            raise ValueError("A StarTrack requires at least two detections")
        if self.quality is not None and not 0.0 <= self.quality <= 1.0:
            raise ValueError("StarTrack.quality must lie in [0, 1]")

    @property
    def diagnostic_class(self) -> TrackClass:
        """Return the non-destructive track class, defaulting to candidate."""
        if self.diagnostics is None:
            return "candidate"
        return self.diagnostics.diagnostic_class
