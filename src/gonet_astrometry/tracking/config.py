"""Configuration for image-plane source tracking."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TrackingConfig:
    """Settings for linking source detections across an image sequence.

    Parameters
    ----------
    max_speed_px_per_minute
        Maximum allowed apparent source speed in native full-sensor pixels per
        minute. The first association for a track uses this value to construct
        a time-scaled search radius.
    prediction_tolerance_px
        Baseline residual tolerance in native full-sensor pixels once a track
        has at least two detections and therefore a measured image-plane
        velocity.
    max_gap_minutes
        Maximum elapsed time allowed between a track's last detection and a new
        candidate detection. This is the primary continuity gate and is based
        on actual exposure timestamps rather than image count.
    max_gap_frames
        Optional legacy guard on the number of missing epochs. ``None`` disables
        the frame-count guard so irregular cadences are governed by time alone.
    max_prediction_gap_scale
        Maximum multiplicative expansion of ``prediction_tolerance_px`` when
        extrapolating farther in time than the interval used to estimate the
        current image-plane velocity.
    min_track_length
        Minimum number of detections required for a returned track.
    low_motion_threshold_px_per_minute
        Tracks moving more slowly than this value are labelled ``low-motion``.
        They are retained because stars near the celestial pole may genuinely
        move slowly in image coordinates.
    poor_fit_rms_px
        Quadratic image-plane fit RMS above which a returned track is labelled
        ``poor-fit``. The label is diagnostic only and does not remove a track.
    location_tolerance_m
        Maximum separation between observing locations in one sequence.

    Notes
    -----
    This configuration belongs to the initial image-plane tracklet linker. It
    intentionally does not encode a celestial rotation model. A later
    spherical-tracking stage will replace these empirical gates with angular
    constraints in camera-ray coordinates.
    """

    max_speed_px_per_minute: float = 20.0
    prediction_tolerance_px: float = 6.0
    max_gap_minutes: float = 15.0
    max_gap_frames: int | None = None
    max_prediction_gap_scale: float = 3.0
    min_track_length: int = 3
    low_motion_threshold_px_per_minute: float = 0.5
    poor_fit_rms_px: float = 3.0
    location_tolerance_m: float = 250.0

    def __post_init__(self) -> None:
        if self.max_speed_px_per_minute <= 0:
            raise ValueError("max_speed_px_per_minute must be strictly positive")
        if self.prediction_tolerance_px <= 0:
            raise ValueError("prediction_tolerance_px must be strictly positive")
        if self.max_gap_minutes <= 0:
            raise ValueError("max_gap_minutes must be strictly positive")
        if self.max_gap_frames is not None and self.max_gap_frames < 0:
            raise ValueError("max_gap_frames cannot be negative")
        if self.max_prediction_gap_scale < 1.0:
            raise ValueError("max_prediction_gap_scale must be at least 1")
        if self.min_track_length < 2:
            raise ValueError("min_track_length must be at least 2")
        if self.low_motion_threshold_px_per_minute < 0:
            raise ValueError("low_motion_threshold_px_per_minute cannot be negative")
        if self.poor_fit_rms_px <= 0:
            raise ValueError("poor_fit_rms_px must be strictly positive")
        if self.location_tolerance_m < 0:
            raise ValueError("location_tolerance_m cannot be negative")
