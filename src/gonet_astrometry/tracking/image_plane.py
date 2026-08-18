"""Initial image-plane tracklet association across multiple GONet frames.

This module deliberately implements only the bootstrap association stage. It
links detections through short sequences using time-scaled displacement gates
and constant-velocity prediction. The resulting tracklets are intended to seed
the later spherical common-rotation-axis solver; they are not themselves an
astrometric solution.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from gonet_astrometry.models.detection import Detection
from gonet_astrometry.models.track import StarTrack, TrackDiagnostics, TrackPoint
from gonet_astrometry.tracking.config import TrackingConfig
from gonet_astrometry.tracking.sequence import DetectionEpoch, DetectionSequence


@dataclass(frozen=True, slots=True)
class ResolvedTrackPoint:
    """One track point with its resolved coordinates and time."""

    epoch_index: int
    epoch: DetectionEpoch
    detection: Detection


@dataclass(frozen=True, slots=True)
class ImagePlaneTrackingResult:
    """Image-plane tracklets and their source detection sequence."""

    sequence: DetectionSequence
    tracks: tuple[StarTrack, ...]

    @property
    def assigned_detection_count(self) -> int:
        """Return the number of detection references contained in tracks."""
        return sum(len(track.points) for track in self.tracks)

    @property
    def unassigned_detection_count(self) -> int:
        """Return detections not represented in any returned track."""
        return self.sequence.total_detections - self.assigned_detection_count

    def diagnostic_counts(self) -> dict[str, int]:
        """Return mutually exclusive counts by track diagnostic class."""
        counts = {"candidate": 0, "low-motion": 0, "poor-fit": 0}
        for track in self.tracks:
            counts[track.diagnostic_class] += 1
        return counts

    def resolve(self, track: StarTrack) -> tuple[ResolvedTrackPoint, ...]:
        """Resolve lightweight :class:`TrackPoint` references into detections."""
        epoch_by_id = {
            epoch.frame_identifier: (index, epoch)
            for index, epoch in enumerate(self.sequence.epochs)
        }
        resolved: list[ResolvedTrackPoint] = []
        for point in track.points:
            index, epoch = epoch_by_id[point.frame_identifier]
            detection = next(
                candidate
                for candidate in epoch.catalog.detections
                if candidate.identifier == point.detection_identifier
            )
            resolved.append(ResolvedTrackPoint(index, epoch, detection))
        return tuple(resolved)


@dataclass
class _TrackState:
    """Mutable association state used only while constructing tracklets."""

    points: list[ResolvedTrackPoint]

    @property
    def last(self) -> ResolvedTrackPoint:
        return self.points[-1]


class ImagePlaneTracker:
    """Associate detections with a conservative image-plane motion model."""

    def __init__(self, config: TrackingConfig | None = None) -> None:
        self.config = config or TrackingConfig()

    def track(self, sequence: DetectionSequence) -> ImagePlaneTrackingResult:
        """Link detections chronologically and return diagnostic tracklets.

        Notes
        -----
        The first association for each nascent track is based only on the
        configured maximum image-plane speed. Once two detections have been
        linked, constant-velocity prediction is used. Associations are resolved
        greedily from the smallest normalized residual so a detection can
        belong to at most one track at an epoch.
        """
        states = [
            _TrackState([ResolvedTrackPoint(0, sequence.epochs[0], detection)])
            for detection in sequence.epochs[0].catalog.detections
        ]

        for epoch_index, epoch in enumerate(sequence.epochs[1:], start=1):
            active = [
                state
                for state in states
                if self._is_active(state, epoch_index, epoch)
            ]
            assignments = self._assign_epoch(active, epoch_index, epoch)
            matched_detections: set[int] = set()
            for state_index, detection_index in assignments:
                detection = epoch.catalog.detections[detection_index]
                active[state_index].points.append(
                    ResolvedTrackPoint(epoch_index, epoch, detection)
                )
                matched_detections.add(detection_index)

            for detection_index, detection in enumerate(epoch.catalog.detections):
                if detection_index not in matched_detections:
                    states.append(
                        _TrackState([ResolvedTrackPoint(epoch_index, epoch, detection)])
                    )

        retained = [
            state
            for state in states
            if len(state.points) >= self.config.min_track_length
        ]
        tracks = tuple(
            self._finalize_track(identifier, state)
            for identifier, state in enumerate(retained)
        )
        return ImagePlaneTrackingResult(sequence=sequence, tracks=tracks)

    def _is_active(
        self,
        state: _TrackState,
        epoch_index: int,
        epoch: DetectionEpoch,
    ) -> bool:
        """Return whether a track may still be propagated to ``epoch``.

        Actual elapsed time is the primary continuity criterion. The optional
        frame-count guard exists only for callers that explicitly want the old
        epoch-count behavior.
        """
        elapsed_minutes = (
            epoch.exposure_midpoint - state.last.epoch.exposure_midpoint
        ).total_seconds() / 60.0
        if elapsed_minutes > self.config.max_gap_minutes:
            return False
        if self.config.max_gap_frames is None:
            return True
        missing_epochs = epoch_index - state.last.epoch_index - 1
        return missing_epochs <= self.config.max_gap_frames

    def _assign_epoch(
        self,
        active: list[_TrackState],
        epoch_index: int,
        epoch: DetectionEpoch,
    ) -> list[tuple[int, int]]:
        """Return one-to-one active-track/detection assignments for one epoch."""
        candidates: list[tuple[float, int, int]] = []
        for state_index, state in enumerate(active):
            predicted_x, predicted_y, gate = self._prediction(state, epoch_index, epoch)
            last = state.last
            dt_minutes = (
                epoch.exposure_midpoint - last.epoch.exposure_midpoint
            ).total_seconds() / 60.0
            max_displacement = self.config.max_speed_px_per_minute * dt_minutes
            for detection_index, detection in enumerate(epoch.catalog.detections):
                step_distance = math.hypot(
                    detection.x - last.detection.x,
                    detection.y - last.detection.y,
                )
                if (
                    step_distance
                    > max_displacement + self.config.prediction_tolerance_px
                ):
                    continue
                residual = math.hypot(
                    detection.x - predicted_x,
                    detection.y - predicted_y,
                )
                if residual <= gate:
                    candidates.append((residual / gate, state_index, detection_index))

        candidates.sort(key=lambda item: item[0])
        used_states: set[int] = set()
        used_detections: set[int] = set()
        assignments: list[tuple[int, int]] = []
        for _score, state_index, detection_index in candidates:
            if state_index in used_states or detection_index in used_detections:
                continue
            used_states.add(state_index)
            used_detections.add(detection_index)
            assignments.append((state_index, detection_index))
        return assignments

    def _prediction(
        self,
        state: _TrackState,
        epoch_index: int,
        epoch: DetectionEpoch,
    ) -> tuple[float, float, float]:
        """Predict the next image-plane position and association gate."""
        last = state.last
        dt_seconds = (
            epoch.exposure_midpoint - last.epoch.exposure_midpoint
        ).total_seconds()
        if dt_seconds <= 0:
            raise ValueError("Tracking epochs must be strictly chronological")

        if len(state.points) < 2:
            gate = (
                self.config.max_speed_px_per_minute * dt_seconds / 60.0
                + self.config.prediction_tolerance_px
            )
            return last.detection.x, last.detection.y, gate

        previous = state.points[-2]
        velocity_dt = (
            last.epoch.exposure_midpoint - previous.epoch.exposure_midpoint
        ).total_seconds()
        if velocity_dt <= 0:
            raise ValueError("Track timestamps must increase strictly")
        vx = (last.detection.x - previous.detection.x) / velocity_dt
        vy = (last.detection.y - previous.detection.y) / velocity_dt
        time_ratio = max(1.0, dt_seconds / velocity_dt)
        gap_scale = min(
            self.config.max_prediction_gap_scale,
            math.sqrt(time_ratio),
        )
        gate = self.config.prediction_tolerance_px * gap_scale
        return (
            last.detection.x + vx * dt_seconds,
            last.detection.y + vy * dt_seconds,
            gate,
        )

    def _finalize_track(self, identifier: int, state: _TrackState) -> StarTrack:
        """Convert mutable state into an immutable track with diagnostics."""
        diagnostics = _track_diagnostics(state.points, self.config)
        quality = 1.0 / (
            1.0 + (diagnostics.fit_rms_px / self.config.poor_fit_rms_px) ** 2
        )
        return StarTrack(
            identifier=identifier,
            points=tuple(
                TrackPoint(
                    point.epoch.frame_identifier,
                    point.detection.identifier,
                )
                for point in state.points
            ),
            quality=float(max(0.0, min(1.0, quality))),
            diagnostics=diagnostics,
        )


def _track_diagnostics(
    points: list[ResolvedTrackPoint],
    config: TrackingConfig,
) -> TrackDiagnostics:
    """Measure duration, motion, and smoothness for a completed tracklet."""
    first = points[0]
    last = points[-1]
    duration_s = (
        last.epoch.exposure_midpoint - first.epoch.exposure_midpoint
    ).total_seconds()
    displacement = math.hypot(
        last.detection.x - first.detection.x,
        last.detection.y - first.detection.y,
    )
    speed = 0.0 if duration_s <= 0 else displacement / duration_s * 60.0

    times = np.asarray(
        [
            (
                point.epoch.exposure_midpoint - first.epoch.exposure_midpoint
            ).total_seconds()
            for point in points
        ],
        dtype=np.float64,
    )
    xs = np.asarray([point.detection.x for point in points], dtype=np.float64)
    ys = np.asarray([point.detection.y for point in points], dtype=np.float64)
    degree = min(2, len(points) - 1)
    x_coeff = np.polyfit(times, xs, degree)
    y_coeff = np.polyfit(times, ys, degree)
    x_residual = xs - np.polyval(x_coeff, times)
    y_residual = ys - np.polyval(y_coeff, times)
    fit_rms = float(np.sqrt(np.mean(x_residual**2 + y_residual**2)))

    represented = {point.epoch_index for point in points}
    span_frames = last.epoch_index - first.epoch_index + 1
    missed_frames = span_frames - len(represented)
    intervals = np.diff(times)
    median_interval_s = float(np.median(intervals)) if intervals.size else 0.0
    max_interval_s = float(np.max(intervals)) if intervals.size else 0.0
    coverage_fraction = len(points) / span_frames
    if fit_rms > config.poor_fit_rms_px:
        diagnostic_class = "poor-fit"
    elif speed < config.low_motion_threshold_px_per_minute:
        diagnostic_class = "low-motion"
    else:
        diagnostic_class = "candidate"

    return TrackDiagnostics(
        duration_s=duration_s,
        displacement_px=displacement,
        mean_speed_px_per_minute=speed,
        fit_rms_px=fit_rms,
        missed_frames=missed_frames,
        diagnostic_class=diagnostic_class,
        detection_count=len(points),
        span_epoch_count=span_frames,
        coverage_fraction=coverage_fraction,
        median_interval_s=median_interval_s,
        max_interval_s=max_interval_s,
    )
