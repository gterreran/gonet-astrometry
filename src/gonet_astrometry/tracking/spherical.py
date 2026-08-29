"""Grid-aware spherical temporal association for stellar candidates.

This module promotes the validated temporal-tracking diagnostic into the package.
Candidate detections are converted to Grid-frame unit rays before association.
Tracks are predicted in the local tangent plane of the unit sphere and assigned
globally with a one-to-one Hungarian solve.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment

from gonet_astrometry.adapters.grid_calibration import validated_pixel_rays
from gonet_astrometry.models.detection import Detection
from gonet_astrometry.models.grid import GridCalibration
from gonet_astrometry.models.track import StarTrack, TrackDiagnostics, TrackPoint
from gonet_astrometry.tracking.image_plane import ImagePlaneTrackingResult
from gonet_astrometry.tracking.sequence import DetectionEpoch, DetectionSequence


@dataclass(frozen=True, slots=True)
class SphericalTrackingConfig:
    """Configuration for Grid-aware temporal source association.

    Defaults reproduce the validated ``track_channel_candidates_temporally.py``
    experiment.
    """

    min_track_length: int = 3
    max_gap_minutes: float = 10.0
    single_point_max_gap_seconds: float = 60.0
    max_initial_speed_deg_per_minute: float = 0.30
    initial_position_margin_arcmin: float = 2.0
    prediction_tolerance_arcmin: float = 5.0
    prediction_reference_seconds: float = 15.0
    max_prediction_tolerance_arcmin: float = 25.0
    velocity_fit_points: int = 5
    inverse_reprojection_tolerance_px: float = 1.0e-2

    def __post_init__(self) -> None:
        if self.min_track_length < 2:
            raise ValueError("min_track_length must be at least 2")
        if self.max_gap_minutes <= 0.0:
            raise ValueError("max_gap_minutes must be positive")
        if self.single_point_max_gap_seconds <= 0.0:
            raise ValueError("single_point_max_gap_seconds must be positive")
        if self.max_initial_speed_deg_per_minute <= 0.0:
            raise ValueError("max_initial_speed_deg_per_minute must be positive")
        if self.initial_position_margin_arcmin < 0.0:
            raise ValueError("initial_position_margin_arcmin cannot be negative")
        if self.prediction_tolerance_arcmin <= 0.0:
            raise ValueError("prediction_tolerance_arcmin must be positive")
        if self.prediction_reference_seconds <= 0.0:
            raise ValueError("prediction_reference_seconds must be positive")
        if self.max_prediction_tolerance_arcmin <= 0.0:
            raise ValueError("max_prediction_tolerance_arcmin must be positive")
        if self.velocity_fit_points < 2:
            raise ValueError("velocity_fit_points must be at least 2")
        if self.inverse_reprojection_tolerance_px <= 0.0:
            raise ValueError("inverse_reprojection_tolerance_px must be positive")


@dataclass(frozen=True, slots=True)
class _Candidate:
    epoch_index: int
    epoch: DetectionEpoch
    detection: Detection
    timestamp: float
    ray: NDArray[np.float64]
    channel_support: int


@dataclass(slots=True)
class _TrackState:
    identifier: int
    candidates: list[_Candidate] = field(default_factory=list)

    @property
    def last(self) -> _Candidate:
        return self.candidates[-1]

    @property
    def length(self) -> int:
        return len(self.candidates)


class SphericalTracker:
    """Associate detections through time directly on the calibrated unit sphere."""

    def __init__(self, config: SphericalTrackingConfig | None = None) -> None:
        self.config = config or SphericalTrackingConfig()

    def track(
        self,
        sequence: DetectionSequence,
        calibration: GridCalibration,
    ) -> ImagePlaneTrackingResult:
        """Return temporal tracklets whose association is performed in ray space."""
        if sequence.epochs[0].image_shape != calibration.image_shape:
            raise ValueError(
                "Grid calibration sensor shape does not match the detection sequence: "
                f"calibration={calibration.image_shape}, "
                f"sequence={sequence.epochs[0].image_shape}"
            )

        candidates_by_epoch = [
            self._epoch_candidates(index, epoch, calibration)
            for index, epoch in enumerate(sequence.epochs)
        ]

        states: list[_TrackState] = []
        next_identifier = 0

        for frame_candidates in candidates_by_epoch:
            if not frame_candidates:
                continue

            timestamp = frame_candidates[0].timestamp
            assignments, used = self._associate_frame(
                states,
                frame_candidates,
                timestamp,
            )

            for state_index, candidate_index in assignments.items():
                states[state_index].candidates.append(frame_candidates[candidate_index])

            for candidate_index, candidate in enumerate(frame_candidates):
                if candidate_index in used:
                    continue
                states.append(
                    _TrackState(
                        identifier=next_identifier,
                        candidates=[candidate],
                    )
                )
                next_identifier += 1

        retained = [
            state for state in states if state.length >= self.config.min_track_length
        ]
        tracks = tuple(
            self._finalize_track(identifier, state)
            for identifier, state in enumerate(retained)
        )
        return ImagePlaneTrackingResult(sequence, tracks)

    def _epoch_candidates(
        self,
        epoch_index: int,
        epoch: DetectionEpoch,
        calibration: GridCalibration,
    ) -> list[_Candidate]:
        detections = epoch.catalog.detections
        if not detections:
            return []

        x = np.asarray([item.x for item in detections], dtype=np.float64)
        y = np.asarray([item.y for item in detections], dtype=np.float64)
        converted = validated_pixel_rays(
            calibration,
            x,
            y,
            reprojection_tolerance_px=self.config.inverse_reprojection_tolerance_px,
        )

        timestamp = epoch.exposure_midpoint.timestamp()
        output: list[_Candidate] = []
        for index, detection in enumerate(detections):
            if not bool(converted.valid[index]):
                continue
            output.append(
                _Candidate(
                    epoch_index=epoch_index,
                    epoch=epoch,
                    detection=detection,
                    timestamp=timestamp,
                    ray=np.asarray(converted.rays[index], dtype=np.float64),
                    channel_support=_channel_support(detection),
                )
            )
        return output

    def _associate_frame(
        self,
        tracks: list[_TrackState],
        frame_candidates: list[_Candidate],
        timestamp: float,
    ) -> tuple[dict[int, int], set[int]]:
        active_indices: list[int] = []
        predictions: list[NDArray[np.float64]] = []
        gates: list[float] = []

        for track_index, track in enumerate(tracks):
            gate = self._track_gate_arcmin(track, timestamp)
            if gate is None:
                continue
            active_indices.append(track_index)
            predictions.append(self._predict_track_ray(track, timestamp))
            gates.append(gate)

        if not active_indices or not frame_candidates:
            return {}, set()

        large = 1.0e9
        cost = np.full(
            (len(active_indices), len(frame_candidates)),
            large,
            dtype=np.float64,
        )

        for row, (prediction, gate) in enumerate(zip(predictions, gates, strict=True)):
            for column, candidate in enumerate(frame_candidates):
                separation = _angular_separation_arcmin(
                    prediction,
                    candidate.ray,
                )
                if separation <= gate:
                    # Preserve the validated diagnostic behavior: channel support
                    # is only a tiny tie-breaker. Geometry dominates association.
                    cost[row, column] = separation + 0.02 * max(
                        0, 4 - candidate.channel_support
                    )

        rows, columns = linear_sum_assignment(cost)
        assignments: dict[int, int] = {}
        used_candidates: set[int] = set()

        for row, column in zip(rows, columns, strict=True):
            if cost[int(row), int(column)] >= large:
                continue
            assignments[active_indices[int(row)]] = int(column)
            used_candidates.add(int(column))

        return assignments, used_candidates

    def _track_gate_arcmin(
        self,
        track: _TrackState,
        timestamp: float,
    ) -> float | None:
        dt_seconds = float(timestamp - track.last.timestamp)
        if dt_seconds <= 0.0:
            return None

        if track.length == 1:
            if dt_seconds > self.config.single_point_max_gap_seconds:
                return None
            motion_arcmin = (
                self.config.max_initial_speed_deg_per_minute
                * (dt_seconds / 60.0)
                * 60.0
            )
            return motion_arcmin + self.config.initial_position_margin_arcmin

        if dt_seconds > self.config.max_gap_minutes * 60.0:
            return None

        scale = np.sqrt(max(dt_seconds / self.config.prediction_reference_seconds, 1.0))
        return min(
            self.config.prediction_tolerance_arcmin * float(scale),
            self.config.max_prediction_tolerance_arcmin,
        )

    def _predict_track_ray(
        self,
        track: _TrackState,
        timestamp: float,
    ) -> NDArray[np.float64]:
        if track.length < 2:
            return track.last.ray

        dt_seconds = float(timestamp - track.last.timestamp)
        velocity = _estimate_tangent_velocity(
            track,
            self.config.velocity_fit_points,
        )
        return _sphere_exp_map(
            track.last.ray,
            velocity * dt_seconds,
        )

    def _finalize_track(
        self,
        identifier: int,
        state: _TrackState,
    ) -> StarTrack:
        points = tuple(
            TrackPoint(
                item.epoch.frame_identifier,
                item.detection.identifier,
            )
            for item in state.candidates
        )
        diagnostics = _track_diagnostics(state.candidates)
        return StarTrack(
            identifier=identifier,
            points=points,
            diagnostics=diagnostics,
        )


def _channel_support(detection: Detection) -> int:
    """Return multi-channel support recorded by the detector, defaulting to one."""
    prefix = "channel-support:"
    for flag in detection.flags:
        if not flag.startswith(prefix):
            continue
        try:
            value = int(flag[len(prefix) :])
        except ValueError:
            continue
        if 1 <= value <= 4:
            return value
    return 1


def _normalize(vector: NDArray[np.float64]) -> NDArray[np.float64]:
    value = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(value))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("Cannot normalize a non-finite or zero vector")
    return value / norm


def _angular_separation_arcmin(
    left: NDArray[np.float64],
    right: NDArray[np.float64],
) -> float:
    dot = float(np.clip(np.dot(left, right), -1.0, 1.0))
    return float(np.rad2deg(np.arccos(dot)) * 60.0)


def _sphere_log_map(
    base: NDArray[np.float64],
    target: NDArray[np.float64],
) -> NDArray[np.float64]:
    base = _normalize(base)
    target = _normalize(target)
    dot = float(np.clip(np.dot(base, target), -1.0, 1.0))
    angle = float(np.arccos(dot))
    if angle < 1.0e-12:
        return np.zeros(3, dtype=np.float64)

    tangent = target - dot * base
    norm = float(np.linalg.norm(tangent))
    if norm < 1.0e-12:
        return np.zeros(3, dtype=np.float64)
    return np.asarray(tangent * (angle / norm), dtype=np.float64)


def _sphere_exp_map(
    base: NDArray[np.float64],
    tangent: NDArray[np.float64],
) -> NDArray[np.float64]:
    base = _normalize(base)
    tangent = np.asarray(tangent, dtype=np.float64)
    tangent = tangent - float(np.dot(tangent, base)) * base
    angle = float(np.linalg.norm(tangent))
    if angle < 1.0e-12:
        return base.copy()

    direction = tangent / angle
    return _normalize(np.cos(angle) * base + np.sin(angle) * direction)


def _estimate_tangent_velocity(
    track: _TrackState,
    max_points: int,
) -> NDArray[np.float64]:
    recent = track.candidates[-max_points:]
    if len(recent) < 2:
        return np.zeros(3, dtype=np.float64)

    base = recent[-1].ray
    t0 = recent[-1].timestamp
    dt_values: list[float] = []
    displacements: list[NDArray[np.float64]] = []

    for candidate in recent[:-1]:
        dt = float(candidate.timestamp - t0)
        if abs(dt) < 1.0e-9:
            continue
        dt_values.append(dt)
        displacements.append(_sphere_log_map(base, candidate.ray))

    if not dt_values:
        return np.zeros(3, dtype=np.float64)

    dt_array = np.asarray(dt_values, dtype=np.float64)
    displacement_array = np.stack(displacements, axis=0)
    denominator = float(np.sum(np.square(dt_array)))
    velocity = np.sum(dt_array[:, None] * displacement_array, axis=0) / denominator
    velocity -= float(np.dot(velocity, base)) * base
    return np.asarray(velocity, dtype=np.float64)


def _track_diagnostics(
    candidates: list[_Candidate],
) -> TrackDiagnostics:
    """Return generic bookkeeping diagnostics for a spherical tracklet.

    The diagnostic class is intentionally ``candidate``. Physical rejection is
    deferred to the sidereal fixed-rate solver rather than an image-plane
    straightness heuristic.
    """
    first = candidates[0]
    last = candidates[-1]
    duration_s = float(last.timestamp - first.timestamp)
    displacement = math.hypot(
        last.detection.x - first.detection.x,
        last.detection.y - first.detection.y,
    )
    speed = 0.0 if duration_s <= 0.0 else displacement / duration_s * 60.0

    times = np.asarray(
        [item.timestamp - first.timestamp for item in candidates],
        dtype=np.float64,
    )
    xs = np.asarray(
        [item.detection.x for item in candidates],
        dtype=np.float64,
    )
    ys = np.asarray(
        [item.detection.y for item in candidates],
        dtype=np.float64,
    )
    degree = min(2, len(candidates) - 1)
    x_coeff = np.polyfit(times, xs, degree)
    y_coeff = np.polyfit(times, ys, degree)
    x_residual = xs - np.polyval(x_coeff, times)
    y_residual = ys - np.polyval(y_coeff, times)
    fit_rms = float(np.sqrt(np.mean(np.square(x_residual) + np.square(y_residual))))

    represented = {item.epoch_index for item in candidates}
    span_epoch_count = last.epoch_index - first.epoch_index + 1
    missed_frames = span_epoch_count - len(represented)

    intervals = np.diff(times)
    median_interval_s = float(np.median(intervals)) if intervals.size else 0.0
    max_interval_s = float(np.max(intervals)) if intervals.size else 0.0
    coverage_fraction = (
        len(candidates) / span_epoch_count if span_epoch_count > 0 else 0.0
    )

    return TrackDiagnostics(
        duration_s=duration_s,
        displacement_px=float(displacement),
        mean_speed_px_per_minute=float(speed),
        fit_rms_px=fit_rms,
        missed_frames=missed_frames,
        diagnostic_class="candidate",
        detection_count=len(candidates),
        span_epoch_count=span_epoch_count,
        coverage_fraction=float(coverage_fraction),
        median_interval_s=median_interval_s,
        max_interval_s=max_interval_s,
    )
