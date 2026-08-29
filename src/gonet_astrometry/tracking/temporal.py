"""Temporal diagnostics for irregularly sampled tracking sequences."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gonet_astrometry.tracking.image_plane import ImagePlaneTrackingResult
from gonet_astrometry.tracking.sequence import DetectionSequence


@dataclass(frozen=True, slots=True)
class SequenceTimingDiagnostics:
    """Cadence statistics measured directly from exposure midpoints."""

    epoch_count: int
    duration_s: float
    min_interval_s: float
    median_interval_s: float
    p90_interval_s: float
    max_interval_s: float


@dataclass(frozen=True, slots=True)
class TrackPopulationDiagnostics:
    """Track-length and temporal-coverage statistics for one tracking result."""

    track_count: int
    min_length: int
    median_length: float
    p90_length: float
    max_length: int
    median_duration_s: float
    p90_duration_s: float
    max_duration_s: float
    median_coverage_fraction: float


def sequence_timing_diagnostics(
    sequence: DetectionSequence,
) -> SequenceTimingDiagnostics:
    """Summarize the actual intervals between chronological exposure midpoints."""
    intervals = np.asarray(sequence.epoch_intervals_seconds, dtype=np.float64)
    if intervals.size == 0:
        return SequenceTimingDiagnostics(
            epoch_count=len(sequence.epochs),
            duration_s=sequence.duration_seconds,
            min_interval_s=0.0,
            median_interval_s=0.0,
            p90_interval_s=0.0,
            max_interval_s=0.0,
        )
    return SequenceTimingDiagnostics(
        epoch_count=len(sequence.epochs),
        duration_s=sequence.duration_seconds,
        min_interval_s=float(np.min(intervals)),
        median_interval_s=float(np.median(intervals)),
        p90_interval_s=float(np.percentile(intervals, 90.0)),
        max_interval_s=float(np.max(intervals)),
    )


def track_population_diagnostics(
    result: ImagePlaneTrackingResult,
) -> TrackPopulationDiagnostics:
    """Summarize track fragmentation and temporal coverage."""
    if not result.tracks:
        return TrackPopulationDiagnostics(0, 0, 0.0, 0.0, 0, 0.0, 0.0, 0.0, 0.0)

    lengths = np.asarray([len(track.points) for track in result.tracks], dtype=float)
    diagnostics = [track.diagnostics for track in result.tracks]
    durations = np.asarray(
        [item.duration_s for item in diagnostics if item is not None], dtype=float
    )
    coverage = np.asarray(
        [item.coverage_fraction for item in diagnostics if item is not None],
        dtype=float,
    )
    return TrackPopulationDiagnostics(
        track_count=len(result.tracks),
        min_length=int(np.min(lengths)),
        median_length=float(np.median(lengths)),
        p90_length=float(np.percentile(lengths, 90.0)),
        max_length=int(np.max(lengths)),
        median_duration_s=float(np.median(durations)) if durations.size else 0.0,
        p90_duration_s=(
            float(np.percentile(durations, 90.0)) if durations.size else 0.0
        ),
        max_duration_s=float(np.max(durations)) if durations.size else 0.0,
        median_coverage_fraction=(float(np.median(coverage)) if coverage.size else 0.0),
    )
