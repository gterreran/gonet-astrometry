"""Timing records for source-detection runs."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DetectionTiming:
    """Wall-clock timing for one source-detection run.

    Parameters
    ----------
    frame_identifier
        Stable identifier of the processed image frame.
    detector_name
        Stable name reported by the detector backend.
    frame_load_seconds
        Time spent loading or retrieving the scientific frame.
    detector_setup_seconds
        Time spent constructing the selected detector backend.
    preprocessing_seconds
        Time spent building the shared mask-aware Bayer significance image.
    backend_seconds
        Time spent in the backend-specific detection algorithm after shared
        preprocessing is complete.
    total_seconds
        End-to-end elapsed time from requesting the frame through receiving the
        final detection catalog.
    source_count
        Number of source candidates returned by the backend.

    Raises
    ------
    ValueError
        If a duration or source count is negative, or if ``total_seconds`` is
        shorter than the sum of its measured components.
    """

    frame_identifier: str
    detector_name: str
    frame_load_seconds: float
    detector_setup_seconds: float
    preprocessing_seconds: float
    backend_seconds: float
    total_seconds: float
    source_count: int

    def __post_init__(self) -> None:
        durations = (
            self.frame_load_seconds,
            self.detector_setup_seconds,
            self.preprocessing_seconds,
            self.backend_seconds,
            self.total_seconds,
        )
        if any(duration < 0 for duration in durations):
            raise ValueError("Detection durations cannot be negative")
        if self.source_count < 0:
            raise ValueError("Detection source count cannot be negative")
        measured = (
            self.frame_load_seconds
            + self.detector_setup_seconds
            + self.preprocessing_seconds
            + self.backend_seconds
        )
        tolerance = max(1e-12, self.total_seconds * 1e-9)
        if measured > self.total_seconds + tolerance:
            raise ValueError(
                "Detection total time cannot be shorter than its measured components"
            )

    @property
    def detector_seconds(self) -> float:
        """Return shared preprocessing plus backend execution time.

        This compatibility property preserves the original aggregate detector
        duration while the backend-specific duration is available separately.
        """
        return self.preprocessing_seconds + self.backend_seconds

    @property
    def sources_per_second(self) -> float | None:
        """Return backend throughput, or ``None`` for a zero-duration run."""
        if self.backend_seconds <= 0:
            return None
        return self.source_count / self.backend_seconds
