"""Static PDF diagnostics for command-line detection and tracking runs."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from numpy.typing import NDArray
from scipy.ndimage import binary_erosion

from gonet_astrometry.adapters.gonet_wizard import GONetChannel
from gonet_astrometry.detection.config import DetectionConfig
from gonet_astrometry.models.detection import DetectionClass
from gonet_astrometry.models.track import TrackClass
from gonet_astrometry.tracking.config import TrackingConfig
from gonet_astrometry.tracking.temporal import (
    sequence_timing_diagnostics,
    track_population_diagnostics,
)

if TYPE_CHECKING:
    from gonet_astrometry.runner import TrackingRunArtifacts

_DETECTION_STYLES: dict[DetectionClass, tuple[str, str, str]] = {
    "compact": ("Compact", "#f97316", "o"),
    "elongated": ("Elongated", "#facc15", "D"),
    "extended": ("Extended", "#e879f9", "s"),
    "mask-adjacent": ("Mask-adjacent", "#22d3ee", "x"),
    "backend-flagged": ("Backend-flagged", "#ef4444", "x"),
    "unclassified": ("Unclassified", "#d1d5db", "o"),
}
_TRACK_STYLES: dict[TrackClass, tuple[str, str, str]] = {
    "candidate": ("Candidate tracks", "#34d399", "o"),
    "low-motion": ("Low-motion tracks", "#94a3b8", "D"),
    "poor-fit": ("Poor-fit tracks", "#fb7185", "x"),
}


def write_tracking_report_pdf(
    output_path: Path,
    *,
    image_data: NDArray[np.float64],
    channel: GONetChannel,
    artifacts: TrackingRunArtifacts,
    detection_config: DetectionConfig,
    tracking_config: TrackingConfig,
) -> Path:
    """Write one static PDF figure for a detection and tracking run.

    The report mirrors the portal view: one compact native channel forms the
    grayscale background, the usable field and dynamic bright-region masks are
    outlined, detections from the reference image are marked, and complete
    bootstrap track histories are drawn in the same compact display coordinates.

    Parameters
    ----------
    output_path
        Destination PDF path. Parent directories are created automatically.
    image_data
        Compact native channel used as the background image.
    channel
        Native Bayer channel represented by ``image_data``.
    artifacts
        Scientific detections, masks, and tracking products from the run.
    detection_config
        Detection settings recorded in the figure annotation.
    tracking_config
        Tracking settings recorded in the figure annotation.

    Returns
    -------
    pathlib.Path
        Absolute path of the written PDF.
    """
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)

    figure = Figure(figsize=(12.0, 8.5))
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    lower, upper = _display_limits(image_data)
    axis.imshow(
        image_data,
        cmap="gray",
        origin="upper",
        vmin=lower,
        vmax=upper,
        interpolation="nearest",
    )

    prepared = artifacts.reference_prepared
    _draw_mask_boundary(
        axis,
        _compact_field_mask(prepared.field_mask),
        color="#22d3ee",
        label="Usable field boundary",
    )
    _draw_mask_boundary(
        axis,
        _compact_dynamic_mask(prepared.dynamic_mask),
        color="#f472b6",
        label="Bright-region mask",
    )
    _draw_detections(axis, artifacts)
    _draw_tracks(axis, artifacts)

    result = artifacts.tracking_result
    axis.set_title(
        f"{artifacts.reference_path.name} - {channel} - {artifacts.algorithm} | "
        f"{len(artifacts.reference_catalog)} detections | {len(result.tracks)} tracks",
        fontsize=10,
    )
    axis.set_xlabel("Compact channel column")
    axis.set_ylabel("Compact channel row")
    axis.set_xlim(-0.5, image_data.shape[1] - 0.5)
    axis.set_ylim(image_data.shape[0] - 0.5, -0.5)
    axis.set_aspect("equal")

    figure.suptitle("GONet Astrometry Tracking Diagnostic", fontsize=14)
    figure.text(
        0.01,
        0.015,
        _parameter_summary(artifacts, detection_config, tracking_config),
        ha="left",
        va="bottom",
        fontsize=7,
        family="monospace",
    )
    handles, labels = axis.get_legend_handles_labels()
    if handles:
        axis.legend(
            handles,
            labels,
            loc="center left",
            bbox_to_anchor=(1.01, 0.5),
            fontsize=7,
            frameon=False,
        )
    figure.subplots_adjust(left=0.07, right=0.78, top=0.90, bottom=0.11)
    temporal_figure = _build_temporal_diagnostic_figure(artifacts)
    with PdfPages(destination) as pdf:
        pdf.savefig(figure)
        pdf.savefig(temporal_figure)
    return destination



def _build_temporal_diagnostic_figure(
    artifacts: TrackingRunArtifacts,
) -> Figure:
    """Return a second report page describing cadence and track fragmentation."""
    figure = Figure(figsize=(12.0, 8.5))
    FigureCanvasAgg(figure)
    axes = figure.subplots(2, 2)
    sequence = artifacts.sequence
    result = artifacts.tracking_result
    intervals = np.asarray(sequence.epoch_intervals_seconds, dtype=np.float64)
    lengths = np.asarray(
        [len(track.points) for track in result.tracks], dtype=np.float64
    )
    durations = np.asarray(
        [
            track.diagnostics.duration_s / 60.0
            for track in result.tracks
            if track.diagnostics is not None
        ],
        dtype=np.float64,
    )
    coverage = np.asarray(
        [
            100.0 * track.diagnostics.coverage_fraction
            for track in result.tracks
            if track.diagnostics is not None
        ],
        dtype=np.float64,
    )

    axes[0, 0].hist(intervals, bins=min(60, max(10, int(np.sqrt(intervals.size)))))
    axes[0, 0].set_title("Consecutive exposure intervals")
    axes[0, 0].set_xlabel("Interval (s)")
    axes[0, 0].set_ylabel("Count")

    if lengths.size:
        axes[0, 1].hist(lengths, bins=min(60, max(10, int(np.sqrt(lengths.size)))))
    axes[0, 1].set_title("Track lengths")
    axes[0, 1].set_xlabel("Detections per track")
    axes[0, 1].set_ylabel("Count")

    if durations.size:
        axes[1, 0].hist(
            durations, bins=min(60, max(10, int(np.sqrt(durations.size))))
        )
    axes[1, 0].set_title("Track temporal spans")
    axes[1, 0].set_xlabel("Duration (min)")
    axes[1, 0].set_ylabel("Count")

    if coverage.size:
        axes[1, 1].hist(coverage, bins=20, range=(0.0, 100.0))
    axes[1, 1].set_title("Track epoch coverage")
    axes[1, 1].set_xlabel("Detections / epochs in track span (%)")
    axes[1, 1].set_ylabel("Count")

    cadence = sequence_timing_diagnostics(sequence)
    population = track_population_diagnostics(result)
    figure.suptitle("GONet Astrometry Temporal Diagnostics", fontsize=14)
    figure.text(
        0.01,
        0.015,
        (
            f"sequence: {cadence.epoch_count} images over "
            f"{cadence.duration_s / 60.0:.1f} min | interval min/median/p90/max "
            f"{cadence.min_interval_s:.1f}/{cadence.median_interval_s:.1f}/"
            f"{cadence.p90_interval_s:.1f}/{cadence.max_interval_s:.1f} s\n"
            f"tracks: {population.track_count} | length min/median/p90/max "
            f"{population.min_length}/{population.median_length:.1f}/"
            f"{population.p90_length:.1f}/{population.max_length} | "
            f"duration median/p90/max "
            f"{population.median_duration_s / 60.0:.1f}/"
            f"{population.p90_duration_s / 60.0:.1f}/"
            f"{population.max_duration_s / 60.0:.1f} min | "
            f"median epoch coverage {100.0 * population.median_coverage_fraction:.1f}%"
        ),
        ha="left",
        va="bottom",
        fontsize=8,
        family="monospace",
    )
    figure.subplots_adjust(left=0.08, right=0.97, top=0.90, bottom=0.13, hspace=0.35)
    return figure


def _draw_detections(axis: object, artifacts: TrackingRunArtifacts) -> None:
    """Draw reference-image detections grouped by diagnostic class."""
    catalog = artifacts.reference_catalog
    for source_class, (label, color, marker) in _DETECTION_STYLES.items():
        detections = [
            detection
            for detection in catalog.detections
            if detection.diagnostic_class == source_class
        ]
        if not detections:
            continue
        axis.scatter(  # type: ignore[attr-defined]
            [detection.x / 2.0 for detection in detections],
            [detection.y / 2.0 for detection in detections],
            s=28,
            marker=marker,
            facecolors="none" if marker not in {"x"} else color,
            edgecolors=color,
            linewidths=0.8,
            label=f"{label} detections ({len(detections)})",
        )


def _draw_tracks(axis: object, artifacts: TrackingRunArtifacts) -> None:
    """Draw complete image-plane histories grouped by track class."""
    result = artifacts.tracking_result
    grouped: dict[TrackClass, list[object]] = {
        track_class: [] for track_class in _TRACK_STYLES
    }
    for track in result.tracks:
        grouped[track.diagnostic_class].append(track)

    for track_class, tracks in grouped.items():
        if not tracks:
            continue
        label, color, marker = _TRACK_STYLES[track_class]
        first = True
        for track in tracks:
            resolved = result.resolve(track)  # type: ignore[arg-type]
            axis.plot(  # type: ignore[attr-defined]
                [point.detection.x / 2.0 for point in resolved],
                [point.detection.y / 2.0 for point in resolved],
                color=color,
                marker=marker,
                markersize=2.5,
                linewidth=0.8,
                alpha=0.9,
                label=f"{label} ({len(tracks)})" if first else None,
            )
            first = False


def _draw_mask_boundary(
    axis: object,
    mask: NDArray[np.bool_],
    *,
    color: str,
    label: str,
) -> None:
    """Draw a compact-coordinate mask boundary when it is nontrivial."""
    if not np.any(mask) or np.all(mask):
        return
    boundary = mask & ~binary_erosion(mask, border_value=0)
    rows, columns = np.nonzero(boundary)
    if rows.size == 0:
        return
    step = max(1, rows.size // 6000)
    axis.scatter(  # type: ignore[attr-defined]
        columns[::step],
        rows[::step],
        s=1.5,
        color=color,
        marker=".",
        linewidths=0,
        label=label,
    )


def _compact_field_mask(mask: NDArray[np.bool_]) -> NDArray[np.bool_]:
    """Require every full-sensor sample in a compact Bayer quad to be usable."""
    height = mask.shape[0] // 2
    width = mask.shape[1] // 2
    return np.asarray(
        mask[0 : 2 * height : 2, 0 : 2 * width : 2]
        & mask[0 : 2 * height : 2, 1 : 2 * width : 2]
        & mask[1 : 2 * height : 2, 0 : 2 * width : 2]
        & mask[1 : 2 * height : 2, 1 : 2 * width : 2],
        dtype=np.bool_,
    )


def _compact_dynamic_mask(mask: NDArray[np.bool_]) -> NDArray[np.bool_]:
    """Mark a compact Bayer quad if any full-sensor sample is dynamically masked."""
    height = mask.shape[0] // 2
    width = mask.shape[1] // 2
    return np.asarray(
        mask[0 : 2 * height : 2, 0 : 2 * width : 2]
        | mask[0 : 2 * height : 2, 1 : 2 * width : 2]
        | mask[1 : 2 * height : 2, 0 : 2 * width : 2]
        | mask[1 : 2 * height : 2, 1 : 2 * width : 2],
        dtype=np.bool_,
    )


def _display_limits(data: NDArray[np.float64]) -> tuple[float, float]:
    """Return robust grayscale limits without modifying channel values."""
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return 0.0, 1.0
    lower, upper = np.percentile(finite, (1.0, 99.5))
    if lower == upper:
        upper = lower + 1.0
    return float(lower), float(upper)


def _parameter_summary(
    artifacts: TrackingRunArtifacts,
    detection: DetectionConfig,
    tracking: TrackingConfig,
) -> str:
    """Return the compact parameter annotation printed below the figure."""
    return (
        f"images={len(artifacts.files)}  threshold={detection.threshold_sigma:g}sigma  "
        f"fwhm={detection.fwhm_px:g}px  min_pixels={detection.min_pixels}  "
        f"max_speed={tracking.max_speed_px_per_minute:g}px/min  "
        f"prediction_tol={tracking.prediction_tolerance_px:g}px  "
        f"max_gap={tracking.max_gap_minutes:g}min  "
        f"frame_gap={tracking.max_gap_frames}  "
        f"gap_scale={tracking.max_prediction_gap_scale:g}  "
        f"min_track={tracking.min_track_length}"
    )
