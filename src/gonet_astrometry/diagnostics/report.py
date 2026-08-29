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
from gonet_astrometry.adapters.grid_calibration import grid_pole_pixel
from gonet_astrometry.detection.config import DetectionConfig
from gonet_astrometry.models.detection import DetectionClass
from gonet_astrometry.models.track import TrackClass
from gonet_astrometry.solving.sidereal import SIDEREAL_RATE_RAD_PER_SECOND
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

_SIDEREAL_TRACK_STYLES: dict[str, tuple[str, str]] = {
    "sidereal-consistent": ("Sidereal-consistent", "#22c55e"),
    "sidereal-rejected": ("Sidereal-rejected", "#ef4444"),
    "insufficient": ("Insufficient", "#94a3b8"),
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
    sidereal_figure = (
        _build_sidereal_diagnostic_figure(artifacts, image_data)
        if artifacts.sidereal_solution is not None
        and artifacts.grid_calibration is not None
        else None
    )
    orientation_figure = (
        _build_orientation_diagnostic_figure(artifacts, image_data)
        if artifacts.orientation_solution is not None
        and artifacts.grid_calibration is not None
        else None
    )
    with PdfPages(destination) as pdf:
        pdf.savefig(figure)
        pdf.savefig(temporal_figure)
        if sidereal_figure is not None:
            pdf.savefig(sidereal_figure)
        if orientation_figure is not None:
            pdf.savefig(orientation_figure)
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
        axes[1, 0].hist(durations, bins=min(60, max(10, int(np.sqrt(durations.size)))))
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


def _build_sidereal_diagnostic_figure(
    artifacts: TrackingRunArtifacts,
    image_data: NDArray[np.float64],
) -> Figure:
    """Return a report page for the shared fixed-rate sidereal rotation fit."""
    solution = artifacts.sidereal_solution
    calibration = artifacts.grid_calibration
    if solution is None or calibration is None:  # pragma: no cover - caller guards
        raise ValueError("Sidereal diagnostics require a solution and Grid calibration")

    figure = Figure(figsize=(12.0, 8.5))
    FigureCanvasAgg(figure)
    axes = figure.subplots(2, 2)
    image_axis = axes[0, 0]
    lower, upper = _display_limits(image_data)
    image_axis.imshow(
        image_data,
        cmap="gray",
        origin="upper",
        vmin=lower,
        vmax=upper,
        interpolation="nearest",
    )
    diagnostics = {item.track_identifier: item for item in solution.track_diagnostics}
    for class_name, (label, color) in _SIDEREAL_TRACK_STYLES.items():
        tracks = [
            track
            for track in artifacts.tracking_result.tracks
            if diagnostics[track.identifier].diagnostic_class == class_name
        ]
        first = True
        for track in tracks:
            resolved = artifacts.tracking_result.resolve(track)
            image_axis.plot(
                [point.detection.x / 2.0 for point in resolved],
                [point.detection.y / 2.0 for point in resolved],
                color=color,
                linewidth=0.6,
                alpha=0.75 if class_name != "insufficient" else 0.25,
                label=f"{label} ({len(tracks)})" if first else None,
            )
            first = False
    pole_pixel = grid_pole_pixel(calibration, solution.axis_grid)
    if pole_pixel is not None:
        image_axis.scatter(
            [pole_pixel[0] / 2.0],
            [pole_pixel[1] / 2.0],
            s=100,
            marker="*",
            facecolors="#facc15",
            edgecolors="black",
            linewidths=0.8,
            label="Fitted apparent rotation pole",
            zorder=20,
        )
    image_axis.set_title("Track consistency in image coordinates")
    image_axis.set_xlabel("Compact channel column")
    image_axis.set_ylabel("Compact channel row")
    image_axis.set_xlim(-0.5, image_data.shape[1] - 0.5)
    image_axis.set_ylim(image_data.shape[0] - 0.5, -0.5)
    image_axis.set_aspect("equal")
    handles, labels = image_axis.get_legend_handles_labels()
    if handles:
        image_axis.legend(handles, labels, fontsize=6, frameon=False, loc="best")

    sufficient = [
        item for item in solution.track_diagnostics if item.rms_residual_deg is not None
    ]
    rms_arcmin = np.asarray(
        [60.0 * item.rms_residual_deg for item in sufficient],
        dtype=np.float64,
    )
    if rms_arcmin.size:
        axes[0, 1].hist(
            rms_arcmin,
            bins=min(60, max(10, int(np.sqrt(rms_arcmin.size)))),
        )
    threshold_arcmin = (
        60.0 * artifacts.sidereal_fit_config.consistent_rms_deg
        if artifacts.sidereal_fit_config is not None
        else np.nan
    )
    if np.isfinite(threshold_arcmin):
        axes[0, 1].axvline(
            threshold_arcmin,
            linestyle="--",
            linewidth=1.0,
            label=f"Consistency threshold ({threshold_arcmin:.1f} arcmin)",
        )
        axes[0, 1].legend(fontsize=7, frameon=False)
    axes[0, 1].set_title("Per-track sidereal residuals")
    axes[0, 1].set_xlabel("De-rotated RMS (arcmin)")
    axes[0, 1].set_ylabel("Count")

    duration_minutes = np.asarray(
        [item.duration_s / 60.0 for item in sufficient], dtype=np.float64
    )
    class_colors = [
        _SIDEREAL_TRACK_STYLES[item.diagnostic_class][1] for item in sufficient
    ]
    if rms_arcmin.size:
        axes[1, 0].scatter(
            duration_minutes,
            rms_arcmin,
            s=8,
            c=class_colors,
            alpha=0.6,
            linewidths=0,
        )
    axes[1, 0].set_title("Residual versus track duration")
    axes[1, 0].set_xlabel("Track duration (min)")
    axes[1, 0].set_ylabel("De-rotated RMS (arcmin)")

    counts = solution.diagnostic_counts()
    labels = ["Consistent", "Rejected", "Insufficient"]
    values = [
        counts["sidereal-consistent"],
        counts["sidereal-rejected"],
        counts["insufficient"],
    ]
    colors = ["#22c55e", "#ef4444", "#94a3b8"]
    axes[1, 1].bar(labels, values, color=colors)
    axes[1, 1].set_title("Spherical track classification")
    axes[1, 1].set_ylabel("Tracks")

    figure.suptitle("GONet Astrometry Sidereal Rotation Diagnostics", fontsize=14)
    figure.text(
        0.01,
        0.015,
        (
            f"axis Grid-frame r/theta = {solution.axis_r_deg:.4f}/"
            f"{solution.axis_theta_deg:.4f} deg | rotation sign "
            f"{solution.rotation_sign:+d} | vector "
            f"[{solution.axis_grid[0]:+.6f}, {solution.axis_grid[1]:+.6f}, "
            f"{solution.axis_grid[2]:+.6f}]\n"
            f"global fit RMS/median/P95 = "
            f"{60.0 * solution.fit_rms_deg:.3f}/"
            f"{60.0 * solution.fit_median_deg:.3f}/"
            f"{60.0 * solution.fit_p95_deg:.3f} arcmin | "
            f"fitted tracks={solution.fitted_track_count} | "
            f"fitted points={solution.fitted_point_count}"
        ),
        ha="left",
        va="bottom",
        fontsize=8,
        family="monospace",
    )
    figure.subplots_adjust(
        left=0.07, right=0.97, top=0.90, bottom=0.13, hspace=0.34, wspace=0.26
    )
    return figure


def _build_orientation_diagnostic_figure(
    artifacts: TrackingRunArtifacts,
    image_data: NDArray[np.float64],
) -> Figure:
    """Return a report page for the catalog-assisted absolute camera attitude."""
    solution = artifacts.orientation_solution
    calibration = artifacts.grid_calibration
    if solution is None or calibration is None:  # pragma: no cover - caller guards
        raise ValueError("Orientation diagnostics require a solution and calibration")

    figure = Figure(figsize=(12.0, 8.5))
    FigureCanvasAgg(figure)
    axes = figure.subplots(2, 2)
    image_axis = axes[0, 0]
    lower, upper = _display_limits(image_data)
    image_axis.imshow(
        image_data,
        cmap="gray",
        origin="upper",
        vmin=lower,
        vmax=upper,
        interpolation="nearest",
    )

    matched_pixels: list[tuple[float, float]] = []
    report_timestamp = _reference_image_timestamp(artifacts)
    signed_axis: NDArray[np.float64] | None = None
    angle = 0.0
    if artifacts.sidereal_solution is not None:
        signed_axis = float(artifacts.sidereal_solution.rotation_sign) * np.asarray(
            artifacts.sidereal_solution.axis_grid,
            dtype=np.float64,
        )
        angle = SIDEREAL_RATE_RAD_PER_SECOND * (
            report_timestamp - solution.reference_time.timestamp()
        )
    for match in solution.matches:
        display_ray = (
            _rotate_vector(match.ray_grid, signed_axis, angle)
            if signed_axis is not None
            else match.ray_grid
        )
        pixel = grid_pole_pixel(calibration, display_ray)
        if pixel is not None:
            matched_pixels.append((pixel[0] / 2.0, pixel[1] / 2.0))
    if matched_pixels:
        image_axis.scatter(
            [item[0] for item in matched_pixels],
            [item[1] for item in matched_pixels],
            s=24,
            marker="o",
            facecolors="none",
            edgecolors="#22c55e",
            linewidths=0.9,
            label=f"Catalog matches ({len(matched_pixels)})",
        )

    direction_styles = (
        ("NCP", solution.ncp_grid, "*", "#facc15"),
        (
            "Zenith",
            solution.local_direction_in_grid(
                np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
            ),
            "P",
            "#38bdf8",
        ),
        (
            "North horizon",
            solution.local_direction_in_grid(
                np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
            ),
            "^",
            "#fb7185",
        ),
        (
            "East horizon",
            solution.local_direction_in_grid(
                np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
            ),
            ">",
            "#a78bfa",
        ),
    )
    for label, ray, marker, color in direction_styles:
        pixel = grid_pole_pixel(calibration, ray)
        if pixel is None:
            continue
        image_axis.scatter(
            [pixel[0] / 2.0],
            [pixel[1] / 2.0],
            s=100,
            marker=marker,
            facecolors=color,
            edgecolors="black",
            linewidths=0.8,
            label=label,
            zorder=20,
        )
    image_axis.set_title("Absolute orientation in image coordinates")
    image_axis.set_xlabel("Compact channel column")
    image_axis.set_ylabel("Compact channel row")
    image_axis.set_xlim(-0.5, image_data.shape[1] - 0.5)
    image_axis.set_ylim(image_data.shape[0] - 0.5, -0.5)
    image_axis.set_aspect("equal")
    handles, labels = image_axis.get_legend_handles_labels()
    if handles:
        image_axis.legend(handles, labels, fontsize=6, frameon=False, loc="best")

    residual_arcmin = np.asarray(
        [60.0 * match.residual_deg for match in solution.matches],
        dtype=np.float64,
    )
    axes[0, 1].hist(
        residual_arcmin,
        bins=min(30, max(8, int(np.sqrt(max(1, residual_arcmin.size))))),
    )
    axes[0, 1].set_title("Catalog-match residuals")
    axes[0, 1].set_xlabel("Angular residual (arcmin)")
    axes[0, 1].set_ylabel("Matches")

    magnitudes = np.asarray(
        [
            np.nan if match.catalog_magnitude is None else match.catalog_magnitude
            for match in solution.matches
        ],
        dtype=np.float64,
    )
    valid_magnitude = np.isfinite(magnitudes)
    if np.any(valid_magnitude):
        axes[1, 0].scatter(
            magnitudes[valid_magnitude],
            residual_arcmin[valid_magnitude],
            s=18,
            alpha=0.7,
            linewidths=0,
        )
    axes[1, 0].set_title("Residual versus catalog magnitude")
    axes[1, 0].set_xlabel("V magnitude")
    axes[1, 0].set_ylabel("Angular residual (arcmin)")

    matrix_axis = axes[1, 1]
    matrix_axis.axis("off")
    matrix_lines = [
        "Grid -> ENU rotation matrix",
        *[
            "[" + "  ".join(f"{value:+.7f}" for value in row) + "]"
            for row in solution.grid_to_enu
        ],
        "",
        f"reference UTC: {solution.reference_time.isoformat()}",
        (
            f"site: lat {solution.location.latitude_deg:.6f} deg, "
            f"lon {solution.location.longitude_deg:.6f} deg, "
            f"elev {solution.location.elevation_m:.1f} m"
        ),
        f"pole-axis twist: {solution.twist_deg:.4f} deg",
        (
            f"optical axis: az {solution.optical_axis_azimuth_deg:.4f} deg, "
            f"alt {solution.optical_axis_altitude_deg:.4f} deg"
        ),
        f"unique observed anchors: {solution.anchor_count}",
        f"visible bright catalog stars: {solution.catalog_star_count}",
        f"accepted catalog matches: {len(solution.matches)}",
    ]
    matrix_axis.text(
        0.02,
        0.98,
        "\n".join(matrix_lines),
        ha="left",
        va="top",
        fontsize=9,
        family="monospace",
        transform=matrix_axis.transAxes,
    )

    figure.suptitle("GONet Astrometry Absolute Orientation Diagnostics", fontsize=14)
    figure.text(
        0.01,
        0.015,
        (
            f"catalog matches={len(solution.matches)} | fit RMS/median/P95 "
            f"{60.0 * solution.fit_rms_deg:.3f}/"
            f"{60.0 * solution.fit_median_deg:.3f}/"
            f"{60.0 * solution.fit_p95_deg:.3f} arcmin | "
            f"optical-axis az/alt "
            f"{solution.optical_axis_azimuth_deg:.4f}/"
            f"{solution.optical_axis_altitude_deg:.4f} deg"
        ),
        ha="left",
        va="bottom",
        fontsize=8,
        family="monospace",
    )
    figure.subplots_adjust(
        left=0.07, right=0.97, top=0.90, bottom=0.13, hspace=0.34, wspace=0.26
    )
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


def _reference_image_timestamp(artifacts: TrackingRunArtifacts) -> float:
    """Return the exposure midpoint timestamp of the report background image."""
    reference = artifacts.reference_path.resolve()
    for epoch in artifacts.sequence.epochs:
        if epoch.source_path.resolve() == reference:
            return epoch.exposure_midpoint.timestamp()
    raise ValueError("Reference image is not part of the detection sequence")


def _rotate_vector(
    vector: NDArray[np.float64],
    axis: NDArray[np.float64],
    angle: float,
) -> NDArray[np.float64]:
    """Rotate ``vector`` about ``axis`` by ``angle`` radians."""
    cosine = float(np.cos(angle))
    sine = float(np.sin(angle))
    return np.asarray(
        vector * cosine
        + np.cross(axis, vector) * sine
        + axis * float(np.dot(vector, axis)) * (1.0 - cosine),
        dtype=np.float64,
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
    spherical = artifacts.spherical_tracking_config
    multichannel = artifacts.multichannel_detection_config
    merge = artifacts.stellar_merge_config
    if spherical is not None and multichannel is not None and merge is not None:
        edge_setting = (
            f"auto:{detection.footprint_threshold_fraction:g}"
            if detection.use_provisional_field_mask
            else "off"
        )
        grid_search = (
            "none"
            if multichannel.grid_search_radius_deg is None
            else f"{multichannel.grid_search_radius_deg:g}deg"
        )
        grid_accept = (
            "none"
            if multichannel.grid_acceptance_radius_deg is None
            else f"{multichannel.grid_acceptance_radius_deg:g}deg"
        )
        return (
            f"images={len(artifacts.files)}  "
            f"threshold={detection.threshold_sigma:g}sigma  "
            f"edge={edge_setting}  "
            f"edge_keep={multichannel.field_edge_keep_margin_px:g}px  "
            f"grid_search={grid_search}  grid_accept={grid_accept}  "
            f"channel_match={multichannel.channel_match_radius_px:g}px  "
            f"initial_speed={spherical.max_initial_speed_deg_per_minute:g}deg/min  "
            f"prediction_tol={spherical.prediction_tolerance_arcmin:g}arcmin  "
            f"max_gap={spherical.max_gap_minutes:g}min  "
            f"min_track={spherical.min_track_length}  "
            f"merge={60.0 * merge.merge_radius_deg:g}arcmin  "
            f"merged_rms={60.0 * merge.merged_rms_deg:g}arcmin"
        )

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
