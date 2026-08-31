#!/usr/bin/env python3
"""Compare portable Grid and direct stellar camera calibrations geometrically.

The two calibrations do not share an intrinsic reference frame: the Grid model
uses the fitted printed-grid frame, while the stellar model uses a sensor-aligned
camera frame plus an absolute camera-to-ENU attitude.  This diagnostic therefore
samples common sensor positions, converts them to unit rays with both models,
and first removes the best-fitting *rigid* rotation between the two ray fields.
Any remaining disagreement is a non-rigid geometric difference between the
Grid-derived mapping and the stellar camera mapping.

The script writes a multi-page PDF, two PNG figures suitable for documentation,
a CSV with every retained sample, and a JSON summary of the comparison.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from gonet_astrometry.adapters.grid_calibration import (
    load_grid_calibration,
    validated_pixel_rays,
)
from gonet_astrometry.calibration.stellar_camera import StellarCameraCalibration
from gonet_astrometry.products.stellar_camera_calibration import (
    load_stellar_camera_calibration_product,
)

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class ComparisonSamples:
    """Common sensor samples and Grid-minus-stellar geometric discrepancies."""

    x_px: FloatArray
    y_px: FloatArray
    theta_deg: FloatArray
    grid_rays: FloatArray
    stellar_rays: FloatArray
    aligned_grid_rays: FloatArray
    angular_residual_arcmin: FloatArray
    radial_residual_arcmin: FloatArray
    tangential_residual_arcmin: FloatArray
    equivalent_dx_px: FloatArray
    equivalent_dy_px: FloatArray
    equivalent_pixel_residual_px: FloatArray


@dataclass(frozen=True, slots=True)
class ComparisonResult:
    """Full Grid-versus-stellar comparison and fitted rigid alignment."""

    samples: ComparisonSamples
    grid_to_camera: FloatArray
    rigid_rotation_angle_deg: float
    rigid_rotation_axis_grid: FloatArray


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--grid-calibration",
        type=Path,
        required=True,
        help="Portable Grid calibration artifact.",
    )
    parser.add_argument(
        "--stellar-calibration",
        type=Path,
        required=True,
        help="Portable stellar_camera_calibration.npz artifact.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory receiving PDF/PNG/CSV/JSON diagnostics.",
    )
    parser.add_argument(
        "--spacing-px",
        type=float,
        default=40.0,
        help="Sensor sampling spacing in pixels (default: 40).",
    )
    parser.add_argument(
        "--profile-bin-width-deg",
        type=float,
        default=5.0,
        help="Field-angle bin width for the residual profile (default: 5 deg).",
    )
    parser.add_argument(
        "--quiver-spacing-px",
        type=float,
        default=160.0,
        help="Approximate spacing of vector samples in the map (default: 160).",
    )
    return parser.parse_args()


def _sample_sensor(
    image_shape: tuple[int, int],
    spacing_px: float,
) -> tuple[FloatArray, FloatArray]:
    """Return a regular full-sensor sampling grid including near-edge points."""
    if spacing_px <= 0.0 or not np.isfinite(spacing_px):
        raise ValueError("spacing_px must be finite and positive")
    rows, columns = image_shape
    x = np.arange(spacing_px / 2.0, columns, spacing_px, dtype=np.float64)
    y = np.arange(spacing_px / 2.0, rows, spacing_px, dtype=np.float64)
    xx, yy = np.meshgrid(x, y)
    return xx.ravel(), yy.ravel()


def _stellar_calibrated_mask(
    calibration: StellarCameraCalibration,
    x_px: FloatArray,
    y_px: FloatArray,
) -> BoolArray:
    """Select pixels inside the stellar model's empirically calibrated radius."""
    t_max = np.deg2rad(calibration.calibrated_theta_max_deg) / (np.pi / 2.0)
    radius_max = (
        calibration.radial_c1_px * t_max + calibration.radial_c3_px * t_max**3
    )
    radius = np.hypot(
        x_px - calibration.center_x_px,
        y_px - calibration.center_y_px,
    )
    return np.asarray(radius <= radius_max + 1e-9, dtype=np.bool_)


def _fit_rigid_alignment(
    grid_rays: FloatArray,
    stellar_rays: FloatArray,
) -> tuple[FloatArray, float, FloatArray]:
    """Fit the proper rotation that maps Grid-frame rays to camera-frame rays."""
    rotation, _ = Rotation.align_vectors(stellar_rays, grid_rays)
    matrix = np.asarray(rotation.as_matrix(), dtype=np.float64)
    rotvec = np.asarray(rotation.as_rotvec(), dtype=np.float64)
    angle = float(np.linalg.norm(rotvec))
    if angle > 1e-15:
        axis = rotvec / angle
    else:
        axis = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    return matrix, float(np.rad2deg(angle)), axis


def _spherical_log_components(
    reference_rays: FloatArray,
    comparison_rays: FloatArray,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Return total, radial, and tangential angular offsets on the unit sphere."""
    dot = np.sum(reference_rays * comparison_rays, axis=1)
    dot = np.clip(dot, -1.0, 1.0)
    angle = np.arccos(dot)

    tangent = np.zeros_like(reference_rays)
    sin_angle = np.sin(angle)
    usable = sin_angle > 1e-14
    if np.any(usable):
        direction = (
            comparison_rays[usable]
            - dot[usable, None] * reference_rays[usable]
        ) / sin_angle[usable, None]
        tangent[usable] = angle[usable, None] * direction

    qx = reference_rays[:, 0]
    qy = reference_rays[:, 1]
    qz = np.clip(reference_rays[:, 2], -1.0, 1.0)
    theta = np.arccos(qz)
    phi = np.arctan2(qy, qx)
    e_theta = np.stack(
        (
            np.cos(theta) * np.cos(phi),
            np.cos(theta) * np.sin(phi),
            -np.sin(theta),
        ),
        axis=1,
    )
    e_phi = np.stack((-np.sin(phi), np.cos(phi), np.zeros_like(phi)), axis=1)

    scale = np.rad2deg(1.0) * 60.0
    total_arcmin = angle * scale
    radial_arcmin = np.sum(tangent * e_theta, axis=1) * scale
    tangential_arcmin = np.sum(tangent * e_phi, axis=1) * scale
    return total_arcmin, radial_arcmin, tangential_arcmin


def compare_calibrations(
    grid_path: Path,
    stellar_path: Path,
    *,
    spacing_px: float,
) -> ComparisonResult:
    """Sample and compare Grid and stellar calibrations over their common domain."""
    grid = load_grid_calibration(grid_path)
    stellar_product = load_stellar_camera_calibration_product(stellar_path)
    stellar = stellar_product.fit.calibration

    if tuple(grid.image_shape) != tuple(stellar.image_shape):
        raise ValueError(
            "Grid and stellar calibrations have different sensor dimensions: "
            f"{grid.image_shape} versus {stellar.image_shape}"
        )

    x_all, y_all = _sample_sensor(stellar.image_shape, spacing_px)
    stellar_mask = _stellar_calibrated_mask(stellar, x_all, y_all)
    x_stellar = x_all[stellar_mask]
    y_stellar = y_all[stellar_mask]

    converted = validated_pixel_rays(grid, x_stellar, y_stellar)
    common = np.asarray(converted.valid, dtype=np.bool_)
    if np.count_nonzero(common) < 20:
        raise ValueError(
            "Fewer than 20 sensor samples lie inside both calibrated domains"
        )

    x = x_stellar[common]
    y = y_stellar[common]
    grid_rays = np.asarray(converted.rays[common], dtype=np.float64)
    stellar_rays = np.asarray(
        stellar.pixel_to_camera_ray(x, y, extrapolate=False), dtype=np.float64
    )

    matrix, rotation_angle_deg, rotation_axis = _fit_rigid_alignment(
        grid_rays, stellar_rays
    )
    aligned_grid = grid_rays @ matrix.T
    total, radial, tangential = _spherical_log_components(
        stellar_rays, aligned_grid
    )

    theta_deg = np.rad2deg(
        np.arccos(np.clip(stellar_rays[:, 2], -1.0, 1.0))
    )
    aligned_grid_enu = aligned_grid @ np.asarray(stellar.camera_to_enu).T
    x_equiv, y_equiv = stellar.enu_to_pixel(aligned_grid_enu, extrapolate=True)
    dx = np.asarray(x_equiv, dtype=np.float64) - x
    dy = np.asarray(y_equiv, dtype=np.float64) - y
    pixel_residual = np.hypot(dx, dy)

    samples = ComparisonSamples(
        x_px=x,
        y_px=y,
        theta_deg=theta_deg,
        grid_rays=grid_rays,
        stellar_rays=stellar_rays,
        aligned_grid_rays=aligned_grid,
        angular_residual_arcmin=total,
        radial_residual_arcmin=radial,
        tangential_residual_arcmin=tangential,
        equivalent_dx_px=dx,
        equivalent_dy_px=dy,
        equivalent_pixel_residual_px=pixel_residual,
    )
    return ComparisonResult(
        samples=samples,
        grid_to_camera=matrix,
        rigid_rotation_angle_deg=rotation_angle_deg,
        rigid_rotation_axis_grid=rotation_axis,
    )


def _percentiles(values: FloatArray) -> dict[str, float]:
    """Return the standard scalar summary used throughout the report."""
    return {
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90.0)),
        "p99": float(np.percentile(values, 99.0)),
        "max": float(np.max(values)),
    }


def _summary(result: ComparisonResult) -> dict[str, Any]:
    """Build a JSON-serializable numerical comparison summary."""
    samples = result.samples
    return {
        "sample_count": int(samples.x_px.size),
        "best_fit_rigid_rotation_angle_deg": result.rigid_rotation_angle_deg,
        "best_fit_rigid_rotation_axis_grid": result.rigid_rotation_axis_grid.tolist(),
        "grid_to_camera_rotation_matrix": result.grid_to_camera.tolist(),
        "field_angle_deg": {
            "min": float(np.min(samples.theta_deg)),
            "max": float(np.max(samples.theta_deg)),
        },
        "angular_residual_arcmin": _percentiles(samples.angular_residual_arcmin),
        "absolute_radial_component_arcmin": _percentiles(
            np.abs(samples.radial_residual_arcmin)
        ),
        "absolute_tangential_component_arcmin": _percentiles(
            np.abs(samples.tangential_residual_arcmin)
        ),
        "equivalent_pixel_residual_px": _percentiles(
            samples.equivalent_pixel_residual_px
        ),
    }


def _binned_profile(
    theta_deg: FloatArray,
    residual_arcmin: FloatArray,
    bin_width_deg: float,
) -> tuple[FloatArray, FloatArray, FloatArray, NDArray[np.int64]]:
    """Return field-angle bin centers, medians, P90s, and populations."""
    if bin_width_deg <= 0.0 or not np.isfinite(bin_width_deg):
        raise ValueError("profile bin width must be finite and positive")
    upper = np.ceil(np.max(theta_deg) / bin_width_deg) * bin_width_deg
    edges = np.arange(0.0, upper + bin_width_deg, bin_width_deg)
    if edges.size < 2:
        edges = np.asarray([0.0, bin_width_deg], dtype=np.float64)
    indices = np.digitize(theta_deg, edges) - 1
    centers: list[float] = []
    medians: list[float] = []
    p90s: list[float] = []
    counts: list[int] = []
    for index in range(edges.size - 1):
        mask = indices == index
        count = int(np.count_nonzero(mask))
        if count == 0:
            continue
        values = residual_arcmin[mask]
        centers.append(float(0.5 * (edges[index] + edges[index + 1])))
        medians.append(float(np.median(values)))
        p90s.append(float(np.percentile(values, 90.0)))
        counts.append(count)
    return (
        np.asarray(centers, dtype=np.float64),
        np.asarray(medians, dtype=np.float64),
        np.asarray(p90s, dtype=np.float64),
        np.asarray(counts, dtype=np.int64),
    )


def _quiver_subset(
    samples: ComparisonSamples,
    spacing_px: float,
    quiver_spacing_px: float,
) -> BoolArray:
    """Select a regular sparse subset of the comparison samples for vectors."""
    stride = max(1, int(round(quiver_spacing_px / spacing_px)))
    x_index = np.rint(samples.x_px / spacing_px).astype(np.int64)
    y_index = np.rint(samples.y_px / spacing_px).astype(np.int64)
    return np.asarray(
        (x_index % stride == 0) & (y_index % stride == 0), dtype=np.bool_
    )


def _make_map_figure(
    result: ComparisonResult,
    image_shape: tuple[int, int],
    *,
    spacing_px: float,
    quiver_spacing_px: float,
) -> plt.Figure:
    """Create the sensor-map figure used in both PDF and documentation PNG."""
    samples = result.samples
    rows, columns = image_shape
    figure, axes = plt.subplots(1, 2, figsize=(13.0, 6.0), constrained_layout=True)

    scatter = axes[0].scatter(
        samples.x_px,
        samples.y_px,
        c=samples.angular_residual_arcmin,
        s=16,
    )
    axes[0].set_title("Grid vs stellar angular discrepancy")
    axes[0].set_xlabel("x [full-sensor px]")
    axes[0].set_ylabel("y [full-sensor px]")
    axes[0].set_xlim(0.0, columns)
    axes[0].set_ylim(rows, 0.0)
    figure.colorbar(scatter, ax=axes[0], label="angular discrepancy [arcmin]")

    subset = _quiver_subset(samples, spacing_px, quiver_spacing_px)
    axes[1].quiver(
        samples.x_px[subset],
        samples.y_px[subset],
        samples.equivalent_dx_px[subset],
        samples.equivalent_dy_px[subset],
        angles="xy",
        scale_units="xy",
        scale=0.15,
        width=0.003,
    )
    axes[1].set_title("Equivalent Grid deformation in stellar pixel geometry")
    axes[1].set_xlabel("x [full-sensor px]")
    axes[1].set_ylabel("y [full-sensor px]")
    axes[1].set_xlim(0.0, columns)
    axes[1].set_ylim(rows, 0.0)
    axes[1].text(
        0.02,
        0.02,
        "vectors enlarged 6.7×",
        transform=axes[1].transAxes,
        va="bottom",
    )
    return figure


def _make_profile_figure(
    result: ComparisonResult,
    *,
    profile_bin_width_deg: float,
) -> plt.Figure:
    """Create radial-profile and component diagnostics for documentation."""
    samples = result.samples
    centers, medians, p90s, counts = _binned_profile(
        samples.theta_deg,
        samples.angular_residual_arcmin,
        profile_bin_width_deg,
    )
    figure, axes = plt.subplots(2, 2, figsize=(12.0, 9.0), constrained_layout=True)

    axes[0, 0].scatter(
        samples.theta_deg,
        samples.angular_residual_arcmin,
        s=8,
        alpha=0.25,
    )
    axes[0, 0].plot(centers, medians, marker="o", label="median")
    axes[0, 0].plot(centers, p90s, marker="o", label="P90")
    axes[0, 0].set_xlabel("stellar-camera field angle [deg]")
    axes[0, 0].set_ylabel("angular discrepancy [arcmin]")
    axes[0, 0].set_title("Non-rigid discrepancy versus field angle")
    axes[0, 0].legend()

    axes[0, 1].scatter(
        samples.radial_residual_arcmin,
        samples.tangential_residual_arcmin,
        s=9,
        alpha=0.35,
    )
    axes[0, 1].axhline(0.0, linewidth=1.0)
    axes[0, 1].axvline(0.0, linewidth=1.0)
    axes[0, 1].set_xlabel("radial component [arcmin]")
    axes[0, 1].set_ylabel("tangential component [arcmin]")
    axes[0, 1].set_title("Grid-minus-stellar tangent-plane components")

    axes[1, 0].hist(samples.angular_residual_arcmin, bins=50)
    axes[1, 0].set_xlabel("angular discrepancy [arcmin]")
    axes[1, 0].set_ylabel("samples")
    axes[1, 0].set_title("Angular discrepancy distribution")

    axes[1, 1].bar(centers, counts, width=profile_bin_width_deg * 0.8)
    axes[1, 1].set_xlabel("stellar-camera field angle [deg]")
    axes[1, 1].set_ylabel("common sensor samples")
    axes[1, 1].set_title("Comparison-domain sampling")
    return figure


def _write_samples_csv(path: Path, samples: ComparisonSamples) -> None:
    """Write every retained comparison sample to a portable CSV file."""
    fields = (
        "x_px",
        "y_px",
        "field_angle_deg",
        "angular_residual_arcmin",
        "radial_residual_arcmin",
        "tangential_residual_arcmin",
        "equivalent_dx_px",
        "equivalent_dy_px",
        "equivalent_pixel_residual_px",
    )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index in range(samples.x_px.size):
            writer.writerow(
                {
                    "x_px": float(samples.x_px[index]),
                    "y_px": float(samples.y_px[index]),
                    "field_angle_deg": float(samples.theta_deg[index]),
                    "angular_residual_arcmin": float(
                        samples.angular_residual_arcmin[index]
                    ),
                    "radial_residual_arcmin": float(
                        samples.radial_residual_arcmin[index]
                    ),
                    "tangential_residual_arcmin": float(
                        samples.tangential_residual_arcmin[index]
                    ),
                    "equivalent_dx_px": float(samples.equivalent_dx_px[index]),
                    "equivalent_dy_px": float(samples.equivalent_dy_px[index]),
                    "equivalent_pixel_residual_px": float(
                        samples.equivalent_pixel_residual_px[index]
                    ),
                }
            )


def _print_summary(summary: dict[str, Any]) -> None:
    """Print the comparison quantities most useful for quick inspection."""
    angular = summary["angular_residual_arcmin"]
    pixel = summary["equivalent_pixel_residual_px"]
    radial = summary["absolute_radial_component_arcmin"]
    tangential = summary["absolute_tangential_component_arcmin"]
    field = summary["field_angle_deg"]
    print(
        "Grid-vs-stellar common domain | "
        f"{summary['sample_count']} samples | field angle "
        f"{field['min']:.2f}..{field['max']:.2f} deg"
    )
    print(
        "Best rigid Grid->camera alignment | "
        f"rotation {summary['best_fit_rigid_rotation_angle_deg']:.3f} deg"
    )
    print(
        "Residual angular discrepancy after rigid alignment | "
        f"median {angular['median']:.3f}' | P90 {angular['p90']:.3f}' | "
        f"P99 {angular['p99']:.3f}' | max {angular['max']:.3f}'"
    )
    print(
        "Equivalent pixel discrepancy | "
        f"median {pixel['median']:.3f}px | P90 {pixel['p90']:.3f}px | "
        f"P99 {pixel['p99']:.3f}px | max {pixel['max']:.3f}px"
    )
    print(
        "Absolute tangent components | "
        f"radial P90 {radial['p90']:.3f}' | "
        f"tangential P90 {tangential['p90']:.3f}'"
    )


def main() -> int:
    """Run the standalone Grid-versus-stellar calibration comparison."""
    args = _parse_args()
    grid_path = args.grid_calibration.expanduser().resolve()
    stellar_path = args.stellar_calibration.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    stellar_product = load_stellar_camera_calibration_product(stellar_path)
    image_shape = stellar_product.fit.calibration.image_shape
    result = compare_calibrations(
        grid_path,
        stellar_path,
        spacing_px=float(args.spacing_px),
    )
    summary = _summary(result)
    summary.update(
        {
            "grid_calibration": str(grid_path),
            "stellar_calibration": str(stellar_path),
            "sampling_spacing_px": float(args.spacing_px),
            "interpretation": (
                "Residuals are measured after removing the best proper rigid "
                "rotation between the Grid nominal ray field and the stellar "
                "camera ray field. They quantify non-rigid disagreement between "
                "the two calibrations, not the arbitrary orientation of their "
                "native coordinate frames."
            ),
        }
    )

    map_figure = _make_map_figure(
        result,
        image_shape,
        spacing_px=float(args.spacing_px),
        quiver_spacing_px=float(args.quiver_spacing_px),
    )
    profile_figure = _make_profile_figure(
        result,
        profile_bin_width_deg=float(args.profile_bin_width_deg),
    )

    map_png = output_dir / "grid_vs_stellar_angular_map.png"
    profile_png = output_dir / "grid_vs_stellar_residual_profile.png"
    pdf_path = output_dir / "grid_vs_stellar_comparison.pdf"
    csv_path = output_dir / "grid_vs_stellar_samples.csv"
    json_path = output_dir / "grid_vs_stellar_summary.json"

    map_figure.savefig(map_png, dpi=180)
    profile_figure.savefig(profile_png, dpi=180)
    with PdfPages(pdf_path) as pdf:
        pdf.savefig(map_figure)
        pdf.savefig(profile_figure)
    plt.close(map_figure)
    plt.close(profile_figure)

    _write_samples_csv(csv_path, result.samples)
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    _print_summary(summary)
    print(f"Wrote {pdf_path}")
    print(f"Wrote {map_png}")
    print(f"Wrote {profile_png}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
