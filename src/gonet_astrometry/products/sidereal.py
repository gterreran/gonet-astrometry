"""Portable serialization for shared sidereal-rotation solutions."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from gonet_astrometry.products.errors import ProductFormatError, ProductMismatchError
from gonet_astrometry.solving.sidereal import (
    SiderealFitConfig,
    SiderealRotationSolution,
    SiderealTrackClass,
    SiderealTrackDiagnostics,
)

_FORMAT = "gonet-astrometry-sidereal-rotation"
_VERSION = 1


@dataclass(frozen=True, slots=True)
class SiderealProduct:
    """Reusable Grid-calibrated sidereal solution and provenance identifiers.

    Parameters
    ----------
    product_id
        Provenance identifier for the complete spherical-fit product.
    tracking_product_id
        Identifier of the tracking product used as input.
    grid_calibration_path
        Portable Grid Calibration artifact used for pixel-to-ray conversion.
    fit_config
        Configuration of the shared sidereal-axis fit.
    solution
        Reconstructed shared-rotation solution and per-track diagnostics.
    """

    product_id: str
    tracking_product_id: str
    grid_calibration_path: Path
    fit_config: SiderealFitConfig
    solution: SiderealRotationSolution


def save_sidereal_product(path: Path, product: SiderealProduct) -> Path:
    """Write ``product`` as a compressed NPZ containing no object arrays."""
    diagnostics = product.solution.track_diagnostics
    arrays: dict[str, NDArray[np.generic]] = {
        "format": _scalar_string(_FORMAT),
        "version": np.asarray(_VERSION, dtype=np.int64),
        "product_id": _scalar_string(product.product_id),
        "tracking_product_id": _scalar_string(product.tracking_product_id),
        "grid_calibration_path": _scalar_string(str(product.grid_calibration_path)),
        "fit_config_json": _scalar_string(
            json.dumps(
                asdict(product.fit_config),
                sort_keys=True,
                separators=(",", ":"),
            )
        ),
        "axis_grid": np.asarray(product.solution.axis_grid, dtype=np.float64),
        "fit_rms_deg": np.asarray(product.solution.fit_rms_deg, dtype=np.float64),
        "fit_median_deg": np.asarray(product.solution.fit_median_deg, dtype=np.float64),
        "fit_p95_deg": np.asarray(product.solution.fit_p95_deg, dtype=np.float64),
        "fitted_track_count": np.asarray(
            product.solution.fitted_track_count, dtype=np.int64
        ),
        "fitted_point_count": np.asarray(
            product.solution.fitted_point_count, dtype=np.int64
        ),
        "rotation_sign": np.asarray(product.solution.rotation_sign, dtype=np.int64),
        "track_identifier": np.asarray(
            [item.track_identifier for item in diagnostics], dtype=np.int64
        ),
        "track_class": _strings(item.diagnostic_class for item in diagnostics),
        "track_total_point_count": np.asarray(
            [item.total_point_count for item in diagnostics], dtype=np.int64
        ),
        "track_valid_point_count": np.asarray(
            [item.valid_point_count for item in diagnostics], dtype=np.int64
        ),
        "track_duration_s": np.asarray(
            [item.duration_s for item in diagnostics], dtype=np.float64
        ),
        "track_angular_span_deg": np.asarray(
            [item.angular_span_deg for item in diagnostics], dtype=np.float64
        ),
        "track_rms_residual_deg": _optional_float_array(
            item.rms_residual_deg for item in diagnostics
        ),
        "track_median_residual_deg": _optional_float_array(
            item.median_residual_deg for item in diagnostics
        ),
        "track_max_residual_deg": _optional_float_array(
            item.max_residual_deg for item in diagnostics
        ),
    }
    return _atomic_savez(path, arrays)


def load_sidereal_product(
    path: Path,
    *,
    expected_product_id: str | None = None,
    expected_tracking_product_id: str | None = None,
) -> SiderealProduct:
    """Load and validate a portable shared-rotation product."""
    try:
        with np.load(path, allow_pickle=False) as data:
            _validate_header(
                data,
                expected_product_id=expected_product_id,
                expected_tracking_product_id=expected_tracking_product_id,
            )
            axis = np.asarray(data["axis_grid"], dtype=np.float64)
            identifiers = np.asarray(data["track_identifier"], dtype=np.int64)
            classes = _read_strings(data, "track_class")
            count = len(identifiers)
            arrays = {
                name: np.asarray(data[name])
                for name in (
                    "track_total_point_count",
                    "track_valid_point_count",
                    "track_duration_s",
                    "track_angular_span_deg",
                    "track_rms_residual_deg",
                    "track_median_residual_deg",
                    "track_max_residual_deg",
                )
            }
            if len(classes) != count or any(
                len(value) != count for value in arrays.values()
            ):
                raise ProductFormatError(
                    "Sidereal track arrays have inconsistent lengths"
                )
            diagnostics: list[SiderealTrackDiagnostics] = []
            allowed = {
                "sidereal-consistent",
                "sidereal-rejected",
                "insufficient",
            }
            for index, class_name in enumerate(classes):
                if class_name not in allowed:
                    raise ProductFormatError("Invalid sidereal track diagnostic class")
                diagnostics.append(
                    SiderealTrackDiagnostics(
                        track_identifier=int(identifiers[index]),
                        diagnostic_class=cast(SiderealTrackClass, class_name),
                        total_point_count=int(arrays["track_total_point_count"][index]),
                        valid_point_count=int(arrays["track_valid_point_count"][index]),
                        duration_s=float(arrays["track_duration_s"][index]),
                        angular_span_deg=float(arrays["track_angular_span_deg"][index]),
                        rms_residual_deg=_none_if_nan(
                            arrays["track_rms_residual_deg"][index]
                        ),
                        median_residual_deg=_none_if_nan(
                            arrays["track_median_residual_deg"][index]
                        ),
                        max_residual_deg=_none_if_nan(
                            arrays["track_max_residual_deg"][index]
                        ),
                    )
                )
            payload = json.loads(_read_scalar_string(data, "fit_config_json"))
            if not isinstance(payload, dict):
                raise ProductFormatError(
                    "Sidereal fit configuration is not a JSON object"
                )
            fit_config = SiderealFitConfig(**cast(dict[str, Any], payload))
            solution = SiderealRotationSolution(
                axis_grid=axis,
                fit_rms_deg=float(np.asarray(data["fit_rms_deg"]).item()),
                fit_median_deg=float(np.asarray(data["fit_median_deg"]).item()),
                fit_p95_deg=float(np.asarray(data["fit_p95_deg"]).item()),
                fitted_track_count=int(np.asarray(data["fitted_track_count"]).item()),
                fitted_point_count=int(np.asarray(data["fitted_point_count"]).item()),
                track_diagnostics=tuple(diagnostics),
                rotation_sign=int(np.asarray(data["rotation_sign"]).item()),
            )
            product = SiderealProduct(
                product_id=_read_scalar_string(data, "product_id"),
                tracking_product_id=_read_scalar_string(data, "tracking_product_id"),
                grid_calibration_path=Path(
                    _read_scalar_string(data, "grid_calibration_path")
                ),
                fit_config=fit_config,
                solution=solution,
            )
    except ProductMismatchError:
        raise
    except (KeyError, OSError, TypeError, ValueError) as exc:
        if isinstance(exc, ProductFormatError):
            raise
        raise ProductFormatError(
            f"Could not read sidereal product {path}: {exc}"
        ) from exc
    return product


def _validate_header(
    data: np.lib.npyio.NpzFile,
    *,
    expected_product_id: str | None,
    expected_tracking_product_id: str | None,
) -> None:
    if _read_scalar_string(data, "format") != _FORMAT:
        raise ProductFormatError("File is not a GONet Astrometry sidereal product")
    if int(np.asarray(data["version"]).item()) != _VERSION:
        raise ProductFormatError("Unsupported sidereal product version")
    product_id = _read_scalar_string(data, "product_id")
    tracking_id = _read_scalar_string(data, "tracking_product_id")
    if expected_product_id is not None and product_id != expected_product_id:
        raise ProductMismatchError(
            "Sidereal product does not match the current Grid calibration or settings"
        )
    if (
        expected_tracking_product_id is not None
        and tracking_id != expected_tracking_product_id
    ):
        raise ProductMismatchError(
            "Sidereal product belongs to a different tracking product"
        )


def _optional_float_array(
    values: Iterable[float | None],
) -> NDArray[np.float64]:
    return np.asarray(
        [np.nan if item is None else float(item) for item in values],
        dtype=np.float64,
    )


def _none_if_nan(value: object) -> float | None:
    parsed = float(value)
    return None if np.isnan(parsed) else parsed


def _strings(values: Iterable[object]) -> NDArray[np.str_]:
    strings = tuple(str(item) for item in values)
    width = max((len(item) for item in strings), default=1)
    return np.asarray(strings, dtype=f"<U{width}")


def _scalar_string(value: str) -> NDArray[np.str_]:
    return np.asarray(value, dtype=f"<U{max(1, len(value))}")


def _read_strings(data: np.lib.npyio.NpzFile, name: str) -> tuple[str, ...]:
    array = np.asarray(data[name])
    if array.ndim != 1:
        raise ProductFormatError(f"Product field {name!r} must be one-dimensional")
    return tuple(str(item) for item in array.tolist())


def _read_scalar_string(data: np.lib.npyio.NpzFile, name: str) -> str:
    array = np.asarray(data[name])
    if array.ndim != 0:
        raise ProductFormatError(f"Product field {name!r} must be scalar")
    return str(array.item())


def _atomic_savez(path: Path, arrays: dict[str, NDArray[np.generic]]) -> Path:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination
