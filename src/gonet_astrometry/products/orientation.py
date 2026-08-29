"""Portable absolute camera-orientation product."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from gonet_astrometry.models.frame import ObserverLocation
from gonet_astrometry.products.errors import ProductFormatError, ProductMismatchError
from gonet_astrometry.solving.orientation import (
    AbsoluteOrientationSolution,
    OrientationFitConfig,
    OrientationMatch,
)

_FORMAT = "gonet-astrometry-absolute-orientation"
_VERSION = 1


@dataclass(frozen=True, slots=True)
class OrientationProduct:
    """Reusable catalog-assisted absolute attitude and provenance identifiers."""

    product_id: str
    sidereal_product_id: str
    catalog_path: Path
    fit_config: OrientationFitConfig
    solution: AbsoluteOrientationSolution


def save_orientation_product(path: Path, product: OrientationProduct) -> Path:
    """Write ``product`` as a compressed NPZ containing no object arrays."""
    matches = product.solution.matches
    arrays: dict[str, NDArray[np.generic]] = {
        "format": _scalar_string(_FORMAT),
        "version": np.asarray(_VERSION, dtype=np.int64),
        "product_id": _scalar_string(product.product_id),
        "sidereal_product_id": _scalar_string(product.sidereal_product_id),
        "catalog_path": _scalar_string(str(product.catalog_path)),
        "fit_config_json": _scalar_string(
            json.dumps(
                asdict(product.fit_config),
                sort_keys=True,
                separators=(",", ":"),
            )
        ),
        "grid_to_enu": np.asarray(product.solution.grid_to_enu, dtype=np.float64),
        "reference_time": _scalar_string(product.solution.reference_time.isoformat()),
        "location": np.asarray(
            [
                product.solution.location.latitude_deg,
                product.solution.location.longitude_deg,
                product.solution.location.elevation_m,
            ],
            dtype=np.float64,
        ),
        "ncp_grid": np.asarray(product.solution.ncp_grid, dtype=np.float64),
        "ncp_enu": np.asarray(product.solution.ncp_enu, dtype=np.float64),
        "twist_deg": np.asarray(product.solution.twist_deg, dtype=np.float64),
        "fit_rms_deg": np.asarray(product.solution.fit_rms_deg, dtype=np.float64),
        "fit_median_deg": np.asarray(product.solution.fit_median_deg, dtype=np.float64),
        "fit_p95_deg": np.asarray(product.solution.fit_p95_deg, dtype=np.float64),
        "anchor_count": np.asarray(product.solution.anchor_count, dtype=np.int64),
        "catalog_star_count": np.asarray(
            product.solution.catalog_star_count, dtype=np.int64
        ),
        "match_track_identifier": np.asarray(
            [item.track_identifier for item in matches], dtype=np.int64
        ),
        "match_catalog_identifier": _strings(
            item.catalog_identifier for item in matches
        ),
        "match_residual_deg": np.asarray(
            [item.residual_deg for item in matches], dtype=np.float64
        ),
        "match_observed_declination_deg": np.asarray(
            [item.observed_declination_deg for item in matches], dtype=np.float64
        ),
        "match_catalog_declination_deg": np.asarray(
            [item.catalog_declination_deg for item in matches], dtype=np.float64
        ),
        "match_catalog_magnitude": np.asarray(
            [
                np.nan if item.catalog_magnitude is None else item.catalog_magnitude
                for item in matches
            ],
            dtype=np.float64,
        ),
        "match_ray_grid": np.asarray(
            [item.ray_grid for item in matches], dtype=np.float64
        ).reshape((-1, 3)),
    }
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(destination)
    return destination


def load_orientation_product(
    path: Path,
    *,
    expected_product_id: str | None = None,
    expected_sidereal_product_id: str | None = None,
) -> OrientationProduct:
    """Load and validate a portable absolute-orientation product."""
    source = Path(path).expanduser().resolve()
    try:
        with np.load(source, allow_pickle=False) as data:
            _validate_header(
                data,
                expected_product_id=expected_product_id,
                expected_sidereal_product_id=expected_sidereal_product_id,
            )
            parsed_payload = json.loads(_read_scalar_string(data, "fit_config_json"))
            if not isinstance(parsed_payload, dict):
                raise ProductFormatError("Orientation fit configuration is not JSON")
            payload = cast(dict[str, Any], parsed_payload)
            fit_config = OrientationFitConfig(**payload)
            location_values = np.asarray(data["location"], dtype=np.float64)
            if location_values.shape != (3,):
                raise ProductFormatError("Orientation location must have shape (3,)")
            identifiers = np.asarray(data["match_track_identifier"], dtype=np.int64)
            catalog_ids = _read_strings(data, "match_catalog_identifier")
            residual = np.asarray(data["match_residual_deg"], dtype=np.float64)
            observed_dec = np.asarray(
                data["match_observed_declination_deg"], dtype=np.float64
            )
            catalog_dec = np.asarray(
                data["match_catalog_declination_deg"], dtype=np.float64
            )
            magnitude = np.asarray(data["match_catalog_magnitude"], dtype=np.float64)
            rays = np.asarray(data["match_ray_grid"], dtype=np.float64)
            count = len(identifiers)
            if len(catalog_ids) != count or rays.shape != (count, 3):
                raise ProductFormatError("Orientation match arrays are inconsistent")
            if any(
                len(array) != count
                for array in (residual, observed_dec, catalog_dec, magnitude)
            ):
                raise ProductFormatError("Orientation match arrays are inconsistent")
            matches = tuple(
                OrientationMatch(
                    track_identifier=int(identifiers[index]),
                    catalog_identifier=catalog_ids[index],
                    residual_deg=float(residual[index]),
                    observed_declination_deg=float(observed_dec[index]),
                    catalog_declination_deg=float(catalog_dec[index]),
                    catalog_magnitude=(
                        None if np.isnan(magnitude[index]) else float(magnitude[index])
                    ),
                    ray_grid=rays[index],
                )
                for index in range(count)
            )
            solution = AbsoluteOrientationSolution(
                grid_to_enu=np.asarray(data["grid_to_enu"], dtype=np.float64),
                reference_time=datetime.fromisoformat(
                    _read_scalar_string(data, "reference_time")
                ),
                location=ObserverLocation(
                    latitude_deg=float(location_values[0]),
                    longitude_deg=float(location_values[1]),
                    elevation_m=float(location_values[2]),
                ),
                ncp_grid=np.asarray(data["ncp_grid"], dtype=np.float64),
                ncp_enu=np.asarray(data["ncp_enu"], dtype=np.float64),
                twist_deg=float(np.asarray(data["twist_deg"]).item()),
                fit_rms_deg=float(np.asarray(data["fit_rms_deg"]).item()),
                fit_median_deg=float(np.asarray(data["fit_median_deg"]).item()),
                fit_p95_deg=float(np.asarray(data["fit_p95_deg"]).item()),
                anchor_count=int(np.asarray(data["anchor_count"]).item()),
                catalog_star_count=int(np.asarray(data["catalog_star_count"]).item()),
                matches=matches,
            )
            return OrientationProduct(
                product_id=_read_scalar_string(data, "product_id"),
                sidereal_product_id=_read_scalar_string(data, "sidereal_product_id"),
                catalog_path=Path(_read_scalar_string(data, "catalog_path")),
                fit_config=fit_config,
                solution=solution,
            )
    except ProductMismatchError:
        raise
    except (KeyError, OSError, TypeError, ValueError) as exc:
        if isinstance(exc, ProductFormatError):
            raise
        raise ProductFormatError(
            f"Could not read orientation product {source}: {exc}"
        ) from exc


def _validate_header(
    data: np.lib.npyio.NpzFile,
    *,
    expected_product_id: str | None,
    expected_sidereal_product_id: str | None,
) -> None:
    if _read_scalar_string(data, "format") != _FORMAT:
        raise ProductFormatError("File is not a GONet Astrometry orientation product")
    if int(np.asarray(data["version"]).item()) != _VERSION:
        raise ProductFormatError("Unsupported orientation product version")
    product_id = _read_scalar_string(data, "product_id")
    sidereal_id = _read_scalar_string(data, "sidereal_product_id")
    if expected_product_id is not None and product_id != expected_product_id:
        raise ProductMismatchError(
            "Orientation product does not match the current catalog or settings"
        )
    if (
        expected_sidereal_product_id is not None
        and sidereal_id != expected_sidereal_product_id
    ):
        raise ProductMismatchError(
            "Orientation product belongs to a different sidereal solution"
        )


def _strings(values: Iterable[object]) -> NDArray[np.str_]:
    strings = tuple(str(item) for item in values)
    width = max((len(item) for item in strings), default=1)
    return np.asarray(strings, dtype=f"<U{width}")


def _scalar_string(value: str) -> NDArray[np.str_]:
    return np.asarray(value, dtype=f"<U{max(1, len(value))}")


def _read_strings(data: np.lib.npyio.NpzFile, name: str) -> tuple[str, ...]:
    array = np.asarray(data[name])
    if array.ndim != 1:
        raise ProductFormatError(f"{name} must be one-dimensional")
    return tuple(str(item) for item in array.tolist())


def _read_scalar_string(data: np.lib.npyio.NpzFile, name: str) -> str:
    array = np.asarray(data[name])
    if array.ndim != 0:
        raise ProductFormatError(f"{name} must be scalar")
    return str(array.item())
