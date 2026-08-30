"""Portable Grid-independent stellar camera-calibration product."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from gonet_astrometry.calibration.stellar_camera import (
    StellarCameraCalibration,
    StellarCameraCalibrationConfig,
    StellarCameraCalibrationFit,
)
from gonet_astrometry.products.errors import (
    ProductFormatError,
    ProductMismatchError,
)

_FORMAT = "gonet-astrometry-stellar-camera-calibration"
_VERSION = 1
_INTRINSIC_MODEL = "radial-poly3"


@dataclass(frozen=True, slots=True)
class StellarCameraCalibrationProduct:
    """Reusable direct stellar camera calibration and provenance."""

    product_id: str
    detection_product_id: str
    bootstrap_identification_product_id: str
    catalog_path: Path
    config: StellarCameraCalibrationConfig
    fit: StellarCameraCalibrationFit


def save_stellar_camera_calibration_product(
    path: Path,
    product: StellarCameraCalibrationProduct,
) -> Path:
    """Write a pickle-free direct stellar camera calibration product."""
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    calibration = product.fit.calibration
    arrays = {
        "format": _scalar_string(_FORMAT),
        "version": np.asarray(_VERSION, dtype=np.int64),
        "intrinsic_model": _scalar_string(_INTRINSIC_MODEL),
        "product_id": _scalar_string(product.product_id),
        "detection_product_id": _scalar_string(product.detection_product_id),
        "bootstrap_identification_product_id": _scalar_string(
            product.bootstrap_identification_product_id
        ),
        "catalog_path": _scalar_string(str(product.catalog_path)),
        "config_json": _scalar_string(
            json.dumps(asdict(product.config), sort_keys=True)
        ),
        "image_shape": np.asarray(calibration.image_shape, dtype=np.int64),
        "coordinate_convention": _scalar_string(calibration.coordinate_convention),
        "camera_to_enu": np.asarray(calibration.camera_to_enu, dtype=np.float64),
        "center_x_px": np.asarray(calibration.center_x_px, dtype=np.float64),
        "center_y_px": np.asarray(calibration.center_y_px, dtype=np.float64),
        "radial_c1_px": np.asarray(calibration.radial_c1_px, dtype=np.float64),
        "radial_c3_px": np.asarray(calibration.radial_c3_px, dtype=np.float64),
        "calibrated_theta_max_deg": np.asarray(
            calibration.calibrated_theta_max_deg, dtype=np.float64
        ),
        "seed_measurement_count": np.asarray(
            product.fit.seed_measurement_count, dtype=np.int64
        ),
        "seed_star_count": np.asarray(product.fit.seed_star_count, dtype=np.int64),
        "trusted_seed_star_count": np.asarray(
            product.fit.trusted_seed_star_count, dtype=np.int64
        ),
        "rejected_seed_star_ids": _strings(product.fit.rejected_seed_star_ids),
        "direct_match_count": np.asarray(
            product.fit.direct_match_count, dtype=np.int64
        ),
        "direct_star_count": np.asarray(product.fit.direct_star_count, dtype=np.int64),
        "direct_match_median_px": np.asarray(
            product.fit.direct_match_median_px, dtype=np.float64
        ),
        "direct_match_p90_px": np.asarray(
            product.fit.direct_match_p90_px, dtype=np.float64
        ),
        "calibration_measurement_count": np.asarray(
            product.fit.calibration_measurement_count, dtype=np.int64
        ),
        "calibration_star_count": np.asarray(
            product.fit.calibration_star_count, dtype=np.int64
        ),
        "cv_median_arcmin": np.asarray(product.fit.cv_median_arcmin, dtype=np.float64),
        "cv_p90_arcmin": np.asarray(product.fit.cv_p90_arcmin, dtype=np.float64),
        "cv_p90_px": np.asarray(product.fit.cv_p90_px, dtype=np.float64),
    }
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(destination)
    return destination


def load_stellar_camera_calibration_product(
    path: Path,
    *,
    expected_product_id: str | None = None,
    expected_detection_product_id: str | None = None,
    expected_bootstrap_identification_product_id: str | None = None,
) -> StellarCameraCalibrationProduct:
    """Load and validate a portable direct stellar camera calibration."""
    source = Path(path).expanduser().resolve()
    try:
        with np.load(source, allow_pickle=False) as data:
            if _read_scalar_string(data, "format") != _FORMAT:
                raise ProductFormatError(
                    "File is not a stellar camera calibration product"
                )
            if int(np.asarray(data["version"]).item()) != _VERSION:
                raise ProductFormatError(
                    "Unsupported stellar camera calibration product version"
                )
            if _read_scalar_string(data, "intrinsic_model") != _INTRINSIC_MODEL:
                raise ProductFormatError(
                    "Unsupported stellar camera calibration intrinsic model"
                )
            product_id = _read_scalar_string(data, "product_id")
            detection_product_id = _read_scalar_string(data, "detection_product_id")
            bootstrap_id = _read_scalar_string(
                data, "bootstrap_identification_product_id"
            )
            if expected_product_id is not None and product_id != expected_product_id:
                raise ProductMismatchError(
                    "Stellar camera calibration does not match current inputs/settings"
                )
            if (
                expected_detection_product_id is not None
                and detection_product_id != expected_detection_product_id
            ):
                raise ProductMismatchError(
                    "Stellar camera calibration references another detection product"
                )
            if (
                expected_bootstrap_identification_product_id is not None
                and bootstrap_id != expected_bootstrap_identification_product_id
            ):
                raise ProductMismatchError(
                    "Stellar camera calibration references another bootstrap "
                    "identification product"
                )
            config = StellarCameraCalibrationConfig(
                **json.loads(_read_scalar_string(data, "config_json"))
            )
            image_shape_array = np.asarray(data["image_shape"], dtype=np.int64)
            if image_shape_array.shape != (2,):
                raise ProductFormatError("stellar calibration image_shape is invalid")
            calibration = StellarCameraCalibration(
                image_shape=(
                    int(image_shape_array[0]),
                    int(image_shape_array[1]),
                ),
                camera_to_enu=np.asarray(data["camera_to_enu"], dtype=np.float64),
                center_x_px=float(np.asarray(data["center_x_px"]).item()),
                center_y_px=float(np.asarray(data["center_y_px"]).item()),
                radial_c1_px=float(np.asarray(data["radial_c1_px"]).item()),
                radial_c3_px=float(np.asarray(data["radial_c3_px"]).item()),
                calibrated_theta_max_deg=float(
                    np.asarray(data["calibrated_theta_max_deg"]).item()
                ),
                coordinate_convention=_read_scalar_string(
                    data, "coordinate_convention"
                ),
            )
            fit = StellarCameraCalibrationFit(
                calibration=calibration,
                seed_measurement_count=int(
                    np.asarray(data["seed_measurement_count"]).item()
                ),
                seed_star_count=int(np.asarray(data["seed_star_count"]).item()),
                trusted_seed_star_count=int(
                    np.asarray(data["trusted_seed_star_count"]).item()
                ),
                rejected_seed_star_ids=_read_strings(data, "rejected_seed_star_ids"),
                direct_match_count=int(np.asarray(data["direct_match_count"]).item()),
                direct_star_count=int(np.asarray(data["direct_star_count"]).item()),
                direct_match_median_px=float(
                    np.asarray(data["direct_match_median_px"]).item()
                ),
                direct_match_p90_px=float(
                    np.asarray(data["direct_match_p90_px"]).item()
                ),
                calibration_measurement_count=int(
                    np.asarray(data["calibration_measurement_count"]).item()
                ),
                calibration_star_count=int(
                    np.asarray(data["calibration_star_count"]).item()
                ),
                cv_median_arcmin=float(np.asarray(data["cv_median_arcmin"]).item()),
                cv_p90_arcmin=float(np.asarray(data["cv_p90_arcmin"]).item()),
                cv_p90_px=float(np.asarray(data["cv_p90_px"]).item()),
            )
            catalog_path = Path(_read_scalar_string(data, "catalog_path"))
    except (ProductFormatError, ProductMismatchError):
        raise
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ProductFormatError(
            f"Could not read stellar camera calibration product {source}: {exc}"
        ) from exc
    return StellarCameraCalibrationProduct(
        product_id=product_id,
        detection_product_id=detection_product_id,
        bootstrap_identification_product_id=bootstrap_id,
        catalog_path=catalog_path,
        config=config,
        fit=fit,
    )


def _scalar_string(value: str) -> NDArray[np.str_]:
    return np.asarray(value, dtype=f"<U{max(1, len(value))}")


def _strings(values: tuple[str, ...]) -> NDArray[np.str_]:
    width = max((len(value) for value in values), default=1)
    return np.asarray(values, dtype=f"<U{width}")


def _read_scalar_string(data: np.lib.npyio.NpzFile, name: str) -> str:
    value = np.asarray(data[name])
    if value.ndim != 0:
        raise ProductFormatError(f"{name} must be scalar")
    return str(value.item())


def _read_strings(data: np.lib.npyio.NpzFile, name: str) -> tuple[str, ...]:
    value = np.asarray(data[name])
    if value.ndim != 1:
        raise ProductFormatError(f"{name} must be one-dimensional")
    return tuple(str(item) for item in value.tolist())
