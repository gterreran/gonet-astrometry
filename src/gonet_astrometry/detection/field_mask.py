"""Portable static detector masks in native full-sensor coordinates.

A field mask describes pixels that should be excluded from source detection.
The artifact is intentionally independent of any one detector backend so the
same physical obstruction mask can be reused across observations made with the
same camera geometry.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

_FORMAT = "gonet-astrometry-field-mask"
_VERSION = 1


class FieldMaskError(ValueError):
    """Raised when a field-mask artifact is malformed or incompatible."""


@dataclass(frozen=True, slots=True)
class FieldMask:
    """Static full-sensor exclusion mask.

    Parameters
    ----------
    excluded
        Boolean array in native full-sensor coordinates. ``True`` means the
        pixel must not contribute to background estimation or source detection.
    coordinate_convention
        Explicit pixel-coordinate convention used by the mask.
    description
        Optional human-readable note describing the masked structure.
    """

    excluded: NDArray[np.bool_]
    coordinate_convention: str
    description: str = ""

    def __post_init__(self) -> None:
        excluded = np.asarray(self.excluded, dtype=np.bool_)
        if excluded.ndim != 2:
            raise FieldMaskError("Field mask must be a two-dimensional array")
        if excluded.size == 0:
            raise FieldMaskError("Field mask cannot be empty")
        object.__setattr__(self, "excluded", excluded)
        if not self.coordinate_convention:
            raise FieldMaskError("coordinate_convention cannot be empty")

    @property
    def image_shape(self) -> tuple[int, int]:
        """Return native ``(rows, columns)`` sensor shape."""
        return (int(self.excluded.shape[0]), int(self.excluded.shape[1]))

    @property
    def excluded_fraction(self) -> float:
        """Return the fraction of full-sensor pixels excluded by this mask."""
        return float(np.mean(self.excluded))

    def validate_against(
        self,
        image_shape: tuple[int, int],
        coordinate_convention: str,
    ) -> None:
        """Require this mask to match a detector/calibration coordinate system."""
        if self.image_shape != tuple(image_shape):
            raise FieldMaskError(
                "Field mask sensor shape does not match the current image: "
                f"mask={self.image_shape}, image={tuple(image_shape)}"
            )
        if self.coordinate_convention != coordinate_convention:
            raise FieldMaskError(
                "Field mask coordinate convention does not match the Grid "
                "calibration"
            )


def circular_exclusion_mask(
    image_shape: tuple[int, int],
    *,
    center_x: float,
    center_y: float,
    radius_px: float,
    coordinate_convention: str,
    description: str = "",
) -> FieldMask:
    """Return a full-sensor mask excluding one circular region."""
    if radius_px <= 0.0:
        raise ValueError("radius_px must be positive")
    rows, columns = image_shape
    if rows <= 0 or columns <= 0:
        raise ValueError("image_shape must contain positive dimensions")

    yy, xx = np.indices((rows, columns), dtype=np.float64)
    excluded = (
        np.square(xx - float(center_x)) + np.square(yy - float(center_y))
        <= float(radius_px) ** 2
    )
    return FieldMask(
        excluded=np.asarray(excluded, dtype=np.bool_),
        coordinate_convention=coordinate_convention,
        description=description,
    )


def save_field_mask(path: Path, mask: FieldMask) -> Path:
    """Write a portable pickle-free field-mask artifact."""
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    arrays = {
        "format": _scalar_string(_FORMAT),
        "version": np.asarray(_VERSION, dtype=np.int64),
        "excluded": np.asarray(mask.excluded, dtype=np.bool_),
        "coordinate_convention": _scalar_string(mask.coordinate_convention),
        "description": _scalar_string(mask.description),
    }
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def load_field_mask(path: Path) -> FieldMask:
    """Load one portable field-mask artifact."""
    source = Path(path).expanduser().resolve()
    try:
        with np.load(source, allow_pickle=False) as data:
            if _read_scalar_string(data, "format") != _FORMAT:
                raise FieldMaskError("Unexpected field-mask artifact format")
            version = int(np.asarray(data["version"]).item())
            if version != _VERSION:
                raise FieldMaskError(
                    f"Unsupported field-mask version {version}; expected {_VERSION}"
                )
            excluded = np.asarray(data["excluded"], dtype=np.bool_)
            coordinate_convention = _read_scalar_string(
                data, "coordinate_convention"
            )
            description = _read_scalar_string(data, "description")
    except FieldMaskError:
        raise
    except (OSError, KeyError, ValueError, TypeError) as exc:
        raise FieldMaskError(f"Could not load field mask {source}: {exc}") from exc

    return FieldMask(
        excluded=excluded,
        coordinate_convention=coordinate_convention,
        description=description,
    )


def _scalar_string(value: str) -> NDArray[np.str_]:
    return np.asarray(value, dtype=f"<U{max(1, len(value))}")


def _read_scalar_string(data: np.lib.npyio.NpzFile, name: str) -> str:
    array = np.asarray(data[name])
    if array.ndim != 0:
        raise FieldMaskError(f"Field-mask field {name!r} must be scalar")
    return str(array.item())
