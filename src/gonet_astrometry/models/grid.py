"""Grid-calibration model and pixel-to-ray contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray


class PixelRayTransform(Protocol):
    """Protocol implemented by pixel-to-camera-ray transformations."""

    def pixel_to_ray(
        self,
        x: NDArray[np.float64],
        y: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Convert sensor coordinates to camera-frame unit vectors.

        Parameters
        ----------
        x, y
            Matching arrays of zero-indexed sensor coordinates.

        Returns
        -------
        numpy.ndarray
            Array with shape ``x.shape + (3,)`` containing unit vectors.
        """

        ...


@dataclass(frozen=True, slots=True)
class GridCalibration:
    """Geometric calibration imported from the Grid calibration tool.

    Parameters
    ----------
    transform
        Object converting native sensor pixels to camera-frame unit rays.
    image_shape
        Sensor shape for which the calibration is valid.
    coordinate_convention
        Explicit description of camera axes and pixel orientation.
    source
        Optional identifier or path for the serialized Grid calibration.
    """

    transform: PixelRayTransform
    image_shape: tuple[int, int]
    coordinate_convention: str
    source: str | None = None
