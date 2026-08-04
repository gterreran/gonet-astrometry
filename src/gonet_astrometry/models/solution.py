"""Astrometric-solution models."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class AstrometricSolution:
    """Result of an astrometric calibration.

    Parameters
    ----------
    camera_to_local_rotation
        Three-by-three proper rotation matrix mapping camera-frame rays to the
        chosen local topocentric frame.
    celestial_pole_camera
        Three-element unit vector giving the celestial-pole direction in the
        camera frame.
    rms_residual_pixels
        Root-mean-square matched-source residual in pixels.
    matched_detection_count
        Number of detections contributing to the reported solution.
    parameters
        Additional fitted scalar parameters, such as a clock offset or lens
        coefficients.
    diagnostics
        Additional serializable diagnostic values.

    Raises
    ------
    ValueError
        If array shapes are invalid, vectors are not normalized, the matrix is
        not a proper rotation, or scalar quality values are invalid.
    """

    camera_to_local_rotation: NDArray[np.float64]
    celestial_pole_camera: NDArray[np.float64]
    rms_residual_pixels: float
    matched_detection_count: int
    parameters: Mapping[str, float] = field(default_factory=dict)
    diagnostics: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        rotation = np.asarray(self.camera_to_local_rotation, dtype=float)
        pole = np.asarray(self.celestial_pole_camera, dtype=float)
        if rotation.shape != (3, 3):
            raise ValueError("camera_to_local_rotation must have shape (3, 3)")
        if pole.shape != (3,):
            raise ValueError("celestial_pole_camera must have shape (3,)")
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-7):
            raise ValueError("camera_to_local_rotation must be orthonormal")
        if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-7):
            raise ValueError("camera_to_local_rotation must have determinant +1")
        if not np.isclose(np.linalg.norm(pole), 1.0, atol=1e-7):
            raise ValueError("celestial_pole_camera must be a unit vector")
        if self.rms_residual_pixels < 0:
            raise ValueError("rms_residual_pixels must be non-negative")
        if self.matched_detection_count < 0:
            raise ValueError("matched_detection_count must be non-negative")

        object.__setattr__(self, "camera_to_local_rotation", rotation.copy())
        object.__setattr__(self, "celestial_pole_camera", pole.copy())
        object.__setattr__(
            self,
            "parameters",
            MappingProxyType(dict(self.parameters)),
        )
        object.__setattr__(
            self,
            "diagnostics",
            MappingProxyType(dict(self.diagnostics)),
        )
