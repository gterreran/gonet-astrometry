import numpy as np

from gonet_astrometry.models.grid import GridCalibration


class IdentityLikeTransform:
    def pixel_to_ray(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        ones = np.ones_like(x)
        vectors = np.stack((x, y, ones), axis=-1).astype(float)
        return vectors / np.linalg.norm(vectors, axis=-1, keepdims=True)


def test_grid_calibration_preserves_transform_contract() -> None:
    calibration = GridCalibration(
        IdentityLikeTransform(),
        (10, 12),
        "+x right, +y down, +z optical axis",
    )
    rays = calibration.transform.pixel_to_ray(
        np.array([0.0, 1.0]),
        np.array([0.0, 1.0]),
    )
    assert rays.shape == (2, 3)
    assert np.allclose(np.linalg.norm(rays, axis=1), 1.0)
