import sys
from types import ModuleType

import numpy as np
import pytest

from gonet_astrometry.adapters.grid_calibration import (
    PortableGridTransform,
    angles_to_grid_rays,
    grid_pole_pixel,
    load_grid_calibration,
    validated_pixel_rays,
)
from gonet_astrometry.models.grid import GridCalibration


class FakeEvaluator:
    sensor_width_px = 200
    sensor_height_px = 400
    image_coordinate_convention = (
        "x=column,y=row;origin=upper-left;+x=right;+y=down;"
        "pixel-centers-at-integer-coordinates"
    )
    calibrated_angular_range_deg = (0.0, 170.0)

    def pixel_to_angle(self, x, y, **kwargs):
        del kwargs
        return np.asarray(x, dtype=float), np.asarray(y, dtype=float)

    def angle_to_pixel(self, r_deg, theta_deg):
        return np.asarray(r_deg, dtype=float), np.asarray(theta_deg, dtype=float)


def _calibration() -> GridCalibration:
    evaluator = FakeEvaluator()
    return GridCalibration(
        transform=PortableGridTransform(evaluator),
        image_shape=(400, 200),
        coordinate_convention=evaluator.image_coordinate_convention,
        source="fake.npz",
    )


def test_angles_to_grid_rays_has_expected_axes() -> None:
    rays = angles_to_grid_rays(
        np.asarray([0.0, 90.0, 90.0]),
        np.asarray([0.0, 0.0, 90.0]),
    )
    assert np.allclose(rays[0], [0.0, 0.0, 1.0])
    assert np.allclose(rays[1], [1.0, 0.0, 0.0], atol=1e-12)
    assert np.allclose(rays[2], [0.0, 1.0, 0.0], atol=1e-12)


def test_validated_pixel_rays_marks_outside_calibrated_radius() -> None:
    result = validated_pixel_rays(
        _calibration(),
        np.asarray([45.0, 175.0]),
        np.asarray([120.0, 20.0]),
    )
    assert result.valid.tolist() == [True, False]
    assert np.isclose(np.linalg.norm(result.rays[0]), 1.0)
    assert np.isnan(result.rays[1]).all()


def test_validated_pixel_rays_allows_radii_inside_recorded_inner_range() -> None:
    evaluator = FakeEvaluator()
    evaluator.calibrated_angular_range_deg = (20.0, 170.0)
    calibration = GridCalibration(
        transform=PortableGridTransform(evaluator),
        image_shape=(400, 200),
        coordinate_convention=evaluator.image_coordinate_convention,
    )

    result = validated_pixel_rays(
        calibration,
        np.asarray([5.0]),
        np.asarray([120.0]),
    )

    assert result.valid.tolist() == [True]


def test_portable_transform_requires_all_points_for_protocol_method() -> None:
    transform = _calibration().transform
    with pytest.raises(ValueError, match="could not invert"):
        transform.pixel_to_ray(  # type: ignore[attr-defined]
            np.asarray([45.0, 175.0]),
            np.asarray([120.0, 20.0]),
        )


def test_grid_pole_pixel_projects_inside_calibrated_domain() -> None:
    calibration = _calibration()
    ray = angles_to_grid_rays(np.asarray(45.0), np.asarray(120.0))
    assert grid_pole_pixel(calibration, ray) == pytest.approx((45.0, 120.0))
    opposite = np.asarray([0.0, 0.0, -1.0])
    assert grid_pole_pixel(calibration, opposite) is None


def test_load_grid_calibration_uses_public_grid_package(monkeypatch, tmp_path) -> None:
    evaluator = FakeEvaluator()
    module = ModuleType("grid_calibration")
    module.load_calibration = lambda path: evaluator  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "grid_calibration", module)
    path = tmp_path / "camera_calibration.npz"
    path.write_bytes(b"placeholder")

    calibration = load_grid_calibration(path)

    assert calibration.image_shape == (400, 200)
    assert calibration.source == str(path.resolve())


def test_load_grid_calibration_reports_missing_dependency(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setitem(sys.modules, "grid_calibration", None)
    with pytest.raises(RuntimeError, match="grid_calibration package"):
        load_grid_calibration(tmp_path / "camera_calibration.npz")


def test_load_grid_calibration_rejects_coordinate_convention(
    monkeypatch, tmp_path
) -> None:
    evaluator = FakeEvaluator()
    evaluator.image_coordinate_convention = "other"
    module = ModuleType("grid_calibration")
    module.load_calibration = lambda path: evaluator  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "grid_calibration", module)
    path = tmp_path / "camera_calibration.npz"
    path.write_bytes(b"placeholder")

    with pytest.raises(ValueError, match="coordinate convention"):
        load_grid_calibration(path)
