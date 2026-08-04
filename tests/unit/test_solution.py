import numpy as np
import pytest

from gonet_astrometry.models.solution import AstrometricSolution


def make_solution(**overrides: object) -> AstrometricSolution:
    values: dict[str, object] = {
        "camera_to_local_rotation": np.eye(3),
        "celestial_pole_camera": np.array([0.0, 0.0, 1.0]),
        "rms_residual_pixels": 0.4,
        "matched_detection_count": 20,
        "parameters": {"clock_offset_s": 0.0},
    }
    values.update(overrides)
    return AstrometricSolution(**values)  # type: ignore[arg-type]


def test_solution_copies_arrays_and_mappings() -> None:
    rotation = np.eye(3)
    parameters = {"clock_offset_s": 0.0}
    solution = make_solution(
        camera_to_local_rotation=rotation,
        parameters=parameters,
    )
    rotation[0, 0] = 4.0
    parameters["clock_offset_s"] = 9.0
    assert solution.camera_to_local_rotation[0, 0] == 1.0
    assert solution.parameters["clock_offset_s"] == 0.0


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("camera_to_local_rotation", np.eye(2), "shape"),
        ("celestial_pole_camera", np.array([1.0, 0.0]), "shape"),
        (
            "camera_to_local_rotation",
            np.diag([1.0, 1.0, 2.0]),
            "orthonormal",
        ),
        (
            "camera_to_local_rotation",
            np.diag([1.0, 1.0, -1.0]),
            "determinant",
        ),
        ("celestial_pole_camera", np.array([0.0, 0.0, 2.0]), "unit vector"),
        ("rms_residual_pixels", -1.0, "non-negative"),
        ("matched_detection_count", -1, "non-negative"),
    ],
)
def test_solution_validation(field: str, value: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        make_solution(**{field: value})
