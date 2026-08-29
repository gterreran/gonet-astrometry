from pathlib import Path

import numpy as np
import pytest

from gonet_astrometry.detection.field_mask import (
    FieldMask,
    FieldMaskError,
    circular_exclusion_mask,
    load_field_mask,
    save_field_mask,
)

CONVENTION = (
    "x=column,y=row;origin=upper-left;+x=right;+y=down;"
    "pixel-centers-at-integer-coordinates"
)


def test_field_mask_round_trip_is_pickle_free(tmp_path: Path) -> None:
    excluded = np.zeros((6, 8), dtype=bool)
    excluded[2:4, 3:6] = True
    original = FieldMask(
        excluded=excluded,
        coordinate_convention=CONVENTION,
        description="synthetic obstruction",
    )

    path = save_field_mask(tmp_path / "field_mask.npz", original)
    with np.load(path, allow_pickle=False) as data:
        assert str(data["format"]) == "gonet-astrometry-field-mask"

    loaded = load_field_mask(path)
    assert loaded.coordinate_convention == CONVENTION
    assert loaded.description == "synthetic obstruction"
    assert np.array_equal(loaded.excluded, excluded)


def test_circular_exclusion_mask_uses_full_sensor_coordinates() -> None:
    mask = circular_exclusion_mask(
        (7, 9),
        center_x=4.0,
        center_y=3.0,
        radius_px=2.0,
        coordinate_convention=CONVENTION,
    )

    assert mask.excluded[3, 4]
    assert mask.excluded[3, 6]
    assert not mask.excluded[0, 0]


def test_field_mask_validates_shape_and_coordinate_convention() -> None:
    mask = FieldMask(
        excluded=np.zeros((6, 8), dtype=bool),
        coordinate_convention=CONVENTION,
    )

    with pytest.raises(FieldMaskError, match="sensor shape"):
        mask.validate_against((8, 6), CONVENTION)
    with pytest.raises(FieldMaskError, match="coordinate convention"):
        mask.validate_against((6, 8), "different")
