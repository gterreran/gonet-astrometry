from pathlib import Path

import pytest

from gonet_astrometry.models.frame import ObserverLocation
from gonet_astrometry.tracking.location_filter import (
    LocatedInput,
    is_zero_gps,
    select_dominant_location_group,
)


def _item(name: str, latitude: float, longitude: float, elevation: float = 180.0):
    return LocatedInput(
        Path(name),
        ObserverLocation(latitude, longitude, elevation),
    )


def test_zero_gps_requires_all_three_zero_values() -> None:
    assert is_zero_gps(ObserverLocation(0.0, 0.0, 0.0))
    assert not is_zero_gps(ObserverLocation(0.0, 0.0, 1.0))
    assert not is_zero_gps(ObserverLocation(0.0, 1.0, 0.0))


def test_dominant_location_group_skips_zero_and_remote_outliers() -> None:
    first = _item("a.jpg", 41.0000, -87.0000)
    second = _item("b.jpg", 41.0002, -87.0001)
    third = _item("c.jpg", 41.0001, -87.0002)
    remote = _item("wrong-folder.jpg", -30.0, 150.0)
    zero = _item("bad-gps.jpg", 0.0, 0.0, 0.0)

    result = select_dominant_location_group(
        [remote, second, zero, third, first],
        tolerance_m=100.0,
    )

    assert set(result.files) == {first.path, second.path, third.path}
    assert result.zero_gps == (zero,)
    assert result.outliers == (remote,)
    assert result.reference in {first, second, third}


def test_dominant_location_group_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="cannot be negative"):
        select_dominant_location_group([_item("a.jpg", 41.0, -87.0)], -1.0)

    with pytest.raises(ValueError, match="valid nonzero GPS"):
        select_dominant_location_group([_item("zero.jpg", 0.0, 0.0, 0.0)], 250.0)
