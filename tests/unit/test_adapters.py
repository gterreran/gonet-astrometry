from pathlib import Path

import pytest

from gonet_astrometry.adapters.gonet_wizard import (
    load_gonet_image,
    load_grid_calibration,
)


def test_image_adapter_is_explicitly_unimplemented() -> None:
    with pytest.raises(NotImplementedError, match="image adapter"):
        load_gonet_image(Path("frame.raw"))


def test_grid_adapter_is_explicitly_unimplemented() -> None:
    with pytest.raises(NotImplementedError, match="Grid calibration adapter"):
        load_grid_calibration(Path("grid.json"))
