"""Compatibility boundary for reusable GONet Wizard functionality.

This module is intentionally free of direct imports from GONet Wizard until the
current Wizard API and Grid-calibration serialization format have been
inspected. All future Wizard-specific imports should remain confined here or in
closely related adapter modules.
"""

from __future__ import annotations

from pathlib import Path

from gonet_astrometry.models.frame import ImageFrame
from gonet_astrometry.models.grid import GridCalibration


def load_gonet_image(path: Path) -> ImageFrame:
    """Load a GONet image through the future Wizard compatibility adapter.

    Parameters
    ----------
    path
        Path to a native GONet image.

    Returns
    -------
    ImageFrame
        Loaded image and validated metadata.

    Raises
    ------
    NotImplementedError
        Until the GONet Wizard image API has been inspected and integrated.
    """
    raise NotImplementedError("GONet Wizard image adapter is not implemented")


def load_grid_calibration(path: Path) -> GridCalibration:
    """Load Grid calibration output through the future Wizard adapter.

    Parameters
    ----------
    path
        Path to serialized Grid calibration output.

    Returns
    -------
    GridCalibration
        Grid calibration normalized to the package's pixel-to-ray protocol.

    Raises
    ------
    NotImplementedError
        Until the Grid calibration serialization contract has been inspected.
    """
    raise NotImplementedError("Grid calibration adapter is not implemented")
