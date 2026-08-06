"""Compatibility boundary for reusable GONet Wizard functionality.

All imports from GONet Wizard are confined to this module so the numerical
package and documentation can be imported without eagerly importing the Wizard
application stack.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Protocol, cast

import numpy as np
from numpy.typing import NDArray

from gonet_astrometry.models.frame import ImageFrame
from gonet_astrometry.models.grid import GridCalibration

GONetChannel = Literal["blue", "green1", "green2", "red"]
"""Name of one native channel preserved by ``GONetFileRaw``."""

GONET_CHANNELS: tuple[GONetChannel, ...] = (
    "blue",
    "green1",
    "green2",
    "red",
)
"""Native BGGR channels exposed by ``GONetFileRaw``."""


class RawGONetFile(Protocol):
    """Structural interface used from a Wizard ``GONetFileRaw`` instance."""

    filename: str
    is_bayer_planes: bool

    def get_channel(self, channel_name: str) -> NDArray[np.float64]:
        """Return one two-dimensional native channel array."""
        ...


def load_gonet_file_raw(path: Path) -> RawGONetFile:
    """Load a native GONet image through ``GONetFileRaw``.

    Parameters
    ----------
    path
        Path to an original GONet ``.jpg`` file containing packed Bayer data.

    Returns
    -------
    RawGONetFile
        Wizard object preserving the blue, two green, and red samples as
        separate compact channel arrays.

    Raises
    ------
    RuntimeError
        If GONet Wizard is not importable in the active environment.
    FileNotFoundError
        If ``path`` does not identify a file.
    ValueError
        If the file is not a supported original GONet ``.jpg``.

    Notes
    -----
    Metadata parsing is disabled for this initial display path. This guarantees
    that ``gonet-astrometry`` reaches the image exclusively through the native
    Wizard Bayer parser and does not invoke a separate JPEG preview loader.
    """
    try:
        from GONet_Wizard.GONet_utils import GONetFileRaw
    except ImportError as exc:
        raise RuntimeError(
            "GONet Wizard is required to load native GONet images. "
            "Activate an environment in which GONet_Wizard is installed."
        ) from exc

    loaded = GONetFileRaw.from_file(Path(path), meta=False)
    return cast(RawGONetFile, loaded)


def get_raw_channel(
    gonet_file: RawGONetFile,
    channel: GONetChannel,
) -> NDArray[np.float64]:
    """Return and validate one native channel from a loaded GONet file.

    Parameters
    ----------
    gonet_file
        Loaded Wizard ``GONetFileRaw`` object.
    channel
        Native channel to retrieve.

    Returns
    -------
    numpy.ndarray
        Two-dimensional floating-point channel data.

    Raises
    ------
    ValueError
        If ``channel`` is invalid or the Wizard returns a non-two-dimensional
        array.
    """
    if channel not in GONET_CHANNELS:
        raise ValueError(
            f"Unsupported GONet channel {channel!r}; expected one of "
            f"{', '.join(GONET_CHANNELS)}"
        )

    data = np.asarray(gonet_file.get_channel(channel), dtype=np.float64)
    if data.ndim != 2:
        raise ValueError(f"GONet channel {channel!r} must be two-dimensional")
    return data


def load_gonet_image(path: Path) -> ImageFrame:
    """Load a GONet image as a scientific astrometry frame.

    Parameters
    ----------
    path
        Path to a native GONet image.

    Returns
    -------
    ImageFrame
        Loaded image and validated astrometric metadata.

    Raises
    ------
    NotImplementedError
        Until the conversion from ``GONetFileRaw`` metadata and Bayer samples
        into the package's complete
        :class:`~gonet_astrometry.models.frame.ImageFrame` contract is defined.
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
