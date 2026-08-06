"""Compatibility boundary for reusable GONet Wizard functionality.

All imports from GONet Wizard are confined to this module so the numerical
package and documentation can be imported without eagerly importing the Wizard
application stack.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, cast

import numpy as np
from numpy.typing import NDArray

from gonet_astrometry.adapters.gonet_metadata import image_metadata_from_wizard
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
    meta: Mapping[object, object] | None

    def get_channel(self, channel_name: str) -> NDArray[np.float64]:
        """Return one two-dimensional native channel array."""
        ...

    def to_bayer_planes(
        self,
        fill_value: float | int = np.nan,
    ) -> dict[str, NDArray[np.float64]]:
        """Expand compact channels into sparse full-sensor planes."""
        ...


def load_gonet_file_raw(
    path: Path,
    *,
    parse_metadata: bool = False,
) -> RawGONetFile:
    """Load a native GONet image through ``GONetFileRaw``.

    Parameters
    ----------
    path
        Path to an original GONet ``.jpg`` file containing packed Bayer data.
    parse_metadata
        Whether the Wizard should parse the image metadata. Portal display
        loading leaves this disabled; scientific frame loading enables it.

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

    loaded = GONetFileRaw.from_file(Path(path), meta=parse_metadata)
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


def reconstruct_bayer_mosaic(gonet_file: RawGONetFile) -> NDArray[np.float64]:
    """Reconstruct the full native Bayer mosaic from Wizard channels.

    Parameters
    ----------
    gonet_file
        Loaded Wizard ``GONetFileRaw`` object in compact or expanded form.

    Returns
    -------
    numpy.ndarray
        Full-resolution two-dimensional BGGR sensor mosaic.

    Raises
    ------
    ValueError
        If channel planes differ in shape, overlap, or leave sensor pixels
        unassigned.
    """
    planes: dict[str, NDArray[np.float64]]
    if gonet_file.is_bayer_planes:
        planes = {
            channel: get_raw_channel(gonet_file, channel) for channel in GONET_CHANNELS
        }
    else:
        planes = {
            channel: np.asarray(values, dtype=np.float64)
            for channel, values in gonet_file.to_bayer_planes(fill_value=np.nan).items()
            if channel in GONET_CHANNELS
        }

    missing = set(GONET_CHANNELS).difference(planes)
    if missing:
        raise ValueError(
            "Wizard Bayer planes are missing channels: " + ", ".join(sorted(missing))
        )

    shapes = {plane.shape for plane in planes.values()}
    if len(shapes) != 1 or any(plane.ndim != 2 for plane in planes.values()):
        raise ValueError("Wizard Bayer planes must be matching two-dimensional arrays")

    ordered = np.stack([planes[channel] for channel in GONET_CHANNELS])
    finite = np.isfinite(ordered)
    sample_count = finite.sum(axis=0)
    if np.any(sample_count != 1):
        raise ValueError(
            "Each native sensor pixel must be assigned by exactly one Bayer channel"
        )

    mosaic = np.zeros(ordered.shape[1:], dtype=np.float64)
    for index in range(ordered.shape[0]):
        mosaic[finite[index]] = ordered[index][finite[index]]
    return mosaic


@dataclass(frozen=True, slots=True)
class GONetImageLoader:
    """Image-loader implementation backed by GONet Wizard."""

    def load(self, path: Path) -> ImageFrame:
        """Load one native GONet image as an astrometry frame."""
        return load_gonet_image(path)


def load_gonet_image(path: Path) -> ImageFrame:
    """Load a native GONet image as a scientific astrometry frame.

    Parameters
    ----------
    path
        Path to an original GONet ``.jpg`` image.

    Returns
    -------
    ImageFrame
        Full native Bayer mosaic with validated exposure and location metadata.

    Raises
    ------
    RuntimeError
        If GONet Wizard is unavailable.
    ValueError
        If the native channels or required metadata are invalid.
    """
    source_path = Path(path).expanduser()
    gonet_file = load_gonet_file_raw(source_path, parse_metadata=True)
    metadata = image_metadata_from_wizard(source_path, gonet_file.meta)
    return ImageFrame(
        data=reconstruct_bayer_mosaic(gonet_file),
        metadata=metadata,
    )


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
