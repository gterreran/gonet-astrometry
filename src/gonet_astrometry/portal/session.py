"""Server-side runtime state for the astrometry portal.

The browser stores only path selections and lightweight controls. Native image
objects and NumPy arrays remain server-side so multi-file sessions do not copy
large detector payloads into Dash component state.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from gonet_astrometry.adapters.gonet_wizard import RawGONetFile
from gonet_astrometry.portal.discovery import DiscoveryResult, discover_gonet_files

RawLoader = Callable[[Path], RawGONetFile]
"""Callable that loads one native GONet file through the Wizard adapter."""

_EMPTY_DISCOVERY = DiscoveryResult((), (), (), ())


@dataclass
class PortalSession:
    """Runtime state for one astrometry portal process.

    Parameters
    ----------
    loader
        Function used to parse a selected native GONet image.
    discovery
        Current lightweight file-discovery result.

    Notes
    -----
    At most one native image object is cached. Discovery never loads image
    pixels, and loading a different file replaces the previous cached object.
    """

    loader: RawLoader = field(repr=False)
    discovery: DiscoveryResult = _EMPTY_DISCOVERY
    _loaded_path: Path | None = field(default=None, init=False, repr=False)
    _loaded_file: RawGONetFile | None = field(default=None, init=False, repr=False)
    _lock: Any = field(default_factory=threading.RLock, init=False, repr=False)

    @property
    def files(self) -> tuple[Path, ...]:
        """Return the currently discovered candidate files."""
        return self.discovery.files

    @property
    def loaded_path(self) -> Path | None:
        """Return the path of the currently cached native image, if any."""
        return self._loaded_path

    def discover(
        self,
        source_paths: Iterable[Path],
        *,
        recursive: bool = True,
    ) -> DiscoveryResult:
        """Discover candidate files and update the session catalog.

        Parameters
        ----------
        source_paths
            Explicit files and directories to inspect.
        recursive
            Whether nested subdirectories are searched.

        Returns
        -------
        DiscoveryResult
            Newly registered discovery result.
        """
        result = discover_gonet_files(source_paths, recursive=recursive)
        with self._lock:
            self.discovery = result
            if self._loaded_path not in result.files:
                self._loaded_path = None
                self._loaded_file = None
        return result

    def load(self, path: str | Path) -> RawGONetFile:
        """Load or reuse the selected native GONet image.

        Parameters
        ----------
        path
            Candidate path selected in the portal.

        Returns
        -------
        RawGONetFile
            Cached or newly loaded Wizard-native image object.

        Raises
        ------
        ValueError
            If ``path`` is not part of the current discovery catalog.
        OSError
            Propagated when the native loader cannot read the selected file.
        """
        normalized = Path(path).expanduser().resolve()
        with self._lock:
            if normalized not in self.discovery.files:
                raise ValueError(
                    "The selected image is not registered in the current input catalog."
                )
            if normalized == self._loaded_path and self._loaded_file is not None:
                return self._loaded_file

            loaded = self.loader(normalized)
            self._loaded_path = normalized
            self._loaded_file = loaded
            return loaded
