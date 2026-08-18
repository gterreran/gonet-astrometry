"""Server-side runtime state for the astrometry portal.

The browser stores only path selections and lightweight controls. Native image
objects, scientific frames, detections, and NumPy arrays remain server-side so
multi-file sessions do not copy large detector payloads into Dash component
state.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any

from gonet_astrometry.adapters.gonet_wizard import RawGONetFile, load_gonet_image
from gonet_astrometry.detection.base import SourceDetector
from gonet_astrometry.detection.config import DetectionConfig
from gonet_astrometry.detection.diagnostics import enrich_detection_catalog
from gonet_astrometry.detection.preprocessing import (
    PreparedDetectionImage,
    prepare_bayer_detection_image,
)
from gonet_astrometry.detection.registry import create_detector
from gonet_astrometry.detection.timing import DetectionTiming
from gonet_astrometry.models.detection import DetectionCatalog
from gonet_astrometry.models.frame import ImageFrame
from gonet_astrometry.portal.discovery import DiscoveryResult, discover_gonet_files
from gonet_astrometry.tracking.config import TrackingConfig
from gonet_astrometry.tracking.image_plane import (
    ImagePlaneTracker,
    ImagePlaneTrackingResult,
)
from gonet_astrometry.tracking.sequence import DetectionEpoch, DetectionSequence

RawLoader = Callable[[Path], RawGONetFile]
"""Callable that loads one native GONet file through the Wizard adapter."""

FrameLoader = Callable[[Path], ImageFrame]
"""Callable that loads one full scientific image frame."""

DetectorFactory = Callable[[str, DetectionConfig | None], SourceDetector]
"""Callable that constructs one registered detector backend."""

Clock = Callable[[], float]
"""Monotonic high-resolution clock used for performance measurements."""

_EMPTY_DISCOVERY = DiscoveryResult((), (), (), ())

logger = logging.getLogger(__name__)


@dataclass
class PortalSession:
    """Runtime state for one astrometry portal process.

    Parameters
    ----------
    loader
        Function used to parse a selected native GONet image for display.
    frame_loader
        Function used to load the full Bayer mosaic and scientific metadata.
    detector_factory
        Factory used to construct source-detection backends.
    clock
        Monotonic clock used to measure source-detection performance.
    discovery
        Current lightweight file-discovery result.

    Notes
    -----
    At most one native display image and one scientific frame are cached as
    large pixel arrays. Multi-image processing loads additional scientific
    frames sequentially and retains only their lightweight source catalogs and
    observing metadata. Changing the displayed file therefore does not require
    loading the whole observing sequence into memory.
    """

    loader: RawLoader = field(repr=False)
    frame_loader: FrameLoader = field(default=load_gonet_image, repr=False)
    detector_factory: DetectorFactory = field(default=create_detector, repr=False)
    clock: Clock = field(default=perf_counter, repr=False)
    discovery: DiscoveryResult = _EMPTY_DISCOVERY
    _loaded_path: Path | None = field(default=None, init=False, repr=False)
    _loaded_file: RawGONetFile | None = field(default=None, init=False, repr=False)
    _frame_path: Path | None = field(default=None, init=False, repr=False)
    _frame: ImageFrame | None = field(default=None, init=False, repr=False)
    _detection_catalog: DetectionCatalog | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _prepared_image: PreparedDetectionImage | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _detection_timing: DetectionTiming | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _sequence_key: tuple[tuple[Path, ...], str, DetectionConfig] | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _detection_sequence: DetectionSequence | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _tracking_result: ImagePlaneTrackingResult | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _lock: Any = field(default_factory=threading.RLock, init=False, repr=False)

    @property
    def files(self) -> tuple[Path, ...]:
        """Return the currently discovered candidate files."""
        return self.discovery.files

    @property
    def loaded_path(self) -> Path | None:
        """Return the path of the currently cached native image, if any."""
        return self._loaded_path

    @property
    def frame_path(self) -> Path | None:
        """Return the path of the cached scientific frame, if any."""
        return self._frame_path

    @property
    def detection_catalog(self) -> DetectionCatalog | None:
        """Return the most recently computed source catalog, if any."""
        return self._detection_catalog

    @property
    def prepared_image(self) -> PreparedDetectionImage | None:
        """Return preprocessing diagnostics for the latest detection run."""
        return self._prepared_image

    @property
    def detection_timing(self) -> DetectionTiming | None:
        """Return timing for the most recent successful detection run."""
        return self._detection_timing

    @property
    def detection_sequence(self) -> DetectionSequence | None:
        """Return the most recently processed multi-image detection sequence."""
        return self._detection_sequence

    @property
    def tracking_result(self) -> ImagePlaneTrackingResult | None:
        """Return the most recently built image-plane tracklets."""
        return self._tracking_result

    def catalog_for_path(self, path: str | Path) -> DetectionCatalog | None:
        """Return a cached single- or multi-image catalog for ``path``."""
        normalized = Path(path).expanduser().resolve()
        if (
            self._detection_catalog is not None
            and self._detection_catalog.frame_identifier == str(normalized)
        ):
            return self._detection_catalog
        if self._detection_sequence is not None:
            return self._detection_sequence.catalog_for(str(normalized))
        return None

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
            previous_files = self.discovery.files
            self.discovery = result
            if self._loaded_path not in result.files:
                self._clear_loaded_state()
            if result.files != previous_files:
                self._clear_sequence_state()
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
        normalized = self._registered_path(path)
        with self._lock:
            if normalized == self._loaded_path and self._loaded_file is not None:
                return self._loaded_file

            loaded = self.loader(normalized)
            self._loaded_path = normalized
            self._loaded_file = loaded
            if normalized != self._frame_path:
                self._frame_path = None
                self._frame = None
                self._detection_catalog = None
                self._prepared_image = None
                self._detection_timing = None
            return loaded

    def load_frame(self, path: str | Path) -> ImageFrame:
        """Load or reuse the selected full-resolution scientific frame."""
        normalized = self._registered_path(path)
        with self._lock:
            if normalized == self._frame_path and self._frame is not None:
                return self._frame

            frame = self.frame_loader(normalized)
            self._frame_path = normalized
            self._frame = frame
            self._detection_catalog = None
            self._prepared_image = None
            self._detection_timing = None
            return frame

    def detect(
        self,
        path: str | Path,
        detector_identifier: str,
        config: DetectionConfig,
    ) -> DetectionCatalog:
        """Run, time, and cache one detector on the selected scientific frame.

        The detector time starts after the scientific frame has been loaded or
        retrieved from cache, so comparisons are not biased by whichever
        backend happens to run first. Detector construction, shared Bayer
        preprocessing, and backend-specific execution are timed separately.
        """
        normalized = self._registered_path(path)
        with self._lock:
            self._detection_timing = None

        total_start = self.clock()
        frame = self.load_frame(normalized)
        frame_ready = self.clock()
        detector = self.detector_factory(detector_identifier, config)
        detector_ready = self.clock()
        prepared = prepare_bayer_detection_image(frame, config)
        prepared_ready = self.clock()
        catalog = detector.detect_prepared(str(normalized), prepared)
        detection_finished = self.clock()
        catalog = enrich_detection_catalog(catalog, prepared, config)

        timing = DetectionTiming(
            frame_identifier=str(normalized),
            detector_name=catalog.detector_name,
            frame_load_seconds=frame_ready - total_start,
            detector_setup_seconds=detector_ready - frame_ready,
            preprocessing_seconds=prepared_ready - detector_ready,
            backend_seconds=detection_finished - prepared_ready,
            total_seconds=detection_finished - total_start,
            source_count=len(catalog),
        )
        with self._lock:
            self._detection_catalog = catalog
            self._prepared_image = prepared
            self._detection_timing = timing
        return catalog

    def detect_sequence(
        self,
        paths: Iterable[str | Path],
        detector_identifier: str,
        config: DetectionConfig,
    ) -> DetectionSequence:
        """Detect a sequence lazily, retaining only metadata and catalogs.

        Scientific frames and prepared detection arrays are processed one at a
        time. The returned sequence therefore scales primarily with source
        catalog size rather than raw image size. Repeating the same request
        reuses the cached sequence.
        """
        normalized = tuple(
            sorted(
                dict.fromkeys(self._registered_path(path) for path in paths),
                key=str,
            )
        )
        if len(normalized) < 2:
            raise ValueError("Tracking requires at least two registered images")

        key = (normalized, detector_identifier, config)
        with self._lock:
            if key == self._sequence_key and self._detection_sequence is not None:
                logger.info(
                    "Reusing cached detections for %d sequence images.",
                    len(normalized),
                )
                return self._detection_sequence

        detector = self.detector_factory(detector_identifier, config)
        epochs: list[DetectionEpoch] = []
        logger.info(
            "Detecting %d images sequentially with %s; "
            "raw frame arrays are not retained.",
            len(normalized),
            detector_identifier,
        )
        for index, path in enumerate(normalized, start=1):
            frame = self._frame_for_sequence(path)
            prepared = prepare_bayer_detection_image(frame, config)
            catalog = detector.detect_prepared(str(path), prepared)
            catalog = enrich_detection_catalog(catalog, prepared, config)
            with self._lock:
                if path == self._loaded_path:
                    self._frame_path = path
                    self._frame = frame
                    self._detection_catalog = catalog
                    self._prepared_image = prepared
                    self._detection_timing = None
            epoch = DetectionEpoch.from_frame(str(path), frame, catalog)
            epochs.append(epoch)
            logger.info(
                "Sequence detection %d/%d | %s | %d candidates | midpoint %s",
                index,
                len(normalized),
                path.name,
                len(catalog),
                epoch.exposure_midpoint.isoformat(),
            )

        sequence = DetectionSequence.from_epochs(epochs)
        with self._lock:
            self._sequence_key = key
            self._detection_sequence = sequence
            self._tracking_result = None
        return sequence

    def build_tracks(
        self,
        paths: Iterable[str | Path],
        detector_identifier: str,
        detection_config: DetectionConfig,
        tracking_config: TrackingConfig,
    ) -> ImagePlaneTrackingResult:
        """Detect the requested sequence when needed and build image-plane tracks."""
        sequence = self.detect_sequence(paths, detector_identifier, detection_config)
        sequence.validate_locations(tracking_config.location_tolerance_m)
        result = ImagePlaneTracker(tracking_config).track(sequence)
        with self._lock:
            self._tracking_result = result
        return result

    def _frame_for_sequence(self, path: Path) -> ImageFrame:
        """Load one sequence frame without replacing the portal's display cache."""
        with self._lock:
            if path == self._frame_path and self._frame is not None:
                return self._frame
        return self.frame_loader(path)

    def _registered_path(self, path: str | Path) -> Path:
        """Normalize ``path`` and require it in the discovery catalog."""
        normalized = Path(path).expanduser().resolve()
        if normalized not in self.discovery.files:
            raise ValueError(
                "The selected image is not registered in the current input catalog."
            )
        return normalized

    def _clear_loaded_state(self) -> None:
        """Clear cached display, scientific-frame, and detection state."""
        self._loaded_path = None
        self._loaded_file = None
        self._frame_path = None
        self._frame = None
        self._detection_catalog = None
        self._prepared_image = None
        self._detection_timing = None

    def _clear_sequence_state(self) -> None:
        """Clear cached multi-image detections and image-plane tracks."""
        self._sequence_key = None
        self._detection_sequence = None
        self._tracking_result = None
