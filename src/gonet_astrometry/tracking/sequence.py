"""Chronological detection-sequence models for star tracking."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from gonet_astrometry.models.detection import DetectionCatalog
from gonet_astrometry.models.frame import ImageFrame, ObserverLocation

_EARTH_RADIUS_M = 6_371_000.0


@dataclass(frozen=True, slots=True)
class DetectionEpoch:
    """One source catalog paired with its observing metadata.

    Parameters
    ----------
    frame_identifier
        Stable identifier matching
        ``DetectionCatalog.frame_identifier``.
    source_path
        Source GONet image path.
    exposure_midpoint
        Timezone-aware exposure midpoint used for temporal association.
    location
        Geographic observing location.
    image_shape
        Full native Bayer sensor shape as ``(rows, columns)``.
    sensor_orientation
        Sensor-coordinate convention inherited from the scientific frame.
    catalog
        Source detections measured in this epoch.
    """

    frame_identifier: str
    source_path: Path
    exposure_midpoint: datetime
    location: ObserverLocation
    image_shape: tuple[int, int]
    sensor_orientation: str
    catalog: DetectionCatalog

    def __post_init__(self) -> None:
        if self.exposure_midpoint.tzinfo is None:
            raise ValueError("DetectionEpoch.exposure_midpoint must be timezone-aware")
        if self.catalog.frame_identifier != self.frame_identifier:
            raise ValueError(
                "DetectionEpoch frame_identifier must match its detection catalog"
            )
        if self.image_shape[0] <= 0 or self.image_shape[1] <= 0:
            raise ValueError("DetectionEpoch.image_shape must be positive")

    @classmethod
    def from_frame(
        cls,
        frame_identifier: str,
        frame: ImageFrame,
        catalog: DetectionCatalog,
    ) -> DetectionEpoch:
        """Build an epoch from one scientific frame and detection catalog."""
        source_path = frame.metadata.source_path
        if source_path is None:
            source_path = Path(frame_identifier)
        return cls(
            frame_identifier=frame_identifier,
            source_path=Path(source_path),
            exposure_midpoint=frame.metadata.exposure_midpoint,
            location=frame.metadata.location,
            image_shape=frame.shape,
            sensor_orientation=frame.metadata.sensor_orientation,
            catalog=catalog,
        )


@dataclass(frozen=True, slots=True)
class DetectionSequence:
    """Validated chronological source catalogs from one observing sequence.

    Parameters
    ----------
    epochs
        Chronologically ordered detection epochs.

    Raises
    ------
    ValueError
        If no epochs are supplied, identifiers or timestamps are duplicated,
        image geometry changes, or timestamps are not increasing.
    """

    epochs: tuple[DetectionEpoch, ...]

    def __post_init__(self) -> None:
        if not self.epochs:
            raise ValueError("A DetectionSequence requires at least one epoch")
        frame_ids = [epoch.frame_identifier for epoch in self.epochs]
        if len(set(frame_ids)) != len(frame_ids):
            raise ValueError("DetectionSequence frame identifiers must be unique")
        times = [epoch.exposure_midpoint for epoch in self.epochs]
        if any(
            later <= earlier for earlier, later in zip(times, times[1:], strict=False)
        ):
            raise ValueError("DetectionSequence timestamps must increase strictly")
        shape = self.epochs[0].image_shape
        orientation = self.epochs[0].sensor_orientation
        for epoch in self.epochs[1:]:
            if epoch.image_shape != shape:
                raise ValueError("DetectionSequence image shapes must match")
            if epoch.sensor_orientation != orientation:
                raise ValueError("DetectionSequence sensor orientations must match")

    @property
    def duration_seconds(self) -> float:
        """Return elapsed time between the first and last exposure midpoints."""
        delta = self.epochs[-1].exposure_midpoint - self.epochs[0].exposure_midpoint
        return delta.total_seconds()

    @property
    def epoch_intervals_seconds(self) -> tuple[float, ...]:
        """Return actual elapsed seconds between consecutive exposure midpoints."""
        return tuple(
            (later.exposure_midpoint - earlier.exposure_midpoint).total_seconds()
            for earlier, later in zip(self.epochs, self.epochs[1:], strict=False)
        )

    @property
    def total_detections(self) -> int:
        """Return the number of source candidates across all epochs."""
        return sum(len(epoch.catalog) for epoch in self.epochs)

    def catalog_for(self, frame_identifier: str) -> DetectionCatalog | None:
        """Return the catalog for ``frame_identifier`` when present."""
        for epoch in self.epochs:
            if epoch.frame_identifier == frame_identifier:
                return epoch.catalog
        return None

    @classmethod
    def from_epochs(
        cls,
        epochs: tuple[DetectionEpoch, ...] | list[DetectionEpoch],
        *,
        location_tolerance_m: float | None = None,
    ) -> DetectionSequence:
        """Sort epochs chronologically and optionally validate location stability."""
        ordered = tuple(sorted(epochs, key=lambda epoch: epoch.exposure_midpoint))
        sequence = cls(ordered)
        if location_tolerance_m is not None:
            sequence.validate_locations(location_tolerance_m)
        return sequence

    def validate_locations(self, tolerance_m: float) -> None:
        """Require epochs to fit within one tolerance-radius observing site.

        The reference is chosen from the sequence locations to minimize the
        maximum separation to all other epochs. This avoids making validation
        depend on whichever file happens to be chronologically first.
        """
        if tolerance_m < 0:
            raise ValueError("Location tolerance cannot be negative")
        locations = tuple(epoch.location for epoch in self.epochs)
        best_maximum = min(
            max(location_separation_m(reference, other) for other in locations)
            for reference in locations
        )
        if best_maximum > tolerance_m:
            raise ValueError(
                "DetectionSequence observing locations differ by "
                f"{best_maximum:.1f} m, exceeding the {tolerance_m:.1f} m tolerance"
            )


def location_separation_m(
    first: ObserverLocation,
    second: ObserverLocation,
) -> float:
    """Return approximate three-dimensional separation between two GPS locations."""
    lat1 = math.radians(first.latitude_deg)
    lat2 = math.radians(second.latitude_deg)
    dlat = lat2 - lat1
    dlon = math.radians(second.longitude_deg - first.longitude_deg)
    hav = (
        math.sin(dlat / 2.0) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2.0) ** 2
    )
    surface = 2.0 * _EARTH_RADIUS_M * math.asin(min(1.0, math.sqrt(hav)))
    dz = second.elevation_m - first.elevation_m
    return math.hypot(surface, dz)
