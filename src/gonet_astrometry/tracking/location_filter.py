"""Location-based input filtering for multi-image tracking runs."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from gonet_astrometry.models.frame import ObserverLocation
from gonet_astrometry.tracking.sequence import location_separation_m


@dataclass(frozen=True, slots=True)
class LocatedInput:
    """One candidate input path paired with its observing location."""

    path: Path
    location: ObserverLocation


@dataclass(frozen=True, slots=True)
class LocationGroupSelection:
    """Dominant observing-site group selected from candidate inputs.

    Parameters
    ----------
    kept
        Inputs belonging to the largest location-consistent group.
    zero_gps
        Inputs reporting the sentinel location ``(0, 0, 0)``.
    outliers
        Nonzero inputs outside the dominant location group.
    reference
        Input location used as the center of the dominant group.
    tolerance_m
        Maximum allowed separation from ``reference``.
    """

    kept: tuple[LocatedInput, ...]
    zero_gps: tuple[LocatedInput, ...]
    outliers: tuple[LocatedInput, ...]
    reference: LocatedInput
    tolerance_m: float

    @property
    def files(self) -> tuple[Path, ...]:
        """Return paths retained for detection and tracking."""
        return tuple(item.path for item in self.kept)


def is_zero_gps(location: ObserverLocation) -> bool:
    """Return whether ``location`` is the camera's all-zero GPS sentinel."""
    return (
        math.isclose(location.latitude_deg, 0.0, abs_tol=1.0e-12)
        and math.isclose(location.longitude_deg, 0.0, abs_tol=1.0e-12)
        and math.isclose(location.elevation_m, 0.0, abs_tol=1.0e-9)
    )


def select_dominant_location_group(
    inputs: tuple[LocatedInput, ...] | list[LocatedInput],
    tolerance_m: float,
) -> LocationGroupSelection:
    """Keep the largest group consistent with one observing location.

    Every nonzero GPS location is considered as a possible group reference.
    The candidate containing the most inputs within ``tolerance_m`` wins. Ties
    prefer the more compact group, measured by summed separation from the
    reference. This makes the selection deterministic and robust to one or a
    few misplaced files without requiring clustering dependencies.
    """
    if tolerance_m < 0:
        raise ValueError("Location tolerance cannot be negative")

    ordered = tuple(sorted(inputs, key=lambda item: str(item.path).casefold()))
    zero_gps = tuple(item for item in ordered if is_zero_gps(item.location))
    valid = tuple(item for item in ordered if not is_zero_gps(item.location))
    if not valid:
        raise ValueError("No candidate images have a valid nonzero GPS location")

    best_reference = valid[0]
    best_group: tuple[LocatedInput, ...] = ()
    best_compactness = math.inf
    for reference in valid:
        members = tuple(
            item
            for item in valid
            if location_separation_m(reference.location, item.location) <= tolerance_m
        )
        compactness = sum(
            location_separation_m(reference.location, item.location) for item in members
        )
        if len(members) > len(best_group) or (
            len(members) == len(best_group) and compactness < best_compactness
        ):
            best_reference = reference
            best_group = members
            best_compactness = compactness

    kept_paths = {item.path for item in best_group}
    outliers = tuple(item for item in valid if item.path not in kept_paths)
    return LocationGroupSelection(
        kept=best_group,
        zero_gps=zero_gps,
        outliers=outliers,
        reference=best_reference,
        tolerance_m=tolerance_m,
    )
