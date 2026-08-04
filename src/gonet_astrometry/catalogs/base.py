"""Star-catalog interface."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol


@dataclass(frozen=True, slots=True)
class CatalogStar:
    """Minimal catalog-star record used by the solver.

    Parameters
    ----------
    identifier
        Stable catalog identifier.
    right_ascension_deg
        Right ascension in degrees in the catalog's documented reference
        frame and epoch.
    declination_deg
        Declination in degrees.
    magnitude
        Optional apparent magnitude in a documented passband.

    Raises
    ------
    ValueError
        If right ascension or declination lies outside its valid range.
    """

    identifier: str
    right_ascension_deg: float
    declination_deg: float
    magnitude: float | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.right_ascension_deg < 360.0:
            raise ValueError("right_ascension_deg must lie in [0, 360)")
        if not -90.0 <= self.declination_deg <= 90.0:
            raise ValueError("declination_deg must lie in [-90, 90]")


class StarCatalog(Protocol):
    """Protocol for local or remote star-catalog implementations."""

    def query_bright_stars(
        self,
        epoch: datetime,
        limiting_magnitude: float,
    ) -> tuple[CatalogStar, ...]:
        """Return stars bright enough to be useful for an observing session."""

        ...
