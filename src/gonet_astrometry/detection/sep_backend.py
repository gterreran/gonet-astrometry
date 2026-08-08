"""SEP source-extraction backend."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import degrees
from typing import Any

import numpy as np

from gonet_astrometry.detection.common import (
    coordinate_uncertainty,
    elongation_from_axes,
    finalize_catalog,
)
from gonet_astrometry.detection.config import DetectionConfig
from gonet_astrometry.detection.diagnostics import enrich_detection_catalog
from gonet_astrometry.detection.errors import DetectionBackendUnavailableError
from gonet_astrometry.detection.preprocessing import (
    PreparedDetectionImage,
    prepare_bayer_detection_image,
)
from gonet_astrometry.models.detection import (
    Detection,
    DetectionCatalog,
    DetectionDiagnostics,
)
from gonet_astrometry.models.frame import ImageFrame


@dataclass(frozen=True, slots=True)
class SEPDetector:
    """Detect and deblend sources using the SEP extraction library."""

    config: DetectionConfig = field(default_factory=DetectionConfig)

    @property
    def name(self) -> str:
        """Return the stable detector identifier."""
        return "sep"

    def detect(self, frame_identifier: str, frame: ImageFrame) -> DetectionCatalog:
        """Prepare a native frame, run SEP, and add common diagnostics."""
        prepared = prepare_bayer_detection_image(frame, self.config)
        catalog = self.detect_prepared(frame_identifier, prepared)
        return enrich_detection_catalog(catalog, prepared, self.config)

    def detect_prepared(
        self,
        frame_identifier: str,
        prepared: PreparedDetectionImage,
    ) -> DetectionCatalog:
        """Run SEP extraction on a shared preprocessed image."""
        sep = _load_sep()
        data = np.ascontiguousarray(prepared.data, dtype=np.float64)
        mask = np.ascontiguousarray(prepared.mask, dtype=np.bool_)
        objects = sep.extract(
            data,
            self.config.threshold_sigma,
            mask=mask,
            minarea=self.config.min_pixels,
            deblend_cont=0.005 if self.config.deblend else 1.0,
        )

        detections: list[Detection] = []
        for index, source in enumerate(objects, start=1):
            peak = max(_source_float(source, "peak", default=0.0), 0.0)
            uncertainty = coordinate_uncertainty(self.config.fwhm_px, peak)
            backend_flag = _source_int(source, "flag")
            flags = () if backend_flag in (None, 0) else (f"sep:{backend_flag}",)
            major = _source_optional_float(source, "a")
            minor = _source_optional_float(source, "b")
            theta = _source_optional_float(source, "theta")
            diagnostics = DetectionDiagnostics(
                peak_value=peak,
                area_pixels=_positive_source_int(source, ("npix", "tnpix")),
                semimajor_sigma_px=_positive_or_none(major),
                semiminor_sigma_px=_positive_or_none(minor),
                orientation_deg=None if theta is None else degrees(theta),
                ellipticity=_ellipticity_from_axes(major, minor),
                backend_flags=backend_flag,
            )
            detections.append(
                Detection(
                    identifier=index,
                    x=_source_float(source, "x"),
                    y=_source_float(source, "y"),
                    flux=_source_float(source, "flux"),
                    signal_to_noise=peak,
                    x_uncertainty=uncertainty,
                    y_uncertainty=uncertainty,
                    elongation=(
                        None
                        if major is None or minor is None
                        else elongation_from_axes(major, minor)
                    ),
                    flags=flags,
                    diagnostics=diagnostics,
                )
            )

        return finalize_catalog(
            frame_identifier,
            self.name,
            detections,
            config=self.config,
        )


def _load_sep() -> Any:
    """Import SEP lazily or raise a focused dependency error."""
    try:
        import sep
    except ImportError as exc:
        raise DetectionBackendUnavailableError(
            "SEP extraction requires the optional sep dependency."
        ) from exc
    return sep


def _source_optional_float(source: object, name: str) -> float | None:
    """Return an optional finite field from one SEP structured record."""
    names = getattr(getattr(source, "dtype", None), "names", None)
    if names is not None and name not in names:
        return None
    try:
        result = float(source[name])  # type: ignore[index]
    except (KeyError, TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _source_float(source: object, name: str, default: float = 0.0) -> float:
    """Return a finite SEP field or ``default``."""
    result = _source_optional_float(source, name)
    return default if result is None else result


def _source_int(source: object, name: str) -> int | None:
    """Return an optional integer SEP field."""
    value = _source_optional_float(source, name)
    return None if value is None else int(value)


def _positive_source_int(source: object, names: tuple[str, ...]) -> int | None:
    """Return the first positive integer field from a SEP record."""
    for name in names:
        value = _source_int(source, name)
        if value is not None and value > 0:
            return value
    return None


def _positive_or_none(value: float | None) -> float | None:
    """Return a positive finite value or ``None``."""
    if value is None or value <= 0:
        return None
    return value


def _ellipticity_from_axes(major: float | None, minor: float | None) -> float | None:
    """Return ``1 - b/a`` for valid SEP source axes."""
    if major is None or minor is None or major <= 0 or minor <= 0:
        return None
    return min(1.0, max(0.0, 1.0 - minor / major))
