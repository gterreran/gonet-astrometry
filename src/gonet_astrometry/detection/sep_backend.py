"""SEP source-extraction backend."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from gonet_astrometry.detection.common import (
    coordinate_uncertainty,
    elongation_from_axes,
    finalize_catalog,
)
from gonet_astrometry.detection.config import DetectionConfig
from gonet_astrometry.detection.errors import DetectionBackendUnavailableError
from gonet_astrometry.detection.preprocessing import (
    PreparedDetectionImage,
    prepare_bayer_detection_image,
)
from gonet_astrometry.models.detection import Detection, DetectionCatalog
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
        """Prepare a native frame and run SEP extraction."""
        prepared = prepare_bayer_detection_image(frame, self.config)
        return self.detect_prepared(frame_identifier, prepared)

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
            peak = max(float(source["peak"]), 0.0)
            uncertainty = coordinate_uncertainty(self.config.fwhm_px, peak)
            flag = int(source["flag"])
            flags = () if flag == 0 else (f"sep:{flag}",)
            detections.append(
                Detection(
                    identifier=index,
                    x=float(source["x"]),
                    y=float(source["y"]),
                    flux=float(source["flux"]),
                    signal_to_noise=peak,
                    x_uncertainty=uncertainty,
                    y_uncertainty=uncertainty,
                    elongation=elongation_from_axes(
                        float(source["a"]),
                        float(source["b"]),
                    ),
                    flags=flags,
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
