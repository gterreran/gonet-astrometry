"""Reusable, portable mid-level workflow products."""

from gonet_astrometry.products.detections import (
    DetectionProduct,
    load_detection_product,
    save_detection_product,
)
from gonet_astrometry.products.errors import (
    ProductError,
    ProductFormatError,
    ProductMismatchError,
)
from gonet_astrometry.products.orientation import (
    OrientationProduct,
    load_orientation_product,
    save_orientation_product,
)
from gonet_astrometry.products.provenance import (
    DETECTION_PIPELINE_REVISION,
    HYBRID_FALLBACK_TRACKING_PIPELINE_REVISION,
    MULTICHANNEL_DETECTION_PIPELINE_REVISION,
    ORIENTATION_PIPELINE_REVISION,
    SIDEREAL_PIPELINE_REVISION,
    SPHERICAL_TRACKING_PIPELINE_REVISION,
    STELLAR_CAMERA_CALIBRATION_PIPELINE_REVISION,
    STELLAR_IDENTIFICATION_PIPELINE_REVISION,
    STELLAR_TRACKING_PIPELINE_REVISION,
    TRACKING_PIPELINE_REVISION,
    detection_product_id,
    hybrid_fallback_tracking_product_id,
    multichannel_detection_product_id,
    orientation_product_id,
    sidereal_product_id,
    spherical_tracking_product_id,
    stellar_camera_calibration_product_id,
    stellar_identification_product_id,
    stellar_tracking_product_id,
    tracking_product_id,
)
from gonet_astrometry.products.sidereal import (
    SiderealProduct,
    load_sidereal_product,
    save_sidereal_product,
)
from gonet_astrometry.products.stellar_camera_calibration import (
    StellarCameraCalibrationProduct,
    load_stellar_camera_calibration_product,
    save_stellar_camera_calibration_product,
)
from gonet_astrometry.products.stellar_identifications import (
    StellarIdentificationProduct,
    load_stellar_identification_product,
    save_stellar_identification_product,
)
from gonet_astrometry.products.stellar_tracking import (
    StellarTrackingProduct,
    TemporalTrackingProduct,
    load_stellar_tracking_product,
    load_temporal_tracking_product,
    save_stellar_tracking_product,
    save_temporal_tracking_product,
)
from gonet_astrometry.products.store import ProductStore
from gonet_astrometry.products.tracks import (
    TrackingProduct,
    load_tracking_product,
    save_tracking_product,
)

__all__ = [
    "DETECTION_PIPELINE_REVISION",
    "HYBRID_FALLBACK_TRACKING_PIPELINE_REVISION",
    "DetectionProduct",
    "ORIENTATION_PIPELINE_REVISION",
    "MULTICHANNEL_DETECTION_PIPELINE_REVISION",
    "OrientationProduct",
    "ProductError",
    "ProductFormatError",
    "ProductMismatchError",
    "ProductStore",
    "SIDEREAL_PIPELINE_REVISION",
    "SPHERICAL_TRACKING_PIPELINE_REVISION",
    "STELLAR_CAMERA_CALIBRATION_PIPELINE_REVISION",
    "STELLAR_IDENTIFICATION_PIPELINE_REVISION",
    "STELLAR_TRACKING_PIPELINE_REVISION",
    "SiderealProduct",
    "StellarCameraCalibrationProduct",
    "StellarIdentificationProduct",
    "StellarTrackingProduct",
    "TemporalTrackingProduct",
    "TRACKING_PIPELINE_REVISION",
    "TrackingProduct",
    "detection_product_id",
    "hybrid_fallback_tracking_product_id",
    "load_detection_product",
    "load_orientation_product",
    "load_sidereal_product",
    "load_stellar_camera_calibration_product",
    "load_stellar_identification_product",
    "load_stellar_tracking_product",
    "load_temporal_tracking_product",
    "load_tracking_product",
    "multichannel_detection_product_id",
    "save_detection_product",
    "save_orientation_product",
    "save_sidereal_product",
    "save_stellar_camera_calibration_product",
    "save_stellar_identification_product",
    "save_stellar_tracking_product",
    "save_temporal_tracking_product",
    "save_tracking_product",
    "orientation_product_id",
    "sidereal_product_id",
    "spherical_tracking_product_id",
    "stellar_camera_calibration_product_id",
    "stellar_identification_product_id",
    "stellar_tracking_product_id",
    "tracking_product_id",
]
