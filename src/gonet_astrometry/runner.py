"""Non-interactive detection and bootstrap-tracking workflow.

This module exposes the backend used by the ``gonet-astrometry run`` command.
It deliberately avoids importing Dash or pywebview so batch experiments can be
run from a terminal with only the numerical, detection, and reporting stack.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from gonet_astrometry.adapters.grid_calibration import load_grid_calibration
from gonet_astrometry.adapters.gonet_wizard import (
    GONetChannel,
    get_raw_channel,
    load_gonet_file_raw,
    load_gonet_image,
    load_gonet_metadata,
)
from gonet_astrometry.catalogs import load_or_fetch_bright_star_catalog
from gonet_astrometry.detection.base import SourceDetector
from gonet_astrometry.detection.config import DetectionConfig
from gonet_astrometry.detection.diagnostics import enrich_detection_catalog
from gonet_astrometry.detection.preprocessing import (
    PreparedDetectionImage,
    prepare_bayer_detection_image,
)
from gonet_astrometry.detection.registry import create_detector
from gonet_astrometry.diagnostics.report import write_tracking_report_pdf
from gonet_astrometry.io.discovery import discover_gonet_files
from gonet_astrometry.models.detection import DetectionCatalog
from gonet_astrometry.models.frame import ImageFrame, ImageMetadata
from gonet_astrometry.models.grid import GridCalibration
from gonet_astrometry.products import (
    DetectionProduct,
    OrientationProduct,
    ProductError,
    ProductStore,
    SiderealProduct,
    TrackingProduct,
    detection_product_id,
    load_detection_product,
    load_orientation_product,
    load_sidereal_product,
    load_tracking_product,
    orientation_product_id,
    save_detection_product,
    save_orientation_product,
    save_sidereal_product,
    save_tracking_product,
    sidereal_product_id,
    tracking_product_id,
)
from gonet_astrometry.solving.orientation import (
    AbsoluteOrientationSolution,
    AbsoluteOrientationSolver,
    OrientationFitConfig,
)
from gonet_astrometry.solving.sidereal import (
    SiderealAxisFitter,
    SiderealFitConfig,
    SiderealRotationSolution,
)
from gonet_astrometry.tracking.config import TrackingConfig
from gonet_astrometry.tracking.location_filter import (
    LocatedInput,
    LocationGroupSelection,
    select_dominant_location_group,
)
from gonet_astrometry.tracking.image_plane import (
    ImagePlaneTracker,
    ImagePlaneTrackingResult,
)
from gonet_astrometry.tracking.sequence import (
    DetectionEpoch,
    DetectionSequence,
    location_separation_m,
)
from gonet_astrometry.tracking.temporal import (
    sequence_timing_diagnostics,
    track_population_diagnostics,
)

logger = logging.getLogger(__name__)

FrameLoader = Callable[[Path], ImageFrame]
MetadataLoader = Callable[[Path], ImageMetadata]
DetectorFactory = Callable[[str, DetectionConfig | None], SourceDetector]
Clock = Callable[[], float]


@dataclass(frozen=True, slots=True)
class DetectionRunArtifacts:
    """Scientific products retained after sequential source detection.

    Parameters
    ----------
    files
        Unique input files processed by the detection pass.
    algorithm
        Registered detector identifier.
    sequence
        Chronologically validated detection sequence.
    reference_path
        Input image used as the static report background.
    reference_catalog
        Detections belonging to ``reference_path``.
    reference_prepared
        Shared preprocessing products for ``reference_path``.
    """

    files: tuple[Path, ...]
    algorithm: str
    sequence: DetectionSequence
    reference_path: Path
    reference_catalog: DetectionCatalog
    reference_prepared: PreparedDetectionImage


@dataclass(frozen=True, slots=True)
class TrackingRunArtifacts:
    """Scientific products retained from one non-interactive tracking run.

    Parameters
    ----------
    files
        Unique input files processed by the run.
    algorithm
        Registered detector identifier.
    sequence
        Chronologically validated detection sequence.
    tracking_result
        Bootstrap image-plane tracklets.
    reference_path
        Input image used as the static report background.
    reference_catalog
        Detections belonging to ``reference_path``.
    reference_prepared
        Shared preprocessing products for ``reference_path``.
    """

    files: tuple[Path, ...]
    algorithm: str
    sequence: DetectionSequence
    tracking_result: ImagePlaneTrackingResult
    reference_path: Path
    reference_catalog: DetectionCatalog
    reference_prepared: PreparedDetectionImage
    grid_calibration: GridCalibration | None = None
    sidereal_solution: SiderealRotationSolution | None = None
    sidereal_fit_config: SiderealFitConfig | None = None
    orientation_solution: AbsoluteOrientationSolution | None = None
    orientation_fit_config: OrientationFitConfig | None = None


@dataclass(frozen=True, slots=True)
class RunSummary:
    """Compact result returned after writing a command-line PDF report."""

    output_path: Path
    file_count: int
    detection_count: int
    track_count: int
    assigned_detection_count: int
    unassigned_detection_count: int
    skipped_file_count: int = 0
    zero_gps_count: int = 0
    location_outlier_count: int = 0
    metadata_error_count: int = 0
    detection_product_path: Path | None = None
    tracking_product_path: Path | None = None
    sidereal_product_path: Path | None = None
    orientation_product_path: Path | None = None
    reused_detection_product: bool = False
    reused_tracking_product: bool = False
    reused_sidereal_product: bool = False
    reused_orientation_product: bool = False


@dataclass(frozen=True, slots=True)
class LocationPreflight:
    """Location filtering result computed before source detection starts."""

    selection: LocationGroupSelection
    metadata_errors: tuple[tuple[Path, str], ...] = ()

    @property
    def files(self) -> tuple[Path, ...]:
        """Return files retained for the expensive detection pass."""
        return self.selection.files


def preflight_tracking_locations(
    paths: Iterable[Path],
    tolerance_m: float,
    *,
    metadata_loader: MetadataLoader = load_gonet_metadata,
) -> LocationPreflight:
    """Filter obvious GPS failures and keep the dominant observing site.

    Metadata is inspected before detector preprocessing begins. Files whose
    metadata cannot be interpreted are skipped with their error recorded.
    Locations equal to ``(0, 0, 0)`` are treated as invalid camera GPS fixes.
    Of the remaining files, only the largest group lying within
    ``tolerance_m`` of one representative location is retained.
    """
    located: list[LocatedInput] = []
    errors: list[tuple[Path, str]] = []
    ordered = tuple(
        sorted(
            {Path(path).expanduser().resolve() for path in paths},
            key=lambda item: str(item).casefold(),
        )
    )
    for index, path in enumerate(ordered, start=1):
        try:
            metadata = metadata_loader(path)
        except (OSError, RuntimeError, ValueError) as exc:
            errors.append((path, str(exc)))
            logger.warning(
                "Location preflight %d/%d | skipping %s | metadata error: %s",
                index,
                len(ordered),
                path.name,
                exc,
            )
            continue
        located.append(LocatedInput(path=path, location=metadata.location))

    selection = select_dominant_location_group(located, tolerance_m)
    return LocationPreflight(selection=selection, metadata_errors=tuple(errors))


def _log_location_preflight(preflight: LocationPreflight) -> None:
    """Log retained and skipped files before expensive source detection."""
    selection = preflight.selection
    reference = selection.reference.location
    logger.info(
        "Location preflight | kept %d | zero GPS %d | location outliers %d | "
        "metadata errors %d | reference %.6f, %.6f, %.1f m | tolerance %.1f m",
        len(selection.kept),
        len(selection.zero_gps),
        len(selection.outliers),
        len(preflight.metadata_errors),
        reference.latitude_deg,
        reference.longitude_deg,
        reference.elevation_m,
        selection.tolerance_m,
    )
    for item in selection.zero_gps:
        logger.warning("Skipping %s because GPS is 0,0,0.", item.path)
    for item in selection.outliers:
        separation = location_separation_m(
            selection.reference.location, item.location
        )
        logger.warning(
            "Skipping %s because GPS is %.1f m from the dominant site.",
            item.path,
            separation,
        )


def execute_detection_run(
    paths: Iterable[Path],
    algorithm: str,
    detection_config: DetectionConfig,
    tracking_config: TrackingConfig,
    *,
    frame_loader: FrameLoader = load_gonet_image,
    detector_factory: DetectorFactory = create_detector,
    clock: Clock = perf_counter,
) -> DetectionRunArtifacts:
    """Detect a sequence sequentially without retaining all scientific frames.

    The full native image for each epoch is released after preprocessing and
    source extraction. Only the detection catalog and astrometric metadata are
    retained, plus the first prepared image used by the diagnostic report.
    """
    files = tuple(
        sorted(
            {Path(path).expanduser().resolve() for path in paths},
            key=lambda item: str(item).casefold(),
        )
    )
    if len(files) < 2:
        raise ValueError("The run command requires at least two GONet images")

    detector = detector_factory(algorithm, detection_config)
    epochs: list[DetectionEpoch] = []
    reference_prepared: PreparedDetectionImage | None = None
    reference_catalog: DetectionCatalog | None = None

    for index, path in enumerate(files, start=1):
        frame = frame_loader(path)
        preprocess_start = clock()
        prepared = prepare_bayer_detection_image(frame, detection_config)
        preprocess_finished = clock()
        catalog = detector.detect_prepared(str(path), prepared)
        backend_finished = clock()
        catalog = enrich_detection_catalog(catalog, prepared, detection_config)

        if reference_prepared is None:
            reference_prepared = prepared
            reference_catalog = catalog

        epoch = DetectionEpoch.from_frame(str(path), frame, catalog)
        epochs.append(epoch)
        logger.info(
            "Detection %d/%d | %s | %d candidates | preprocess %.3f s | "
            "backend %.3f s | midpoint %s",
            index,
            len(files),
            path.name,
            len(catalog),
            preprocess_finished - preprocess_start,
            backend_finished - preprocess_finished,
            epoch.exposure_midpoint.isoformat(),
        )

    if reference_prepared is None or reference_catalog is None:  # pragma: no cover
        raise RuntimeError("Detection run did not produce a reference catalog")

    sequence = DetectionSequence.from_epochs(epochs)
    sequence.validate_locations(tracking_config.location_tolerance_m)
    _log_sequence_timing(sequence)
    return DetectionRunArtifacts(
        files=files,
        algorithm=algorithm,
        sequence=sequence,
        reference_path=files[0],
        reference_catalog=reference_catalog,
        reference_prepared=reference_prepared,
    )


def _log_sequence_timing(sequence: DetectionSequence) -> None:
    """Log cadence diagnostics for a detection sequence."""
    sequence_timing = sequence_timing_diagnostics(sequence)
    logger.info(
        "Sequence timing | %d images | duration %.1f min | intervals "
        "min %.1f s | median %.1f s | p90 %.1f s | max %.1f s",
        sequence_timing.epoch_count,
        sequence_timing.duration_s / 60.0,
        sequence_timing.min_interval_s,
        sequence_timing.median_interval_s,
        sequence_timing.p90_interval_s,
        sequence_timing.max_interval_s,
    )


def _track_sequence(
    sequence: DetectionSequence,
    tracking_config: TrackingConfig,
) -> ImagePlaneTrackingResult:
    """Build and log bootstrap image-plane tracklets for ``sequence``."""
    tracking_result = ImagePlaneTracker(tracking_config).track(sequence)
    counts = tracking_result.diagnostic_counts()
    logger.info(
        "Built %d tracklets from %d images and %d detections: "
        "%d candidate, %d low-motion, %d poor-fit.",
        len(tracking_result.tracks),
        len(sequence.epochs),
        sequence.total_detections,
        counts["candidate"],
        counts["low-motion"],
        counts["poor-fit"],
    )
    population = track_population_diagnostics(tracking_result)
    logger.info(
        "Track population | length min %d | median %.1f | p90 %.1f | max %d | "
        "duration median %.1f min | p90 %.1f min | max %.1f min | "
        "median epoch coverage %.1f%%",
        population.min_length,
        population.median_length,
        population.p90_length,
        population.max_length,
        population.median_duration_s / 60.0,
        population.p90_duration_s / 60.0,
        population.max_duration_s / 60.0,
        100.0 * population.median_coverage_fraction,
    )
    return tracking_result


def execute_tracking_run(
    paths: Iterable[Path],
    algorithm: str,
    detection_config: DetectionConfig,
    tracking_config: TrackingConfig,
    *,
    frame_loader: FrameLoader = load_gonet_image,
    detector_factory: DetectorFactory = create_detector,
    clock: Clock = perf_counter,
) -> TrackingRunArtifacts:
    """Detect a sequence sequentially and build bootstrap image-plane tracks."""
    detected = execute_detection_run(
        paths,
        algorithm,
        detection_config,
        tracking_config,
        frame_loader=frame_loader,
        detector_factory=detector_factory,
        clock=clock,
    )
    tracking_result = _track_sequence(detected.sequence, tracking_config)
    return TrackingRunArtifacts(
        files=detected.files,
        algorithm=detected.algorithm,
        sequence=detected.sequence,
        tracking_result=tracking_result,
        reference_path=detected.reference_path,
        reference_catalog=detected.reference_catalog,
        reference_prepared=detected.reference_prepared,
    )


def _resolve_report_reference_path(
    sequence: DetectionSequence,
    default_path: Path,
    requested_path: Path | None,
) -> Path:
    """Return the retained sequence image to use as the static report background.

    The report reference is deliberately independent of product provenance. Changing
    it must never invalidate detections, tracking, or downstream astrometric products.
    """
    reference = (requested_path or default_path).expanduser().resolve()
    retained = {epoch.source_path.expanduser().resolve() for epoch in sequence.epochs}
    if reference not in retained:
        raise ValueError(
            "Report reference image is not part of the retained detection sequence: "
            f"{reference}"
        )
    return reference


def _prepare_report_reference(
    sequence: DetectionSequence,
    reference_path: Path,
    detection_config: DetectionConfig,
    *,
    frame_loader: FrameLoader = load_gonet_image,
) -> tuple[PreparedDetectionImage, DetectionCatalog]:
    """Prepare one selected report image and retrieve its cached detections."""
    frame = frame_loader(reference_path)
    prepared = prepare_bayer_detection_image(frame, detection_config)
    catalog = sequence.catalog_for(str(reference_path))
    if catalog is None:
        raise ValueError(
            f"Detection product is missing the catalog for report image {reference_path}"
        )
    return prepared, catalog


def _prepare_cached_reference(
    product: DetectionProduct,
    detection_config: DetectionConfig,
    *,
    frame_loader: FrameLoader = load_gonet_image,
) -> tuple[PreparedDetectionImage, DetectionCatalog]:
    """Prepare the detection product's original report reference image."""
    return _prepare_report_reference(
        product.sequence,
        product.reference_path,
        detection_config,
        frame_loader=frame_loader,
    )


def run_cli_workflow(
    inputs: Iterable[Path],
    *,
    recursive: bool,
    algorithm: str,
    detection_config: DetectionConfig,
    tracking_config: TrackingConfig,
    channel: GONetChannel,
    output_path: Path,
    reference_image_path: Path | None = None,
    output_dir: Path | None = None,
    overwrite_products: bool = False,
    grid_calibration_path: Path | None = None,
    sidereal_config: SiderealFitConfig | None = None,
    overwrite_solution: bool = False,
    solve_orientation: bool = False,
    orientation_config: OrientationFitConfig | None = None,
    catalog_cache_path: Path | None = None,
    overwrite_orientation: bool = False,
    metadata_loader: MetadataLoader = load_gonet_metadata,
) -> RunSummary:
    """Discover inputs, reuse compatible products, and write the PDF report.

    Detection and tracking products are stored in one output directory. A
    product is reused only when its semantic provenance matches the current
    source-file fingerprints and configuration. ``overwrite_products`` forces
    both expensive stages to run again.
    """
    discovery = discover_gonet_files(inputs, recursive=recursive)
    logger.info(discovery.summary)
    if discovery.missing:
        logger.warning(
            "Missing inputs: %s",
            ", ".join(str(path) for path in discovery.missing),
        )
    if discovery.unsupported:
        logger.warning(
            "Unsupported explicit files: %s",
            ", ".join(str(path) for path in discovery.unsupported),
        )
    if len(discovery.files) < 2:
        raise ValueError(
            "Tracking requires at least two candidate GONet images after discovery"
        )

    product_dir = (output_dir or output_path.parent).expanduser().resolve()
    store = ProductStore(product_dir)
    store.ensure()
    expected_detection_id = detection_product_id(
        discovery.files,
        algorithm,
        detection_config,
        location_tolerance_m=tracking_config.location_tolerance_m,
    )

    detection_product: DetectionProduct | None = None
    reused_detection = False
    if store.detections_path.exists() and not overwrite_products:
        try:
            detection_product = load_detection_product(
                store.detections_path,
                expected_product_id=expected_detection_id,
            )
            reused_detection = True
            logger.info(
                "Reusing detection product %s | %d images | %d detections | "
                "%d skipped before detection",
                store.detections_path,
                len(detection_product.sequence.epochs),
                detection_product.sequence.total_detections,
                detection_product.skipped_file_count,
            )
        except ProductError as exc:
            logger.warning(
                "Ignoring incompatible detection product %s: %s",
                store.detections_path,
                exc,
            )

    if detection_product is None:
        preflight = preflight_tracking_locations(
            discovery.files,
            tracking_config.location_tolerance_m,
            metadata_loader=metadata_loader,
        )
        _log_location_preflight(preflight)
        if len(preflight.files) < 2:
            raise ValueError(
                "Tracking requires at least two images after GPS/location preflight"
            )
        detected = execute_detection_run(
            preflight.files,
            algorithm,
            detection_config,
            tracking_config,
        )
        detection_product = DetectionProduct(
            product_id=expected_detection_id,
            algorithm=algorithm,
            detection_config=detection_config,
            location_tolerance_m=tracking_config.location_tolerance_m,
            sequence=detected.sequence,
            reference_path=detected.reference_path,
            discovered_files=tuple(discovery.files),
            zero_gps_files=tuple(item.path for item in preflight.selection.zero_gps),
            location_outlier_files=tuple(
                item.path for item in preflight.selection.outliers
            ),
            metadata_errors=preflight.metadata_errors,
        )
        written_detection = save_detection_product(
            store.detections_path, detection_product
        )
        logger.info(
            "Wrote detection product %s | %d images | %d detections",
            written_detection,
            len(detection_product.sequence.epochs),
            detection_product.sequence.total_detections,
        )
    else:
        detection_product.sequence.validate_locations(
            tracking_config.location_tolerance_m
        )
        _log_sequence_timing(detection_product.sequence)

    report_reference_path = _resolve_report_reference_path(
        detection_product.sequence,
        detection_product.reference_path,
        reference_image_path,
    )
    if (
        not reused_detection
        and report_reference_path == detection_product.reference_path
    ):
        reference_prepared = detected.reference_prepared
        reference_catalog = detected.reference_catalog
    elif report_reference_path == detection_product.reference_path:
        reference_prepared, reference_catalog = _prepare_cached_reference(
            detection_product, detection_config
        )
    else:
        reference_prepared, reference_catalog = _prepare_report_reference(
            detection_product.sequence,
            report_reference_path,
            detection_config,
        )
    if reference_image_path is not None:
        logger.info("Using report reference image %s", report_reference_path)

    expected_tracking_id = tracking_product_id(
        detection_product.product_id, tracking_config
    )
    tracking_product: TrackingProduct | None = None
    reused_tracking = False
    if (
        reused_detection
        and store.tracks_path.exists()
        and not overwrite_products
    ):
        try:
            tracking_product = load_tracking_product(
                store.tracks_path,
                detection_product.sequence,
                expected_product_id=expected_tracking_id,
                expected_detection_product_id=detection_product.product_id,
            )
            reused_tracking = True
            logger.info(
                "Reusing tracking product %s | %d tracks",
                store.tracks_path,
                len(tracking_product.result.tracks),
            )
        except ProductError as exc:
            logger.warning(
                "Ignoring incompatible tracking product %s: %s",
                store.tracks_path,
                exc,
            )

    if tracking_product is None:
        tracking_result = _track_sequence(
            detection_product.sequence, tracking_config
        )
        tracking_product = TrackingProduct(
            product_id=expected_tracking_id,
            detection_product_id=detection_product.product_id,
            tracking_config=tracking_config,
            result=tracking_result,
        )
        written_tracking = save_tracking_product(store.tracks_path, tracking_product)
        logger.info(
            "Wrote tracking product %s | %d tracks",
            written_tracking,
            len(tracking_result.tracks),
        )

    grid_calibration: GridCalibration | None = None
    sidereal_solution: SiderealRotationSolution | None = None
    sidereal_fit_config: SiderealFitConfig | None = None
    orientation_solution: AbsoluteOrientationSolution | None = None
    orientation_fit_config: OrientationFitConfig | None = None
    sidereal_product_path: Path | None = None
    orientation_product_path: Path | None = None
    current_sidereal_product_id: str | None = None
    reused_sidereal = False
    if grid_calibration_path is not None:
        sidereal_fit_config = sidereal_config or SiderealFitConfig()
        grid_path = Path(grid_calibration_path).expanduser().resolve()
        grid_calibration = load_grid_calibration(grid_path)
        expected_sidereal_id = sidereal_product_id(
            tracking_product.product_id,
            grid_path,
            sidereal_fit_config,
        )
        current_sidereal_product_id = expected_sidereal_id
        sidereal_product: SiderealProduct | None = None
        if (
            reused_tracking
            and store.sidereal_path.exists()
            and not overwrite_products
            and not overwrite_solution
        ):
            try:
                sidereal_product = load_sidereal_product(
                    store.sidereal_path,
                    expected_product_id=expected_sidereal_id,
                    expected_tracking_product_id=tracking_product.product_id,
                )
                reused_sidereal = True
                logger.info(
                    "Reusing sidereal product %s | axis r=%.4f deg theta=%.4f deg | "
                    "fit RMS %.4f deg",
                    store.sidereal_path,
                    sidereal_product.solution.axis_r_deg,
                    sidereal_product.solution.axis_theta_deg,
                    sidereal_product.solution.fit_rms_deg,
                )
            except ProductError as exc:
                logger.warning(
                    "Ignoring incompatible sidereal product %s: %s",
                    store.sidereal_path,
                    exc,
                )
        if sidereal_product is None:
            sidereal_solution = SiderealAxisFitter(sidereal_fit_config).fit(
                tracking_product.result,
                grid_calibration,
            )
            sidereal_product = SiderealProduct(
                product_id=expected_sidereal_id,
                tracking_product_id=tracking_product.product_id,
                grid_calibration_path=grid_path,
                fit_config=sidereal_fit_config,
                solution=sidereal_solution,
            )
            written_sidereal = save_sidereal_product(
                store.sidereal_path, sidereal_product
            )
            counts = sidereal_solution.diagnostic_counts()
            logger.info(
                "Wrote sidereal product %s | axis r=%.4f deg theta=%.4f deg | "
                "fit RMS %.4f deg | %d consistent | %d rejected | %d insufficient",
                written_sidereal,
                sidereal_solution.axis_r_deg,
                sidereal_solution.axis_theta_deg,
                sidereal_solution.fit_rms_deg,
                counts["sidereal-consistent"],
                counts["sidereal-rejected"],
                counts["insufficient"],
            )
        else:
            sidereal_solution = sidereal_product.solution
        sidereal_product_path = store.sidereal_path.resolve()

    reused_orientation = False
    if solve_orientation:
        if (
            grid_calibration is None
            or sidereal_solution is None
            or current_sidereal_product_id is None
        ):
            raise ValueError(
                "Absolute orientation requires --grid-calibration and a "
                "sidereal solution"
            )
        orientation_fit_config = orientation_config or OrientationFitConfig()
        cache_path = (
            catalog_cache_path or store.bright_star_catalog_path
        ).expanduser().resolve()
        catalog_was_cached = cache_path.exists()
        bright_catalog = load_or_fetch_bright_star_catalog(cache_path)
        if catalog_was_cached:
            logger.info("Reusing bright-star catalog cache %s", cache_path)
        else:
            logger.info(
                "Fetched and cached Bright Star Catalogue %s | %d stars",
                cache_path,
                len(bright_catalog.stars),
            )
        expected_orientation_id = orientation_product_id(
            current_sidereal_product_id,
            cache_path,
            orientation_fit_config,
        )
        orientation_product: OrientationProduct | None = None
        if (
            reused_sidereal
            and store.orientation_path.exists()
            and not overwrite_products
            and not overwrite_solution
            and not overwrite_orientation
        ):
            try:
                orientation_product = load_orientation_product(
                    store.orientation_path,
                    expected_product_id=expected_orientation_id,
                    expected_sidereal_product_id=current_sidereal_product_id,
                )
                reused_orientation = True
                logger.info(
                    "Reusing orientation product %s | optical axis az=%.3f deg "
                    "alt=%.3f deg | %d catalog matches",
                    store.orientation_path,
                    orientation_product.solution.optical_axis_azimuth_deg,
                    orientation_product.solution.optical_axis_altitude_deg,
                    len(orientation_product.solution.matches),
                )
            except ProductError as exc:
                logger.warning(
                    "Ignoring incompatible orientation product %s: %s",
                    store.orientation_path,
                    exc,
                )
        if orientation_product is None:
            bright_stars = bright_catalog.query_bright_stars(
                detection_product.sequence.epochs[0].exposure_midpoint,
                orientation_fit_config.limiting_magnitude,
            )
            orientation_solution = AbsoluteOrientationSolver(
                orientation_fit_config
            ).fit(
                tracking_product.result,
                sidereal_solution,
                grid_calibration,
                bright_stars,
            )
            orientation_product = OrientationProduct(
                product_id=expected_orientation_id,
                sidereal_product_id=current_sidereal_product_id,
                catalog_path=cache_path,
                fit_config=orientation_fit_config,
                solution=orientation_solution,
            )
            written_orientation = save_orientation_product(
                store.orientation_path, orientation_product
            )
            logger.info(
                "Wrote orientation product %s | optical axis az=%.3f deg "
                "alt=%.3f deg | twist %.3f deg | %d matches | median residual "
                "%.3f arcmin",
                written_orientation,
                orientation_solution.optical_axis_azimuth_deg,
                orientation_solution.optical_axis_altitude_deg,
                orientation_solution.twist_deg,
                len(orientation_solution.matches),
                60.0 * orientation_solution.fit_median_deg,
            )
        else:
            orientation_solution = orientation_product.solution
        orientation_product_path = store.orientation_path.resolve()

    artifacts = TrackingRunArtifacts(
        files=tuple(epoch.source_path for epoch in detection_product.sequence.epochs),
        algorithm=algorithm,
        sequence=detection_product.sequence,
        tracking_result=tracking_product.result,
        reference_path=report_reference_path,
        reference_catalog=reference_catalog,
        reference_prepared=reference_prepared,
        grid_calibration=grid_calibration,
        sidereal_solution=sidereal_solution,
        sidereal_fit_config=sidereal_fit_config,
        orientation_solution=orientation_solution,
        orientation_fit_config=orientation_fit_config,
    )
    raw = load_gonet_file_raw(artifacts.reference_path, parse_metadata=False)
    display_data = get_raw_channel(raw, channel)
    written = write_tracking_report_pdf(
        output_path,
        image_data=display_data,
        channel=channel,
        artifacts=artifacts,
        detection_config=detection_config,
        tracking_config=tracking_config,
    )
    result = artifacts.tracking_result
    return RunSummary(
        output_path=written,
        file_count=len(artifacts.files),
        detection_count=artifacts.sequence.total_detections,
        track_count=len(result.tracks),
        assigned_detection_count=result.assigned_detection_count,
        unassigned_detection_count=result.unassigned_detection_count,
        skipped_file_count=detection_product.skipped_file_count,
        zero_gps_count=len(detection_product.zero_gps_files),
        location_outlier_count=len(detection_product.location_outlier_files),
        metadata_error_count=len(detection_product.metadata_errors),
        detection_product_path=store.detections_path.resolve(),
        tracking_product_path=store.tracks_path.resolve(),
        sidereal_product_path=sidereal_product_path,
        orientation_product_path=orientation_product_path,
        reused_detection_product=reused_detection,
        reused_tracking_product=reused_tracking,
        reused_sidereal_product=reused_sidereal,
        reused_orientation_product=reused_orientation,
    )
