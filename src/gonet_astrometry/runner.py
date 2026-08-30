"""Non-interactive detection and bootstrap-tracking workflow.

This module exposes the backend used by the ``gonet-astrometry run`` command.
It deliberately avoids importing Dash or pywebview so batch experiments can be
run from a terminal with only the numerical, detection, and reporting stack.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
from tqdm import tqdm

from gonet_astrometry.adapters.gonet_wizard import (
    GONetChannel,
    get_raw_channel,
    load_gonet_file_raw,
    load_gonet_image,
    load_gonet_metadata,
)
from gonet_astrometry.adapters.grid_calibration import load_grid_calibration
from gonet_astrometry.catalogs import load_or_fetch_bright_star_catalog
from gonet_astrometry.detection.base import SourceDetector
from gonet_astrometry.detection.config import DetectionConfig
from gonet_astrometry.detection.diagnostics import enrich_detection_catalog
from gonet_astrometry.detection.field_mask import load_field_mask
from gonet_astrometry.detection.multichannel import (
    IndependentChannelSEPDetector,
    MultiChannelSEPConfig,
)
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
    StellarIdentificationProduct,
    StellarTrackingProduct,
    TemporalTrackingProduct,
    TrackingProduct,
    detection_product_id,
    load_detection_product,
    load_orientation_product,
    load_sidereal_product,
    load_stellar_identification_product,
    load_stellar_tracking_product,
    load_temporal_tracking_product,
    load_tracking_product,
    multichannel_detection_product_id,
    orientation_product_id,
    save_detection_product,
    save_orientation_product,
    save_sidereal_product,
    save_stellar_identification_product,
    save_stellar_tracking_product,
    save_temporal_tracking_product,
    save_tracking_product,
    sidereal_product_id,
    spherical_tracking_product_id,
    stellar_identification_product_id,
    stellar_tracking_product_id,
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
from gonet_astrometry.solving.stellar_tracks import (
    SiderealTrackMerger,
    StellarTrackMergeConfig,
    StellarTrackMergeResult,
)
from gonet_astrometry.tracking.catalog_identification import (
    CatalogSequenceMatcher,
    StellarIdentificationConfig,
    StellarIdentificationResult,
)
from gonet_astrometry.tracking.config import TrackingConfig
from gonet_astrometry.tracking.image_plane import (
    ImagePlaneTracker,
    ImagePlaneTrackingResult,
)
from gonet_astrometry.tracking.location_filter import (
    LocatedInput,
    LocationGroupSelection,
    select_dominant_location_group,
)
from gonet_astrometry.tracking.sequence import (
    DetectionEpoch,
    DetectionSequence,
    location_separation_m,
)
from gonet_astrometry.tracking.spherical import (
    SphericalTracker,
    SphericalTrackingConfig,
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
class _MultichannelFileResult:
    """One independently processed image returned by a detection worker."""

    path: Path
    epoch: DetectionEpoch
    elapsed_s: float


_MULTICHANNEL_WORKER_DETECTOR: IndependentChannelSEPDetector | None = None


def _initialize_multichannel_worker(
    grid_calibration_path: Path,
    detection_config: DetectionConfig,
    multichannel_config: MultiChannelSEPConfig,
    field_mask_path: Path | None,
) -> None:
    """Load immutable calibration state once in each detection worker process."""
    global _MULTICHANNEL_WORKER_DETECTOR

    calibration = load_grid_calibration(Path(grid_calibration_path))
    field_mask = (
        load_field_mask(Path(field_mask_path)) if field_mask_path is not None else None
    )
    _MULTICHANNEL_WORKER_DETECTOR = IndependentChannelSEPDetector(
        calibration,
        detection_config,
        multichannel_config,
        field_mask=field_mask,
    )


def _detect_multichannel_file_worker(path: Path) -> _MultichannelFileResult:
    """Load and detect one image inside an initialized worker process."""
    detector = _MULTICHANNEL_WORKER_DETECTOR
    if detector is None:  # pragma: no cover - defensive process initialization guard
        raise RuntimeError("Multichannel detection worker was not initialized")

    source = Path(path).expanduser().resolve()
    frame = load_gonet_image(source)
    started = perf_counter()
    catalog = detector.detect(str(source), frame)
    elapsed = perf_counter() - started
    return _MultichannelFileResult(
        path=source,
        epoch=DetectionEpoch.from_frame(str(source), frame, catalog),
        elapsed_s=elapsed,
    )


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
    multichannel_detection_config: MultiChannelSEPConfig | None = None
    spherical_tracking_config: SphericalTrackingConfig | None = None
    stellar_merge_config: StellarTrackMergeConfig | None = None
    stellar_identification_result: StellarIdentificationResult | None = None
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
    temporal_tracking_product_path: Path | None = None
    stellar_tracking_product_path: Path | None = None
    stellar_identification_product_path: Path | None = None
    sidereal_product_path: Path | None = None
    orientation_product_path: Path | None = None
    reused_detection_product: bool = False
    reused_tracking_product: bool = False
    reused_temporal_tracking_product: bool = False
    reused_stellar_tracking_product: bool = False
    reused_stellar_identification_product: bool = False
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
    show_progress: bool = False,
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
    metadata_progress = tqdm(
        ordered,
        desc="Reading metadata",
        unit="image",
        dynamic_ncols=True,
        disable=None if show_progress else True,
    )
    for index, path in enumerate(metadata_progress, start=1):
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
        separation = location_separation_m(selection.reference.location, item.location)
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


def execute_multichannel_detection_run(
    paths: Iterable[Path],
    detection_config: DetectionConfig,
    multichannel_config: MultiChannelSEPConfig,
    grid_calibration: GridCalibration,
    *,
    location_tolerance_m: float,
    workers: int = 1,
    show_progress: bool = False,
    grid_calibration_path: Path | None = None,
    field_mask_path: Path | None = None,
    detector: IndependentChannelSEPDetector | None = None,
    frame_loader: FrameLoader = load_gonet_image,
    clock: Clock = perf_counter,
) -> DetectionRunArtifacts:
    """Detect one sequence with independent native-channel SEP extraction.

    Images are scientifically independent during source extraction. ``workers=1``
    preserves the serial reference path; larger values distribute whole images
    across worker processes. Channel extraction and fusion remain sequential
    within each image so process-level parallelism cannot oversubscribe both
    dimensions at once.
    """
    files = tuple(
        sorted(
            {Path(path).expanduser().resolve() for path in paths},
            key=lambda item: str(item).casefold(),
        )
    )
    if not files:
        raise ValueError("The run command requires at least one GONet image")
    if workers < 1:
        raise ValueError("Detection workers must be at least one")

    source_detector = detector or IndependentChannelSEPDetector(
        grid_calibration,
        detection_config,
        multichannel_config,
    )
    wall_started = clock()
    epochs: list[DetectionEpoch]
    reference_prepared: PreparedDetectionImage
    reference_catalog: DetectionCatalog

    if workers == 1 or len(files) == 1:
        epochs = []
        reference_prepared_value: PreparedDetectionImage | None = None
        reference_catalog_value: DetectionCatalog | None = None

        detection_progress = tqdm(
            files,
            desc="Detecting sources",
            unit="image",
            dynamic_ncols=True,
            disable=None if show_progress else True,
        )
        for index, path in enumerate(detection_progress, start=1):
            frame = frame_loader(path)
            started = clock()
            catalog = source_detector.detect(str(path), frame)
            finished = clock()

            if reference_prepared_value is None:
                reference_prepared_value = source_detector.prepare_report_image(frame)
                reference_catalog_value = catalog

            epoch = DetectionEpoch.from_frame(str(path), frame, catalog)
            epochs.append(epoch)
            detection_progress.set_postfix(candidates=len(catalog), refresh=False)
            logger.debug(
                "Multichannel detection %d/%d | %s | %d candidates | %.3f s | "
                "midpoint %s",
                index,
                len(files),
                path.name,
                len(catalog),
                finished - started,
                epoch.exposure_midpoint.isoformat(),
            )

        if (
            reference_prepared_value is None or reference_catalog_value is None
        ):  # pragma: no cover
            raise RuntimeError("Detection run did not produce a reference catalog")
        reference_prepared = reference_prepared_value
        reference_catalog = reference_catalog_value
        effective_workers = 1
    else:
        if not isinstance(source_detector, IndependentChannelSEPDetector):
            raise ValueError(
                "Parallel multichannel detection requires "
                "IndependentChannelSEPDetector"
            )

        worker_grid_path = (
            Path(grid_calibration_path).expanduser().resolve()
            if grid_calibration_path is not None
            else (
                Path(grid_calibration.source).expanduser().resolve()
                if grid_calibration.source is not None
                else None
            )
        )
        if worker_grid_path is None:
            raise ValueError(
                "Parallel multichannel detection requires a portable Grid "
                "calibration path"
            )
        worker_field_mask_path = (
            Path(field_mask_path).expanduser().resolve()
            if field_mask_path is not None
            else None
        )
        if (
            getattr(source_detector, "field_mask", None) is not None
            and worker_field_mask_path is None
        ):
            raise ValueError(
                "Parallel multichannel detection requires field_mask_path when "
                "the detector uses a static field mask"
            )
        effective_workers = min(workers, len(files))
        results_by_path: dict[Path, _MultichannelFileResult] = {}

        logger.info(
            "Starting parallel multichannel detection | %d images | %d workers",
            len(files),
            effective_workers,
        )
        with ProcessPoolExecutor(
            max_workers=effective_workers,
            initializer=_initialize_multichannel_worker,
            initargs=(
                worker_grid_path,
                detection_config,
                multichannel_config,
                worker_field_mask_path,
            ),
        ) as executor:
            futures: dict[Future[_MultichannelFileResult], Path] = {
                executor.submit(_detect_multichannel_file_worker, path): path
                for path in files
            }
            detection_progress = tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Detecting sources",
                unit="image",
                dynamic_ncols=True,
                disable=None if show_progress else True,
            )
            for completed, future in enumerate(detection_progress, start=1):
                path = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    for pending in futures:
                        pending.cancel()
                    raise RuntimeError(
                        f"Multichannel detection failed for {path}: {exc}"
                    ) from exc
                results_by_path[result.path] = result
                detection_progress.set_postfix(
                    candidates=len(result.epoch.catalog), refresh=False
                )
                logger.debug(
                    "Multichannel detection %d/%d | %s | %d candidates | %.3f s | "
                    "midpoint %s",
                    completed,
                    len(files),
                    result.path.name,
                    len(result.epoch.catalog),
                    result.elapsed_s,
                    result.epoch.exposure_midpoint.isoformat(),
                )

        missing = [path for path in files if path not in results_by_path]
        if missing:  # pragma: no cover - executor contract guard
            raise RuntimeError(
                "Parallel multichannel detection returned no result for: "
                + ", ".join(str(path) for path in missing)
            )

        # Reconstruct the sequence in deterministic input order regardless of
        # worker completion order. DetectionSequence performs the final
        # chronological ordering below.
        epochs = [results_by_path[path].epoch for path in files]
        reference_frame = frame_loader(files[0])
        reference_prepared = source_detector.prepare_report_image(reference_frame)
        reference_catalog = results_by_path[files[0]].epoch.catalog

    wall_elapsed = clock() - wall_started
    logger.info(
        "Multichannel detection summary | %d images | %d workers | %.3f s wall | "
        "%.3f images/s",
        len(files),
        effective_workers,
        wall_elapsed,
        len(files) / wall_elapsed if wall_elapsed > 0.0 else float("inf"),
    )

    sequence = DetectionSequence.from_epochs(epochs)
    sequence.validate_locations(location_tolerance_m)
    _log_sequence_timing(sequence)
    return DetectionRunArtifacts(
        files=files,
        algorithm=source_detector.name,
        sequence=sequence,
        reference_path=files[0],
        reference_catalog=reference_catalog,
        reference_prepared=reference_prepared,
    )


def _track_spherical_sequence(
    sequence: DetectionSequence,
    calibration: GridCalibration,
    config: SphericalTrackingConfig,
) -> ImagePlaneTrackingResult:
    """Build and log Grid-aware spherical temporal tracklets."""
    result = SphericalTracker(config).track(sequence, calibration)
    logger.info(
        "Built %d spherical temporal tracklets from %d images and %d detections",
        len(result.tracks),
        len(sequence.epochs),
        sequence.total_detections,
    )
    population = track_population_diagnostics(result)
    logger.info(
        "Spherical track population | length min %d | median %.1f | p90 %.1f | "
        "max %d | duration median %.1f min | p90 %.1f min | max %.1f min | "
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
    return result


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
            "Detection product is missing the catalog for report image "
            f"{reference_path}"
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


def _prepare_multichannel_report_reference(
    sequence: DetectionSequence,
    reference_path: Path,
    detector: IndependentChannelSEPDetector,
    *,
    frame_loader: FrameLoader = load_gonet_image,
) -> tuple[PreparedDetectionImage, DetectionCatalog]:
    """Prepare report masks matching the independent-channel science path."""
    frame = frame_loader(reference_path)
    prepared = detector.prepare_report_image(frame)
    catalog = sequence.catalog_for(str(reference_path))
    if catalog is None:
        raise ValueError(
            "Detection product is missing the catalog for report image "
            f"{reference_path}"
        )
    return prepared, catalog


def _run_grid_cli_workflow(
    files: Iterable[Path],
    *,
    discovered_files: tuple[Path, ...],
    algorithm: str,
    detection_config: DetectionConfig,
    tracking_config: TrackingConfig,
    channel: GONetChannel,
    output_path: Path,
    reference_image_path: Path | None,
    store: ProductStore,
    overwrite_products: bool,
    grid_calibration_path: Path,
    field_mask_path: Path | None,
    detection_workers: int,
    multichannel_config: MultiChannelSEPConfig | None,
    spherical_tracking_config: SphericalTrackingConfig | None,
    stellar_merge_config: StellarTrackMergeConfig | None,
    sidereal_config: SiderealFitConfig | None,
    overwrite_solution: bool,
    detection_only: bool,
    solve_orientation: bool,
    orientation_config: OrientationFitConfig | None,
    catalog_cache_path: Path | None,
    overwrite_orientation: bool,
    identify_stars: bool,
    stellar_identification_config: StellarIdentificationConfig | None,
    overwrite_identifications: bool,
    metadata_loader: MetadataLoader,
) -> RunSummary:
    """Run the Grid-assisted multichannel/spherical stellar pipeline."""
    if algorithm != "sep":
        raise ValueError(
            "Grid-assisted stellar tracking currently requires --algorithm sep"
        )
    if detection_only and solve_orientation:
        raise ValueError("--solve-orientation cannot be used with --detection-only")

    grid_path = Path(grid_calibration_path).expanduser().resolve()
    grid_calibration = load_grid_calibration(grid_path)
    resolved_field_mask_path = (
        Path(field_mask_path).expanduser().resolve()
        if field_mask_path is not None
        else None
    )
    field_mask = (
        load_field_mask(resolved_field_mask_path)
        if resolved_field_mask_path is not None
        else None
    )
    if field_mask is not None:
        field_mask.validate_against(
            grid_calibration.image_shape,
            grid_calibration.coordinate_convention,
        )
        logger.info(
            "Using field mask %s | %.2f%% of sensor pixels excluded%s",
            resolved_field_mask_path,
            100.0 * field_mask.excluded_fraction,
            f" | {field_mask.description}" if field_mask.description else "",
        )
    multi_config = multichannel_config or MultiChannelSEPConfig()
    spherical_config = spherical_tracking_config or SphericalTrackingConfig()
    merge_config = stellar_merge_config or StellarTrackMergeConfig()
    sidereal_fit_config = sidereal_config or SiderealFitConfig()
    report_detector = IndependentChannelSEPDetector(
        grid_calibration,
        detection_config,
        multi_config,
        field_mask=field_mask,
    )

    expected_detection_id = multichannel_detection_product_id(
        files,
        detection_config,
        multi_config,
        grid_path,
        tracking_config.location_tolerance_m,
        field_mask_path=resolved_field_mask_path,
    )
    detection_product: DetectionProduct | None = None
    reused_detection = False
    detected: DetectionRunArtifacts | None = None

    if store.detections_path.exists() and not overwrite_products:
        try:
            detection_product = load_detection_product(
                store.detections_path,
                expected_product_id=expected_detection_id,
            )
            reused_detection = True
            logger.info(
                "Reusing multichannel detection product %s | %d images | "
                "%d detections | %d skipped before detection",
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
            files,
            tracking_config.location_tolerance_m,
            metadata_loader=metadata_loader,
        )
        _log_location_preflight(preflight)
        if not preflight.files:
            raise ValueError("No usable images remain after GPS/location preflight")
        detected = execute_multichannel_detection_run(
            preflight.files,
            detection_config,
            multi_config,
            grid_calibration,
            location_tolerance_m=tracking_config.location_tolerance_m,
            workers=detection_workers,
            show_progress=True,
            grid_calibration_path=grid_path,
            field_mask_path=resolved_field_mask_path,
            detector=report_detector,
        )
        detection_product = DetectionProduct(
            product_id=expected_detection_id,
            algorithm=detected.algorithm,
            detection_config=detection_config,
            location_tolerance_m=tracking_config.location_tolerance_m,
            sequence=detected.sequence,
            reference_path=detected.reference_path,
            discovered_files=discovered_files,
            zero_gps_files=tuple(item.path for item in preflight.selection.zero_gps),
            location_outlier_files=tuple(
                item.path for item in preflight.selection.outliers
            ),
            metadata_errors=preflight.metadata_errors,
            field_mask=field_mask,
            field_mask_keep_margin_px=multi_config.field_mask_keep_margin_px,
        )
        written_detection = save_detection_product(
            store.detections_path,
            detection_product,
        )
        logger.info(
            "Wrote multichannel detection product %s | %d images | %d detections",
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
        detected is not None
        and report_reference_path == detection_product.reference_path
    ):
        reference_prepared = detected.reference_prepared
        reference_catalog = detected.reference_catalog
    else:
        reference_prepared, reference_catalog = _prepare_multichannel_report_reference(
            detection_product.sequence,
            report_reference_path,
            report_detector,
        )
    if reference_image_path is not None:
        logger.info("Using report reference image %s", report_reference_path)

    identification_result: StellarIdentificationResult | None = None
    stellar_identification_product_path: Path | None = None
    reused_identification = False
    if identify_stars:
        identification_config = (
            stellar_identification_config or StellarIdentificationConfig()
        )
        cache_path = (
            (catalog_cache_path or store.bright_star_catalog_path)
            .expanduser()
            .resolve()
        )
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
        expected_identification_id = stellar_identification_product_id(
            detection_product.product_id,
            grid_path,
            cache_path,
            identification_config,
        )
        identification_product: StellarIdentificationProduct | None = None
        if (
            reused_detection
            and store.stellar_identifications_path.exists()
            and not overwrite_products
            and not overwrite_identifications
        ):
            try:
                identification_product = load_stellar_identification_product(
                    store.stellar_identifications_path,
                    detection_product.sequence,
                    expected_product_id=expected_identification_id,
                    expected_detection_product_id=detection_product.product_id,
                )
                reused_identification = True
                logger.info(
                    "Reusing stellar identifications %s | %d/%d star-epochs "
                    "matched | optical axis az=%.3f deg alt=%.3f deg | median "
                    "residual %.3f arcmin",
                    store.stellar_identifications_path,
                    identification_product.result.matched_star_count,
                    identification_product.result.visible_star_count,
                    identification_product.result.optical_axis_azimuth_deg,
                    identification_product.result.optical_axis_altitude_deg,
                    identification_product.result.fit_median_residual_arcmin,
                )
            except ProductError as exc:
                logger.warning(
                    "Ignoring incompatible stellar-identification product %s: %s",
                    store.stellar_identifications_path,
                    exc,
                )
        if identification_product is None:
            catalog_stars = bright_catalog.query_bright_stars(
                detection_product.sequence.epochs[0].exposure_midpoint,
                identification_config.limiting_magnitude,
            )
            identification_result = CatalogSequenceMatcher(
                identification_config
            ).fit_and_match(
                detection_product.sequence,
                grid_calibration,
                catalog_stars,
                field_mask=detection_product.field_mask,
                field_mask_keep_margin_px=(detection_product.field_mask_keep_margin_px),
                show_progress=True,
            )
            identification_product = StellarIdentificationProduct(
                product_id=expected_identification_id,
                detection_product_id=detection_product.product_id,
                grid_calibration_path=grid_path,
                catalog_path=cache_path,
                config=identification_config,
                result=identification_result,
            )
            written_identifications = save_stellar_identification_product(
                store.stellar_identifications_path,
                identification_product,
            )
            catalog_tracks = identification_result.catalog_tracks(
                detection_product.sequence,
                min_track_length=identification_config.minimum_catalog_track_length,
            )
            unmatched_detection_count = (
                detection_product.sequence.total_detections
                - identification_result.matched_star_count
            )
            frame_match_counts = np.asarray(
                [epoch.matched_star_count for epoch in identification_result.epochs],
                dtype=np.float64,
            )
            logger.info(
                "Wrote stellar identifications %s | %d/%d star-epochs matched | "
                "%d unmatched detections | %d catalog tracks | optical axis "
                "az=%.3f deg alt=%.3f deg | median/P90 residual %.3f/%.3f "
                "arcmin | matches/frame median %.1f P10/P90 %.1f/%.1f",
                written_identifications,
                identification_result.matched_star_count,
                identification_result.visible_star_count,
                unmatched_detection_count,
                len(catalog_tracks.tracks),
                identification_result.optical_axis_azimuth_deg,
                identification_result.optical_axis_altitude_deg,
                identification_result.fit_median_residual_arcmin,
                identification_result.fit_p90_residual_arcmin,
                float(np.median(frame_match_counts)),
                float(np.percentile(frame_match_counts, 10)),
                float(np.percentile(frame_match_counts, 90)),
            )
        else:
            identification_result = identification_product.result
        stellar_identification_product_path = (
            store.stellar_identifications_path.resolve()
        )

    result = ImagePlaneTrackingResult(detection_product.sequence, ())
    temporal_product: TemporalTrackingProduct | None = None
    stellar_product: StellarTrackingProduct | None = None
    sidereal_solution: SiderealRotationSolution | None = None
    orientation_solution: AbsoluteOrientationSolution | None = None
    orientation_fit_config: OrientationFitConfig | None = None

    tracking_product_path: Path | None = None
    temporal_tracking_product_path: Path | None = None
    stellar_tracking_product_path: Path | None = None
    sidereal_product_path: Path | None = None
    orientation_product_path: Path | None = None

    reused_temporal = False
    reused_stellar = False
    reused_sidereal = False
    reused_orientation = False

    if detection_only:
        logger.info(
            "Detection-only run requested; skipping spherical tracking, stellar "
            "merging, sidereal fitting, and absolute orientation"
        )
    elif len(detection_product.sequence.epochs) < spherical_config.min_track_length:
        logger.info(
            "Skipping spherical tracking: %d images are available, but "
            "spherical min_track_length=%d",
            len(detection_product.sequence.epochs),
            spherical_config.min_track_length,
        )
    else:
        expected_temporal_id = spherical_tracking_product_id(
            detection_product.product_id,
            grid_path,
            spherical_config,
        )
        if (
            reused_detection
            and store.temporal_tracks_path.exists()
            and not overwrite_products
        ):
            try:
                temporal_product = load_temporal_tracking_product(
                    store.temporal_tracks_path,
                    detection_product.sequence,
                    expected_product_id=expected_temporal_id,
                    expected_detection_product_id=detection_product.product_id,
                )
                reused_temporal = True
                logger.info(
                    "Reusing spherical temporal tracks %s | %d tracklets",
                    store.temporal_tracks_path,
                    len(temporal_product.result.tracks),
                )
            except ProductError as exc:
                logger.warning(
                    "Ignoring incompatible temporal track product %s: %s",
                    store.temporal_tracks_path,
                    exc,
                )

        if temporal_product is None:
            temporal_result = _track_spherical_sequence(
                detection_product.sequence,
                grid_calibration,
                spherical_config,
            )
            temporal_product = TemporalTrackingProduct(
                product_id=expected_temporal_id,
                detection_product_id=detection_product.product_id,
                grid_calibration_path=grid_path,
                tracking_config=spherical_config,
                result=temporal_result,
            )
            written_temporal = save_temporal_tracking_product(
                store.temporal_tracks_path,
                temporal_product,
            )
            logger.info(
                "Wrote spherical temporal tracks %s | %d tracklets",
                written_temporal,
                len(temporal_result.tracks),
            )

        result = temporal_product.result
        tracking_product_path = store.temporal_tracks_path.resolve()
        temporal_tracking_product_path = store.temporal_tracks_path.resolve()

        sidereal_fitter = SiderealAxisFitter(sidereal_fit_config)
        fit_candidate_count = (
            sidereal_fitter.fit_candidate_count(
                temporal_product.result,
                grid_calibration,
            )
            if temporal_product.result.tracks
            else 0
        )
        if fit_candidate_count < 3:
            logger.info(
                "Skipping stellar merge and sidereal fit: only %d temporal tracks "
                "satisfy the sidereal fit requirements; at least 3 are required",
                fit_candidate_count,
            )
        else:
            expected_stellar_id = stellar_tracking_product_id(
                temporal_product.product_id,
                merge_config,
                sidereal_fit_config,
            )
            merge_result: StellarTrackMergeResult | None = None
            if (
                reused_temporal
                and store.stellar_tracks_path.exists()
                and not overwrite_products
                and not overwrite_solution
            ):
                try:
                    stellar_product = load_stellar_tracking_product(
                        store.stellar_tracks_path,
                        detection_product.sequence,
                        expected_product_id=expected_stellar_id,
                        expected_temporal_tracking_product_id=(
                            temporal_product.product_id
                        ),
                    )
                    reused_stellar = True
                    logger.info(
                        "Reusing merged stellar tracks %s | %d physical stars",
                        store.stellar_tracks_path,
                        len(stellar_product.result.tracks),
                    )
                except ProductError as exc:
                    logger.warning(
                        "Ignoring incompatible stellar track product %s: %s",
                        store.stellar_tracks_path,
                        exc,
                    )

            if stellar_product is None:
                merge_result = SiderealTrackMerger(
                    merge_config,
                    sidereal_fit_config,
                ).fit_and_merge(
                    temporal_product.result,
                    grid_calibration,
                )
                stellar_product = StellarTrackingProduct(
                    product_id=expected_stellar_id,
                    temporal_tracking_product_id=temporal_product.product_id,
                    grid_calibration_path=grid_path,
                    merge_config=merge_config,
                    sidereal_config=sidereal_fit_config,
                    reference_time=merge_result.reference_time,
                    source_fragment_ids=merge_result.source_fragment_ids,
                    result=merge_result.tracking,
                )
                written_stellar = save_stellar_tracking_product(
                    store.stellar_tracks_path,
                    stellar_product,
                )
                logger.info(
                    "Wrote merged stellar tracks %s | %d physical stars",
                    written_stellar,
                    len(stellar_product.result.tracks),
                )

            result = stellar_product.result
            tracking_product_path = store.stellar_tracks_path.resolve()
            stellar_tracking_product_path = store.stellar_tracks_path.resolve()

            expected_sidereal_id = sidereal_product_id(
                stellar_product.product_id,
                grid_path,
                sidereal_fit_config,
            )
            sidereal_product: SiderealProduct | None = None
            if (
                reused_stellar
                and store.sidereal_path.exists()
                and not overwrite_products
                and not overwrite_solution
            ):
                try:
                    sidereal_product = load_sidereal_product(
                        store.sidereal_path,
                        expected_product_id=expected_sidereal_id,
                        expected_tracking_product_id=stellar_product.product_id,
                    )
                    reused_sidereal = True
                    logger.info(
                        "Reusing sidereal product %s | axis r=%.4f deg "
                        "theta=%.4f deg | fit RMS %.4f deg",
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
                sidereal_solution = (
                    merge_result.final_solution
                    if merge_result is not None
                    else sidereal_fitter.fit(
                        stellar_product.result,
                        grid_calibration,
                    )
                )
                sidereal_product = SiderealProduct(
                    product_id=expected_sidereal_id,
                    tracking_product_id=stellar_product.product_id,
                    grid_calibration_path=grid_path,
                    fit_config=sidereal_fit_config,
                    solution=sidereal_solution,
                )
                written_sidereal = save_sidereal_product(
                    store.sidereal_path,
                    sidereal_product,
                )
                counts = sidereal_solution.diagnostic_counts()
                logger.info(
                    "Wrote sidereal product %s | axis r=%.4f deg theta=%.4f deg | "
                    "fit RMS %.4f deg | %d consistent | %d rejected | "
                    "%d insufficient",
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

            if solve_orientation:
                orientation_fit_config = orientation_config or OrientationFitConfig()
                cache_path = (
                    (catalog_cache_path or store.bright_star_catalog_path)
                    .expanduser()
                    .resolve()
                )
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
                    expected_sidereal_id,
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
                            expected_sidereal_product_id=expected_sidereal_id,
                        )
                        reused_orientation = True
                        logger.info(
                            "Reusing orientation product %s | optical axis "
                            "az=%.3f deg alt=%.3f deg | %d catalog matches",
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
                        stellar_product.result,
                        sidereal_solution,
                        grid_calibration,
                        bright_stars,
                    )
                    orientation_product = OrientationProduct(
                        product_id=expected_orientation_id,
                        sidereal_product_id=expected_sidereal_id,
                        catalog_path=cache_path,
                        fit_config=orientation_fit_config,
                        solution=orientation_solution,
                    )
                    written_orientation = save_orientation_product(
                        store.orientation_path,
                        orientation_product,
                    )
                    logger.info(
                        "Wrote orientation product %s | optical axis az=%.3f deg "
                        "alt=%.3f deg | twist %.3f deg | %d matches | median "
                        "residual %.3f arcmin",
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

    if solve_orientation and sidereal_solution is None:
        raise ValueError(
            "Absolute orientation requires a successful sidereal solution; "
            "provide a longer sequence or relax the sidereal fit requirements"
        )

    artifacts = TrackingRunArtifacts(
        files=tuple(epoch.source_path for epoch in detection_product.sequence.epochs),
        algorithm=detection_product.algorithm,
        sequence=detection_product.sequence,
        tracking_result=result,
        reference_path=report_reference_path,
        reference_catalog=reference_catalog,
        reference_prepared=reference_prepared,
        grid_calibration=grid_calibration,
        sidereal_solution=sidereal_solution,
        sidereal_fit_config=sidereal_fit_config,
        multichannel_detection_config=multi_config,
        spherical_tracking_config=spherical_config,
        stellar_merge_config=merge_config,
        stellar_identification_result=identification_result,
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
        tracking_product_path=tracking_product_path,
        temporal_tracking_product_path=temporal_tracking_product_path,
        stellar_tracking_product_path=stellar_tracking_product_path,
        stellar_identification_product_path=stellar_identification_product_path,
        sidereal_product_path=sidereal_product_path,
        orientation_product_path=orientation_product_path,
        reused_detection_product=reused_detection,
        reused_tracking_product=(
            reused_stellar
            if stellar_tracking_product_path is not None
            else reused_temporal
        ),
        reused_temporal_tracking_product=reused_temporal,
        reused_stellar_tracking_product=reused_stellar,
        reused_stellar_identification_product=reused_identification,
        reused_sidereal_product=reused_sidereal,
        reused_orientation_product=reused_orientation,
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
    field_mask_path: Path | None = None,
    detection_workers: int = 1,
    multichannel_config: MultiChannelSEPConfig | None = None,
    spherical_tracking_config: SphericalTrackingConfig | None = None,
    stellar_merge_config: StellarTrackMergeConfig | None = None,
    sidereal_config: SiderealFitConfig | None = None,
    overwrite_solution: bool = False,
    detection_only: bool = False,
    solve_orientation: bool = False,
    orientation_config: OrientationFitConfig | None = None,
    catalog_cache_path: Path | None = None,
    overwrite_orientation: bool = False,
    identify_stars: bool = False,
    stellar_identification_config: StellarIdentificationConfig | None = None,
    overwrite_identifications: bool = False,
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
    if not discovery.files:
        raise ValueError("The run command requires at least one candidate GONet image")
    if detection_workers < 1:
        raise ValueError("Detection workers must be at least one")
    if detection_only and grid_calibration_path is None:
        raise ValueError("--detection-only requires --grid-calibration")
    if grid_calibration_path is None and len(discovery.files) < 2:
        raise ValueError(
            "Legacy image-plane tracking requires at least two candidate GONet "
            "images after discovery"
        )

    product_dir = (output_dir or output_path.parent).expanduser().resolve()
    store = ProductStore(product_dir)
    store.ensure()

    if field_mask_path is not None and grid_calibration_path is None:
        raise ValueError("--field-mask requires --grid-calibration")
    if identify_stars and grid_calibration_path is None:
        raise ValueError("--identify-stars requires --grid-calibration")
    if detection_workers != 1 and grid_calibration_path is None:
        raise ValueError(
            "--workers greater than 1 currently requires --grid-calibration"
        )

    if grid_calibration_path is not None:
        return _run_grid_cli_workflow(
            discovery.files,
            discovered_files=tuple(discovery.files),
            algorithm=algorithm,
            detection_config=detection_config,
            tracking_config=tracking_config,
            channel=channel,
            output_path=output_path,
            reference_image_path=reference_image_path,
            store=store,
            overwrite_products=overwrite_products,
            grid_calibration_path=grid_calibration_path,
            field_mask_path=field_mask_path,
            detection_workers=detection_workers,
            multichannel_config=multichannel_config,
            spherical_tracking_config=spherical_tracking_config,
            stellar_merge_config=stellar_merge_config,
            sidereal_config=sidereal_config,
            overwrite_solution=overwrite_solution,
            detection_only=detection_only,
            solve_orientation=solve_orientation,
            orientation_config=orientation_config,
            catalog_cache_path=catalog_cache_path,
            overwrite_orientation=overwrite_orientation,
            identify_stars=identify_stars,
            stellar_identification_config=stellar_identification_config,
            overwrite_identifications=overwrite_identifications,
            metadata_loader=metadata_loader,
        )

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
            show_progress=True,
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
    if reused_detection and store.tracks_path.exists() and not overwrite_products:
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
        tracking_result = _track_sequence(detection_product.sequence, tracking_config)
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

    if solve_orientation:
        raise ValueError("Absolute orientation requires --grid-calibration")

    artifacts = TrackingRunArtifacts(
        files=tuple(epoch.source_path for epoch in detection_product.sequence.epochs),
        algorithm=algorithm,
        sequence=detection_product.sequence,
        tracking_result=tracking_product.result,
        reference_path=report_reference_path,
        reference_catalog=reference_catalog,
        reference_prepared=reference_prepared,
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
        reused_detection_product=reused_detection,
        reused_tracking_product=reused_tracking,
    )
