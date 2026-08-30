"""Tracking-mode helpers for catalog-first and hybrid association.

Catalog identification already provides a stable stellar label for matched
sources.  This module turns those labels into physical tracks directly and
constructs a legacy spherical-tracking view containing only detections that were
not claimed by the catalog matcher.  The latter is the fallback/discovery input
for hybrid tracking.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Literal

from gonet_astrometry.models.detection import DetectionCatalog
from gonet_astrometry.models.track import StarTrack
from gonet_astrometry.tracking.catalog_identification import (
    StellarIdentificationResult,
)
from gonet_astrometry.tracking.image_plane import ImagePlaneTrackingResult
from gonet_astrometry.tracking.sequence import DetectionEpoch, DetectionSequence

TrackingMode = Literal["legacy", "catalog", "hybrid"]
"""Available Grid-assisted source-association strategies."""


def unmatched_detection_sequence(
    sequence: DetectionSequence,
    identifications: StellarIdentificationResult,
) -> DetectionSequence:
    """Return ``sequence`` with catalog-matched detections removed.

    Epoch timing, location, image geometry, source paths, and detector metadata
    are preserved exactly.  Only the per-epoch detection tuples are filtered.
    This makes the returned sequence suitable for the existing spherical
    tracker while guaranteeing that the fallback path cannot reuse detections
    already assigned to catalog stars.
    """
    identification_by_frame = {
        epoch.frame_identifier: epoch for epoch in identifications.epochs
    }
    if set(identification_by_frame) != {
        epoch.frame_identifier for epoch in sequence.epochs
    }:
        raise ValueError("Identification result does not match detection sequence")

    filtered_epochs: list[DetectionEpoch] = []
    for epoch in sequence.epochs:
        identification_epoch = identification_by_frame[epoch.frame_identifier]
        matched = set(identification_epoch.matched_detection_identifiers)
        catalog = DetectionCatalog(
            frame_identifier=epoch.catalog.frame_identifier,
            detections=tuple(
                detection
                for detection in epoch.catalog.detections
                if detection.identifier not in matched
            ),
            detector_name=epoch.catalog.detector_name,
        )
        filtered_epochs.append(
            DetectionEpoch(
                frame_identifier=epoch.frame_identifier,
                source_path=epoch.source_path,
                exposure_midpoint=epoch.exposure_midpoint,
                location=epoch.location,
                image_shape=epoch.image_shape,
                sensor_orientation=epoch.sensor_orientation,
                catalog=catalog,
            )
        )
    return DetectionSequence(tuple(filtered_epochs))


def combine_catalog_and_fallback_tracks(
    sequence: DetectionSequence,
    catalog_result: ImagePlaneTrackingResult,
    fallback_result: ImagePlaneTrackingResult,
) -> ImagePlaneTrackingResult:
    """Combine catalog-labelled tracks and disjoint fallback tracklets.

    The returned track identifiers are re-numbered contiguously.  Catalog
    tracks are retained first so their identifiers remain stable when fallback
    content changes.  A shared detection reference between the two inputs is a
    programming error because hybrid fallback is defined only on unmatched
    detections.
    """
    catalog_points = {
        (point.frame_identifier, point.detection_identifier)
        for track in catalog_result.tracks
        for point in track.points
    }
    fallback_points = {
        (point.frame_identifier, point.detection_identifier)
        for track in fallback_result.tracks
        for point in track.points
    }
    overlap = catalog_points & fallback_points
    if overlap:
        raise ValueError(
            "Catalog and fallback tracking results contain shared detections"
        )

    combined: list[StarTrack] = []
    for track in (*catalog_result.tracks, *fallback_result.tracks):
        combined.append(replace(track, identifier=len(combined)))
    return ImagePlaneTrackingResult(sequence=sequence, tracks=tuple(combined))
