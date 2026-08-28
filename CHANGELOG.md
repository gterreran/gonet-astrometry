# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/), and the
project intends to follow semantic versioning once the public API stabilizes.

## [Unreleased]

### Fixed

- Replace the permissive pairwise orientation-twist bootstrap with a global
  one-dimensional star-pattern registration.  Orientation anchors are now
  filtered by per-track declination stability, catalog candidates use a much
  tighter declination gate, and the full 0--360 degree twist range is scored
  before one-to-one catalog refinement.
- Skip VizieR Bright Star Catalogue rows with missing or invalid identifiers or
  coordinates before constructing the vectorized Astropy ``SkyCoord`` object.

### Added

- CLI ``--reference-image`` selection for choosing any retained epoch as the PDF background without invalidating or recomputing scientific products.
- Catalog-assisted absolute camera orientation that resolves the final twist
  about the fitted celestial pole, caches the Bright Star Catalogue in a
  portable no-pickle artifact, writes ``absolute_orientation.npz``, exposes
  independent orientation cache controls, and adds a fourth PDF diagnostics
  page with matched anchors and the Grid-to-ENU attitude.
- Optional portable Grid Calibration integration with full-model pixel-to-ray
  inversion, robust shared sidereal-axis fitting at the fixed sidereal rate,
  per-track spherical consistency diagnostics, a reusable no-pickle
  ``sidereal_rotation.npz`` product, independent ``--overwrite-solution`` cache
  control, and a third PDF diagnostics page.
- Portable, provenance-checked ``detections.npz`` and ``tracks.npz`` mid-level
  products with automatic cache reuse, a single CLI output directory, and an
  explicit ``--overwrite-products`` escape hatch for forced recomputation.
- Non-interactive ``run`` CLI workflow with configurable detection and tracking parameters plus a static PDF diagnostic figure.
- Bootstrap multi-image source tracking with chronological sequence validation,
  sequential frame processing, cached per-frame detection catalogs, time-scaled
  association gates, constant-velocity prediction, missing-frame support,
  non-destructive track diagnostics, and portal track overlays.
- Common source diagnostics with backend-preserved peak, area, source-shape,
  orientation, quality-bitmask, sharpness, and roundness measurements; shared
  local-image fallback measurements; non-destructive diagnostic flags and portal
  marker classes; detailed hover values; and per-class activity summaries.
- Mask-aware local Bayer preprocessing with an automatically inferred fisheye
  footprint, tiled sigma-clipped background/noise maps, dynamic bright-region
  masking, and optional portal overlays.
- Shared prepared-image execution so portal detector comparisons reuse the same
  preprocessing result and preserve mask diagnostics server-side.
- Pluggable source-detection registry with SciPy local maxima, Photutils
  segmentation, DAOStarFinder, and SEP backends.
- Per-run source-detection timing with separate frame-load, backend-setup,
  detector, total, and candidate-throughput measurements.
- Shared full-resolution Bayer-parity normalization, detector configuration,
  portal execution controls, server-side catalog caching, and image overlays.
- Scientific GONet ``ImageFrame`` adapter with full BGGR mosaic reconstruction,
  strict exposure/GPS metadata normalization, and a concrete pipeline loader.
- Multi-image portal session with lightweight file/folder discovery and a
  single lazily loaded native image cache.
- Sidebar calibration workspace with a provisional star-detection algorithm
  selector.
- Portal activity terminal with buffered package logging, conditional
  auto-scroll, and a desktop Exit control.
- Native pywebview desktop launcher for the Dash calibration portal, with a
  server-only development mode.
- Initial Dash portal for loading and interactively displaying native GONet
  channels.
- GONet Wizard adapter for constructing ``GONetFileRaw`` objects and
  selecting native Bayer channels.
- Initial package, documentation, testing, and continuous-integration scaffold.
- Core frame, metadata, detection, track, Grid calibration, and solution models.
- Protocol boundaries for GONet Wizard adapters, detectors, and catalogs.

### Fixed

- Make the shared sidereal-axis fit resistant to fragmented and false bootstrap tracks by seeding from a per-track small-circle consensus, excluding low-motion/poor-fit tracks from the primary fit when enough candidate tracks exist, and refitting only spherical-consistent inliers.
- Preflight multi-image CLI runs by rejecting ``0,0,0`` GPS fixes and retaining the largest location-consistent image group before source detection begins.
- Prefer the camera-generated Unix timestamp in GONet filenames to
  timezone-naive EXIF datetime fields during scientific frame loading.
- Corrected the Sphinx version configuration so it always contains text.
- Replaced recursive autosummary generation with explicit API pages to avoid
  duplicate toctrees and unresolved object references.

### Removed

- Pillow-based preview loading and the direct Pillow dependency.

- Generated package metadata, generated version files, generated API sources,
  repository-bootstrap helpers, and placeholder tests.
- Redundant environment, Makefile, pre-commit, and standalone contribution
  files; development instructions now live in the README and Sphinx guide.

### Changed

- Bootstrap tracking now uses elapsed exposure time as the primary gap criterion
  instead of assuming a fixed number of missing frames.
- Long-gap prediction tolerances scale conservatively with the actual time
  extrapolation.
- CLI tracking reports now include a second temporal-diagnostics page with
  cadence, track-length, duration, and epoch-coverage distributions.
