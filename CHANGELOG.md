# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/), and the
project intends to follow semantic versioning once the public API stabilizes.

## [Unreleased]

### Added

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
