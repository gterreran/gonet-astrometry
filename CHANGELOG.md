# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/), and the
project intends to follow semantic versioning once the public API stabilizes.

## [Unreleased]

### Added

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

- Corrected the Sphinx version configuration so it always contains text.
- Replaced recursive autosummary generation with explicit API pages to avoid
  duplicate toctrees and unresolved object references.

### Removed

- Pillow-based preview loading and the direct Pillow dependency.

- Generated package metadata, generated version files, generated API sources,
  repository-bootstrap helpers, and placeholder tests.
- Redundant environment, Makefile, pre-commit, and standalone contribution
  files; development instructions now live in the README and Sphinx guide.
