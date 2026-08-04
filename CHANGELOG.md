# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/), and the
project intends to follow semantic versioning once the public API stabilizes.

## [Unreleased]

### Added

- Initial package, documentation, testing, and continuous-integration scaffold.
- Core frame, metadata, detection, track, Grid calibration, and solution models.
- Protocol boundaries for GONet Wizard adapters, detectors, and catalogs.

### Fixed

- Corrected the Sphinx version configuration so it always contains text.
- Replaced recursive autosummary generation with explicit API pages to avoid
  duplicate toctrees and unresolved object references.

### Removed

- Generated package metadata, generated version files, generated API sources,
  repository-bootstrap helpers, and placeholder tests.
- Redundant environment, Makefile, pre-commit, and standalone contribution
  files; development instructions now live in the README and Sphinx guide.
