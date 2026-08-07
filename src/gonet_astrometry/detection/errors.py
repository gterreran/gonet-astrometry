"""Exceptions raised by source-detection components."""


class SourceDetectionError(RuntimeError):
    """Base exception for source-detection failures."""


class DetectionBackendUnavailableError(SourceDetectionError):
    """Raised when an optional detector dependency is unavailable."""


class DetectionInputError(SourceDetectionError):
    """Raised when an image cannot be prepared for source detection."""
