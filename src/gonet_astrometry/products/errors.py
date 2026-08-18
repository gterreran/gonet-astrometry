"""Exceptions raised while reading reusable workflow products."""

from __future__ import annotations


class ProductError(ValueError):
    """Base class for invalid or incompatible workflow products."""


class ProductMismatchError(ProductError):
    """Raised when a valid product belongs to different inputs/settings."""


class ProductFormatError(ProductError):
    """Raised when a product cannot be interpreted safely."""
