"""Sphinx configuration for GONet Astrometry."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version

project = "GONet Astrometry Calibrator"
author = "GONet Astrometry contributors"
copyright = "2026, GONet Astrometry contributors"

try:
    release = package_version("gonet-astrometry")
except PackageNotFoundError:
    release = "0.0.0"

# ``version`` is a reserved Sphinx configuration value and must be a string.
# Keep it separate from ``importlib.metadata.version`` to avoid exposing the
# imported function as configuration data.
version = release.split("+", maxsplit=1)[0]

extensions = [
    "numpydoc",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
]

autodoc_member_order = "bysource"
autodoc_typehints = "description"
numpydoc_show_class_members = False
nitpicky = True

# ``sphinx-autodoc-typehints`` renders NumPy scalar parameters in ``NDArray``
# annotations as fully qualified class targets. NumPy's intersphinx inventory
# does not expose every scalar name, so ignore only the known valid annotations
# used by the public API while retaining strict checks everywhere else.
nitpick_ignore = [
    ("py:class", "numpy.bool_"),
    ("py:class", "numpy.float64"),
    ("py:class", "numpy.uint8"),
    # ``sphinx-autodoc-typehints`` emits class references for several valid
    # aliases used in public annotations even though they are not Python
    # classes with inventory targets.
    ("py:class", "NDArray"),
    ("py:class", "np.bool_"),
    ("py:class", "np.float64"),
    ("py:class", "np.uint8"),
    ("py:class", "Path"),
    ("py:class", "GONetChannel"),
    ("py:class", "SolarAltitudeProvider"),
    ("py:class", "CatalogRayProvider"),
    # Autodoc may render the private protocol as its local short name in the
    # public detector constructor signature.
    ("py:class", "_PreparedDetector"),
    # Private structural protocols deliberately remain implementation details.
    (
        "py:class",
        "gonet_astrometry.adapters.grid_calibration._GridEvaluator",
    ),
    (
        "py:class",
        "gonet_astrometry.detection.multichannel._PreparedDetector",
    ),
]

# Dash and Plotly do not publish complete Sphinx inventories for the public
# classes exposed in the portal API annotations. Keep strict reference checks
# for the rest of the project while allowing those third-party GUI types.
nitpick_ignore_regex = [
    ("py:class", r"dash\..*"),
    ("py:class", r"plotly\..*"),
]

intersphinx_mapping = {
    "astropy": ("https://docs.astropy.org/en/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "python": ("https://docs.python.org/3/", None),
}

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
html_theme = "furo"
html_title = "GONet Astrometry"
