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

# ``sphinx-autodoc-typehints`` renders ``NDArray[np.float64]`` using the
# fully qualified scalar name. NumPy's intersphinx inventory does not expose
# ``numpy.float64`` as a Python class target, so ignore only that known, valid
# annotation while retaining strict reference checking everywhere else.
nitpick_ignore = [
    ("py:class", "numpy.float64"),
    ("py:class", "numpy.uint8"),
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
