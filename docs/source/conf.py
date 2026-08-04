"""Sphinx configuration for GONet Astrometry."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

project = "GONet Astrometry Calibrator"
author = "GONet Astrometry contributors"
copyright = "2026, GONet Astrometry contributors"

try:
    release = version("gonet-astrometry")
except PackageNotFoundError:
    release = "0.0.0"

extensions = [
    "numpydoc",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
]

autosummary_generate = True
autodoc_typehints = "description"
autodoc_member_order = "bysource"
numpydoc_show_class_members = False
nitpicky = True

intersphinx_mapping = {
    "astropy": ("https://docs.astropy.org/en/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "python": ("https://docs.python.org/3/", None),
}

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
html_theme = "furo"
html_title = "GONet Astrometry"
