Contributing
============

Development setup
-----------------

Activate a compatible Python 3.10 or newer environment, then install the
package and development dependencies in editable mode:

.. code-block:: bash

   python -m pip install -e ".[dev]"

A dedicated environment is optional. The existing ``gonet_wizard_dev`` Conda
environment may be reused while the projects remain dependency-compatible.

Branch workflow
---------------

Development branches from ``dev`` and is merged through focused pull requests.
Use ``feature/<name>`` for new functionality and ``fix/<name>`` for focused
corrections. The ``main`` branch remains the stable integration branch.

Required checks
---------------

Run the complete local validation suite before opening a pull request:

.. code-block:: bash

   pytest --cov --cov-report=term-missing
   ruff check .
   black --check .
   mypy src/gonet_astrometry
   rm -rf docs/_build
   sphinx-build -W --keep-going -b html docs/source docs/_build/html

New behavior requires tests. Tests based on multiple package components,
synthetic star fields, or real-data regressions should use the corresponding
pytest marker declared in ``pyproject.toml``.

Documentation standards
-----------------------

Public functions, classes, methods, and protocols require complete NumPy-style
docstrings suitable for automatic API reference generation. Document
parameters, return values, raised exceptions, units, array shapes, coordinate
conventions, and important assumptions where applicable.

Coordinate-system and metadata assumptions must also be stated near the code
that enforces them rather than appearing only in high-level documentation.
