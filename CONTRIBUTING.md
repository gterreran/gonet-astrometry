# Contributing

## Branches

- `main`: stable integration branch;
- `dev`: active integration branch;
- `feature/<name>`: focused feature work;
- `fix/<name>`: focused corrections.

## Required checks

Before opening a pull request, run:

```bash
pytest
ruff check .
black --check .
mypy src/gonet_astrometry
sphinx-build -W -b html docs/source docs/_build/html
```

New behavior requires tests. Public objects require NumPy-style docstrings,
including parameter, return, exception, and notes sections where applicable.
