.PHONY: test coverage lint format format-check typecheck docs check clean

test:
	pytest

coverage:
	pytest --cov --cov-report=term-missing --cov-report=html

lint:
	ruff check .

format:
	ruff check --fix .
	black .

format-check:
	black --check .

typecheck:
	mypy src/gonet_astrometry

docs:
	sphinx-build -W -b html docs/source docs/_build/html

check: test lint format-check typecheck docs

clean:
	rm -rf .coverage .mypy_cache .pytest_cache .ruff_cache htmlcov docs/_build
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
