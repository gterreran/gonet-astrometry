# GONet Astrometry Calibrator

`gonet-astrometry` is an independent, GONet Wizard-adjacent Python package for
astrometrically calibrating wide-field GONet images from stellar detections.
It is designed to combine an optional Grid calibration with metadata-aware,
multi-frame star tracking and a joint camera/astrometric solution.

> **Status:** pre-alpha scaffold. Public APIs and file formats will change.

## Initial goals

- load native GONet images and trustworthy exposure metadata;
- consume Grid calibration output through a narrow compatibility adapter;
- detect sources directly on the Bayer mosaic without requiring demosaicing;
- associate stellar detections across images from a common observing session;
- recover the celestial rotation axis in camera coordinates;
- match tracks to a star catalog and refine the geometric calibration;
- expose numerical functionality independently of any future GUI;
- maintain comprehensive tests and Sphinx API documentation.

## Design principle

The package treats stellar motion as a rotation on the unit sphere. Pixel
coordinates are converted to camera-frame rays before temporal fitting, so the
method does not assume that stellar tracks are Euclidean circles in a distorted
fisheye image.

## Development setup

```bash
conda create -n gonet_astrometry python=3.10
conda activate gonet_astrometry
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
pre-commit install
pytest
ruff check .
black --check .
mypy src/gonet_astrometry
sphinx-build -W -b html docs/source docs/_build/html
```

## Repository workflow

Development work should branch from `dev`. Changes are validated locally and
merged through focused pull requests. `main` remains the stable integration
branch.

## License

MIT. See `LICENSE`.

## Creating the local and GitHub repositories

After extracting the scaffold and reviewing the GitHub owner placeholders in
`pyproject.toml`, run one of:

```bash
./scripts/initialize_repository.sh private
./scripts/initialize_repository.sh public
```

The script creates the initial `main` commit, creates and pushes the GitHub
repository, creates `dev`, pushes it, and leaves the working tree on `dev`.
