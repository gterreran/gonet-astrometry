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

Activate any compatible Python 3.10+ environment, including
`gonet_wizard_dev`, and install the package in editable mode:

```bash
python -m pip install -e ".[dev]"
```

Run the validation commands documented in
`docs/source/developer_guide/contributing.rst` before opening a pull request.

## Contributing

Development branches from `dev` and is merged through focused pull requests.
New behavior requires tests, and public APIs require complete NumPy-style
docstrings. The developer guide contains the full checklist and documentation
standards.

## License

MIT. See `LICENSE`.
