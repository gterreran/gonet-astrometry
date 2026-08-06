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
- provide a Dash portal for interactive image inspection and calibration;
- keep numerical functionality independent of the GUI;
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

Start the desktop calibration portal with an optional image or folder:

```bash
gonet-astrometry portal --input /path/to/image-or-folder
```

The command starts Dash locally and opens the interface in an independent
pywebview window. The portal includes a read-only activity terminal for runtime
feedback and an Exit control for closing the desktop window. Use
``gonet-astrometry portal --server-only`` when a normal browser-based server is
preferable during development.

The sidebar accepts multiple files and folders, discovers candidate original
GONet ``.jpg`` files without parsing all of them, and loads only the selected
file through ``GONetFileRaw.from_file``. The server caches at most one native
image object and displays one compact Bayer channel without a separate JPEG
preview or Pillow-based loading path. The ``gonet_wizard_dev`` environment must
therefore contain the current GONet Wizard package.

The scientific adapter separately loads the selected file with Wizard metadata
enabled, reconstructs the full-resolution native BGGR mosaic, and returns a
validated ``ImageFrame``. Explicit Unix metadata is used first, followed by
the camera-generated Unix token in the filename; timezone-naive EXIF datetime
fields are used only when an offset is available. Filesystem timestamps are
never used. Latitude and longitude must be present in the parsed image metadata.

Run the validation commands documented in
`docs/source/developer_guide/contributing.rst` before opening a pull request.

## Contributing

Development branches from `dev` and is merged through focused pull requests.
New behavior requires tests, and public APIs require complete NumPy-style
docstrings. The developer guide contains the full checklist and documentation
standards.

## License

MIT. See `LICENSE`.
