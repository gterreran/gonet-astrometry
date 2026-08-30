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

The final physical model treats stellar motion as a rotation on the unit sphere.
A conservative image-plane linker may be used to bootstrap source associations,
but those tracklets are converted to camera-frame rays before the celestial
rotation model is fitted. The astrometric solution therefore does not assume
that stellar tracks are Euclidean circles in a distorted fisheye image.

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

The source-detection milestone provides four interchangeable backends: SEP
extraction (the portal default), ``DAOStarFinder``, Photutils segmentation, and
a built-in SciPy local-maximum baseline. Every backend receives the same
full-resolution, Bayer-aware significance image and returns the same
``DetectionCatalog`` model. A provisional fisheye-footprint mask excludes the
dark sensor exterior, tiled sigma-clipped statistics model local background and
noise, and large bright contaminants are masked separately. The portal can
overlay these masks and full-sensor detections on any compact display channel.
Each run reports frame-load, backend-setup, shared-preprocessing,
backend-specific, and total wall-clock durations for direct comparisons. A
common diagnostics pass enriches every candidate with available peak, area,
shape, orientation, backend-quality, and mask-adjacency measurements. The portal
uses those values for detailed hover text and non-destructive compact, elongated,
extended, mask-adjacent, backend-flagged, and unclassified marker groups.

The first temporal-tracking stage can process any selected subset of discovered
images without keeping the full image sequence in memory. Frames are loaded and
detected sequentially, while only source catalogs and observing metadata are
retained. Bootstrap tracklets use the actual exposure midpoints for displacement gates,
track lifetime, and constant-velocity image-plane prediction. The primary gap
limit is expressed in elapsed minutes rather than image count, so short bursts,
multi-minute cadence gaps, and occasional missing exposures do not require a
uniform sampling assumption. Tracklets remain labelled as candidate,
low-motion, or poor-fit rather than being aggressively rejected. Complete track
histories can be inspected over the current image. This image-plane association
is a seed for the later spherical common-rotation-axis solution, not the final
astrometric model.

For faster detector and tracking experiments, ``gonet-astrometry run`` accepts
multiple files and/or folders, requires an explicit ``--algorithm``, exposes
every current detection and bootstrap-tracking parameter, and writes a two-page
PDF diagnostic. The first page contains the effective field of view,
reference-image detections, and calculated track histories; the second reports
cadence and track-fragmentation distributions. Before the expensive detection
pass starts, the runner inspects observing metadata, skips explicit ``0,0,0``
GPS failures, groups the remaining images by the configured location tolerance,
and retains the largest coherent observing-site group. Files with malformed
metadata or GPS locations outside that dominant group are reported and skipped.

The runner also persists portable ``detections.npz`` and ``tracks.npz``
mid-level products in one output directory. Compatible products are reused on
subsequent runs, so a 700-image sequence does not need to be extracted again
when only the later tracking or astrometric-calibration stage changes. Product
provenance includes source-file stat fingerprints and the relevant algorithm
and configuration values; incompatible products are recomputed automatically.
Use ``--overwrite-products`` to force a fresh detection and tracking pass.

When a portable Grid Calibration ``*_calibration.npz`` is supplied, the runner
also converts tracked full-sensor pixels into Grid-frame unit rays and fits one
robust common apparent-sky rotation axis at the fixed sidereal rate. The result
is cached independently as ``sidereal_rotation.npz`` and can be recomputed with
``--overwrite-solution`` without touching cached detections or tracks. The PDF
then gains a third page in legacy tracking mode showing spherical residuals and
accepted/rejected tracklets. For example:

```bash
gonet-astrometry run /path/to/night \
    --algorithm sep \
    --output-dir gonet_astrometry_output \
    --grid-calibration /path/to/camera_calibration.npz
```

Independent images in the Grid-assisted multichannel detection stage can be
processed in separate worker processes. Use ``--workers N`` to enable this;
``--workers 1`` remains the deterministic serial reference path. Worker count is
an execution setting only and does not change product provenance. For example,
``--workers 4`` processes up to four images concurrently while each image still
runs its four native Bayer channels sequentially.

In the Grid-assisted multichannel SEP path, the default source threshold is
``3.5`` sigma. The usable fisheye edge is inferred from each image by default.
``--field-edge-threshold-fraction`` controls the
image-contrast threshold for that footprint, while
``--field-edge-keep-margin-px`` can require accepted detections to lie farther
inside the inferred edge. The latter defaults to zero; the existing conservative
``--footprint-erosion-px`` remains part of the automatic footprint estimate.
Optional ``--grid-search-radius-deg`` and ``--grid-acceptance-radius-deg`` caps
can further restrict the search and retained-detection regions in Grid angular
coordinates. To reproduce the historical 75/70-degree behavior exactly, disable
the automatic footprint with ``--no-use-provisional-field-mask`` and pass
``--grid-search-radius-deg 75 --grid-acceptance-radius-deg 70``. An optional
portable ``--field-mask`` can additionally exclude fixed structures such as the
Adler dome. ``--field-mask-keep-margin-px`` optionally rejects retained centroids
within an additional full-sensor distance of that static-mask boundary while
still allowing the complete static mask to contribute to search/background
estimation.

The PDF background normally uses the detection product's original reference
image. Choose any other retained epoch without recomputing scientific products
with ``--reference-image /path/to/frame.jpg``. The selected file only changes
report rendering; detections, tracks, sidereal rotation, and absolute orientation
products remain reusable.

Catalog-assisted identification can also be run directly from detections with
``--identify-stars``. This path loads or fetches the Bright Star Catalogue once,
uses a V<=3.2 bright subset to bootstrap one common Grid-to-local-ENU attitude,
refines that rigid three-dimensional rotation with matches from the full sequence,
and then identifies visible stars independently in every frame down to V<=4.5 by
default. The portable ``stellar_identifications.npz`` product records expected
stars, matched detection identifiers, unmatched expected stars, residuals, and
the fitted attitude.

Grid-assisted runs now expose ``--tracking-mode legacy|catalog|hybrid``. The
default remains ``legacy`` while the catalog-first path is validated. ``catalog``
groups matched detections directly by catalog identifier, so no blind temporal
association or fragment merge is required for known stars. ``hybrid`` does the
same for catalog matches and then runs the existing spherical tracker only on
detections that were not claimed by catalog identification. The hybrid fallback
tracklets are cached separately in ``fallback_temporal_tracks.npz``. Catalog and
hybrid modes imply stellar identification automatically.

The fitted rotation pole still leaves one exact camera-attitude twist around the
celestial axis. Add ``--solve-orientation`` to resolve that final degree of
freedom by matching de-rotated sidereal-consistent tracks to a cached bright-star
catalog. The resulting portable ``absolute_orientation.npz`` stores the complete
Grid-to-local-ENU rotation matrix and the PDF gains a fourth orientation page.
The first run creates ``bright_star_catalog.npz`` in the output directory; later
runs reuse it. Orientation anchors are selected for stable declination across
their full tracks, and the remaining twist is found with a global 0--360 degree
pattern-registration search rather than independent pairwise star matches.

Run the validation commands documented in
`docs/source/developer_guide/contributing.rst` before opening a pull request.

## Contributing

Development branches from `dev` and is merged through focused pull requests.
New behavior requires tests, and public APIs require complete NumPy-style
docstrings. The developer guide contains the full checklist and documentation
standards.

## License

MIT. See `LICENSE`.
