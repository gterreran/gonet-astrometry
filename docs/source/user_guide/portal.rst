Calibration portal
==================

The calibration portal combines a local Plotly Dash server with a pywebview
shell. By default it opens as an independent desktop window rather than a
browser tab. The current portal is a multi-image workspace: it discovers native
GONet candidates, loads only the selected file, and displays one Bayer channel
at a time.

Starting the portal
-------------------

Activate an environment in which both GONet Wizard and ``gonet-astrometry`` are
installed, then run:

.. code-block:: bash

   python -m pip install -e ".[gui]"
   gonet-astrometry portal --input /path/to/image-or-folder

The ``--input`` option (also available as ``--image``) is optional and may
point to either an original GONet ``.jpg`` file or a directory to scan. The portal starts Dash on
``127.0.0.1:8050`` and opens a native window connected to that local endpoint.
``--host`` and ``--port`` can override the server binding.

For browser-based development, run the server without pywebview:

.. code-block:: bash

   gonet-astrometry portal --server-only

Multi-image input catalog
-------------------------

The sidebar accepts one file or directory per line. Directory discovery may be
recursive. Discovery performs only lightweight filesystem inspection and
registers files with the ``.jpg`` suffix; it does not parse every image.
Candidates are validated through GONet Wizard only when selected for display.

The server-side portal session caches at most one ``GONetFileRaw`` instance.
Changing Bayer channels therefore reuses the current native image, while
selecting another file replaces the cached object. Large NumPy arrays are never
placed in browser-side Dash stores.

Workspace layout
----------------

All interactive controls are grouped in the left sidebar:

* source files and folders;
* recursive candidate discovery;
* current-image selection and loading;
* native Bayer-channel selection and detection-mask overlays;
* a pluggable star-detection algorithm selector;
* multi-image bootstrap tracking controls and sequence selection;
* current-image information; and
* the desktop Exit action.

The large right-hand area is reserved for image inspection and the activity
terminal. Detection masks can be shown or hidden after a source-detection run.
The cyan boundary marks the provisional usable field, while magenta points mark
large dynamic bright-region exclusions.

Native channel display
----------------------

The portal creates a ``GONetFileRaw`` through the Wizard's native ``from_file``
loader. It does not decode a JPEG preview, demosaic the image, or construct an
RGB representation.

The channel selector exposes ``blue``, ``green1``, ``green2``, and ``red``.
``green1`` is selected initially. Each displayed array remains in the compact
``(H, W)`` Bayer-channel coordinates returned by GONet Wizard. The Plotly
figure applies percentile display limits, but the channel values are neither
modified nor resampled.

Portal channel display keeps metadata parsing disabled so switching images and
channels remains lightweight. The scientific loader is a separate path: it
loads the selected image with Wizard metadata enabled, reconstructs the complete
native Bayer mosaic, and returns a validated
:class:`~gonet_astrometry.models.frame.ImageFrame`.

Scientific frame loading
------------------------

The scientific adapter combines the four sparse Wizard Bayer planes into one
full-resolution BGGR sensor array. Each sensor location must be populated by
exactly one channel; overlapping or missing samples are rejected rather than
silently repaired.

Required astrometric metadata are normalized into
:class:`~gonet_astrometry.models.frame.ImageMetadata`:

* exposure start in UTC;
* strictly positive exposure duration;
* observing latitude and longitude; and
* optional elevation, defaulting to zero when it is absent.

The adapter accepts common Wizard and EXIF-style key names, decimal or
DMS-formatted GPS coordinates, and rational exposure values. After an explicit
Unix value in Wizard metadata, it prefers the ten-digit Unix timestamp embedded
in a standard GONet filename to timezone-naive EXIF datetime fields. File
creation and modification times are never used.

Activity terminal
-----------------

The lower activity panel behaves like a read-only terminal. Package log records
are copied into a bounded, thread-safe in-memory buffer and polled by Dash twice
per second. Discovery, loading, validation, and future calibration progress
therefore share one user-facing feedback path.

The panel follows new output automatically while the scrollbar remains near the
bottom. Scrolling upward pauses automatic movement so earlier messages can be
read; returning to the bottom resumes it.

The ``Exit`` button requests closure through the ``WebviewAPI`` bridge exposed
by the desktop launcher. In ``--server-only`` mode there is no pywebview window
to close, so the button leaves the server running and records a warning in the
activity terminal instead.

Source detection
----------------

The star-detection sidebar runs one of four registered backends on the selected
scientific frame:

* ``SciPy local maxima`` provides a deterministic baseline using Gaussian
  smoothing and non-maximum suppression;
* ``Photutils segmentation`` detects connected regions and can deblend
  overlapping sources;
* ``DAOStarFinder`` applies the Photutils DAOFIND point-source algorithm; and
* ``SEP extraction`` exposes Source Extractor-style detection and deblending.

All backends receive one common full-resolution detection image. The four native
Bayer parities are processed independently, which reduces fixed color-channel
sensitivity differences without demosaicing, interpolation, or coordinate
resampling.

Before background estimation, a provisional geometric mask separates the
illuminated fisheye footprint from the dark rectangular sensor exterior. The
largest connected illuminated region is filled and eroded inward so detector
kernels do not straddle the uncertain optical rim. If the image does not contain
enough center-to-corner contrast to infer a reliable footprint, the preprocessor
falls back to the input validity mask rather than inventing a boundary.

Within the usable field, each Bayer parity is divided into coarse boxes.
Sigma-clipped medians and robust noise estimates are interpolated into smooth
local background and noise maps. This makes the detection threshold local to
the sky brightness and vignetting instead of allowing the black exterior, Moon,
horizon glow, or broad gradients to determine one global statistic.

After normalization, only large connected regions above a high significance
threshold are dynamically masked and dilated. The minimum-area requirement is
intended to retain isolated bright stars while excluding extended contaminants
such as the Moon core and saturated artificial lights. The geometric and dynamic
masks remain separate diagnostics and are cached with the latest detection run.
SEP is the default portal backend, while every registered backend remains
available for comparison.

The shared controls are:

``Threshold``
    Detection threshold in approximate background-noise sigma units.

``Approximate FWHM``
    Expected source width in full native sensor pixels. Point-source and
    smoothing backends use this value directly.

``Minimum connected pixels``
    Minimum source area used by Photutils segmentation and SEP. Other backends
    retain the setting in the common configuration but do not use it directly.

The resulting :class:`~gonet_astrometry.models.detection.DetectionCatalog` is
kept server-side. Detection coordinates remain in the full sensor system. For
visual inspection, the portal divides them by two before drawing markers over
the selected compact Bayer-channel image. Switching display channels reuses the
same catalog; selecting a different file invalidates it.

Detection diagnostics
~~~~~~~~~~~~~~~~~~~~~

After backend extraction, every candidate passes through one common diagnostics
step on the same prepared significance image. Backend measurements are retained
when available, while missing peak, area, axis, orientation, elongation, and
ellipticity values are estimated from a local native-coordinate cutout. SEP also
retains its integer extraction bitmask, DAOStarFinder retains sharpness and both
roundness values, and Photutils segmentation retains segment area and source
axes.

Candidates near the provisional field boundary or a dynamic bright-region mask
are flagged without being removed. Large footprints, high axis ratios, and
backend quality flags are likewise descriptive only. Each detection receives one
mutually exclusive display class: compact, elongated, extended, mask-adjacent,
backend-flagged, or unclassified. The priority ordering is deliberately
conservative, so a backend-flagged source is displayed as such even when it is
also extended or elongated.

The image overlay uses a separate marker style for each represented class and
shows native coordinates, peak significance, flux, footprint area, elongation,
ellipticity, and flags in the hover text. The sidebar and activity terminal
report class counts after every run. These diagnostics do not yet reject or
filter candidates; they provide the measurements needed to tune later filtering
and temporal tracking.

Detector timing
~~~~~~~~~~~~~~~

Each successful run records separate wall-clock durations for scientific-frame
loading, detector construction, shared mask-aware preprocessing,
backend-specific execution, and the complete request. Backend time is the
primary algorithm-comparison value shown in the sidebar. It starts only after
the common significance image is ready, so all algorithms are compared against
identical prepared data. The activity terminal reports the full breakdown,
candidate count, and candidates per backend second.

The first invocation of an optional backend can include Python module import
cost in its setup or backend duration. Repeat runs therefore provide a useful
warm-cache comparison in addition to the cold-start measurement.


Multi-image bootstrap tracking
------------------------------

The star-tracking controls operate on an explicit subset of the discovered
images. Up to the first five discovered candidates are preselected so opening a
large directory cannot accidentally trigger a full-night detection run. The
selection can be expanded or replaced before tracking. Discovery still stores
only paths. When tracking is requested, the
portal loads one scientific frame at a time, preprocesses it, runs the selected
detector, retains only the lightweight detection catalog and observing metadata,
and then releases the large frame arrays before advancing to the next file. A
repeated tracking run with the same image set and detector settings reuses the
cached catalogs, so association parameters can be tuned without repeating image
loading and source detection.

Detection epochs are ordered by exposure midpoint rather than filename. The
sequence requires matching native sensor shape and orientation, strictly unique
timestamps, and observing positions within the configured GPS tolerance.

The first tracking implementation is deliberately a bootstrap image-plane
linker. A nascent track uses the elapsed time and ``Maximum motion`` control to
set a search radius. After two detections have been associated, the next
position is predicted from the measured image-plane velocity and matched within
the ``Prediction tolerance``. Tracks may bridge the configured number of
missing epochs, and only associations reaching ``Minimum detections per track``
are returned. One detection can participate in at most one track at a given
epoch.

Returned tracks are not yet identified as stars and are not rejected solely on
single-image source diagnostics. They receive non-destructive temporal classes:

``candidate``
    A sufficiently smooth track with measurable image-plane motion.

``low-motion``
    A smooth track with little end-to-end displacement. Fixed lights naturally
    fall in this class, but the class is retained because real stars close to
    the celestial pole may also move slowly.

``poor-fit``
    A track whose points have a large RMS residual around a local quadratic
    image-plane path.

The image viewer draws complete track histories over the currently displayed
compact Bayer channel and reports track identifier, frame, timestamp, mean
speed, and fit RMS on hover. These tracklets are seeds for the next, physically
stronger stage: converting detections to camera rays and fitting the common
celestial rotation axis. The image-plane linker is therefore intentionally
conservative and should not be interpreted as an astrometric solution.
