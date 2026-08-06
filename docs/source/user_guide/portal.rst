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
* native Bayer-channel selection;
* a provisional star-detection algorithm selector;
* current-image information; and
* the desktop Exit action.

The large right-hand area is reserved for image inspection and the activity
terminal. The detector selector is intentionally provisional and does not yet
run a backend; its configuration controls will be implemented with the first
source-detection backend.

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
