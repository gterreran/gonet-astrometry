Command-line detection and tracking
===================================

The ``run`` subcommand executes the same scientific detection and bootstrap
tracking components used by the portal without starting Dash or pywebview. It
is intended for rapid parameter comparisons over repeated image sequences.

Basic usage
-----------

Supply two or more image files, one or more folders, or a mixture of both. The
source-detection backend is mandatory::

   gonet-astrometry run /path/to/night \
       --algorithm sep \
       --output tracking.pdf

Folders are searched recursively by default. Use ``--no-recursive`` to inspect
only their top level. The report background defaults to the compact ``green1``
channel; ``--channel`` also accepts ``blue``, ``green2``, and ``red``.

Detection settings
------------------

Every field of :class:`~gonet_astrometry.detection.config.DetectionConfig` is
available as a command-line option. For example::

   gonet-astrometry run frame1.jpg frame2.jpg frame3.jpg \
       --algorithm sep \
       --threshold-sigma 4.5 \
       --fwhm-px 3.0 \
       --min-pixels 5 \
       --background-box-size-px 256 \
       --bright-mask-sigma 40

``--max-sources none`` retains every backend detection and
``--bright-mask-sigma none`` disables dynamic bright-region masking. Boolean
settings use paired options such as ``--deblend`` / ``--no-deblend`` and
``--use-provisional-field-mask`` / ``--no-use-provisional-field-mask``.

Location preflight
------------------

Before source detection begins, the command performs a metadata-only preflight
over the discovered files. Images reporting the camera GPS sentinel
``0,0,0`` are skipped. The remaining locations are grouped using
``--location-tolerance-m`` and only the largest group around one representative
site is retained. Files outside that group, as well as files whose required
astrometric metadata cannot be interpreted, are logged and skipped before the
expensive preprocessing and detector passes.

This protects long runs from failing at the end because one misplaced image or
bad GPS fix belongs to a different observing site. At least two images must
remain after the preflight.

Tracking settings
-----------------

Every field of :class:`~gonet_astrometry.tracking.config.TrackingConfig` is
also exposed. A typical experimental command is::

   gonet-astrometry run /path/to/night \
       --algorithm sep \
       --max-speed-px-per-minute 20 \
       --prediction-tolerance-px 6 \
       --max-gap-minutes 15 \
       --min-track-length 3 \
       --output sep_tracking.pdf

``--max-motion-px-min`` is an alias for ``--max-speed-px-per-minute``.
Track continuity is governed primarily by actual elapsed time through
``--max-gap-minutes``; no evenly spaced cadence is assumed. ``--max-gap-frames``
is an optional legacy guard and defaults to ``none``. ``--max-missing`` remains
an alias for that optional frame-count guard.

When extrapolating farther in time than the interval used to estimate the
current image-plane velocity, the association tolerance grows with the square
root of that time ratio and is capped by ``--max-prediction-gap-scale``. This
allows short bursts, multi-minute gaps, and occasional missing exposures to be
handled by one timestamp-aware linker without hard-coding a burst cadence.

Reusable mid-level products
---------------------------

Batch runs store reusable scientific products in one output directory. The
default layout is::

   gonet_astrometry_output/
       detections.npz
       tracks.npz
       tracking.pdf

Use ``--output-dir`` to choose a different directory. ``--output`` may still
place the PDF elsewhere, but the reusable products always remain together in
``--output-dir``.

``detections.npz`` stores the chronologically validated detection sequence,
observing metadata, detector diagnostics, and the result of the GPS/location
preflight. ``tracks.npz`` stores the bootstrap track references and their
diagnostics. Both are compressed NumPy archives containing only plain arrays;
they load with ``np.load(..., allow_pickle=False)``.

Before expensive processing begins, the runner computes a provenance identity
from the discovered source paths, file sizes and modification times, detection
algorithm, detection configuration, and location tolerance. A compatible
``detections.npz`` therefore skips GPS metadata preflight, Bayer reconstruction,
background preprocessing, and source detection for the complete sequence. The
tracking product additionally depends on the detection-product identity and all
tracking settings. Changing only tracking parameters reuses detections and
rebuilds only the tracks.

Stale or incompatible products are reported and replaced rather than silently
reused. To deliberately rerun both stages with unchanged inputs and settings,
use::

   gonet-astrometry run /path/to/night \
       --algorithm sep \
       --output-dir gonet_astrometry_output \
       --overwrite-products

The source files themselves are not content-hashed because reading hundreds of
large raw files merely to validate the cache would erase much of the benefit.
The cache instead uses absolute path, file size, and nanosecond modification
time as a conservative and inexpensive source fingerprint.

PDF diagnostic
--------------

The generated PDF contains two pages. The first is the static image-plane
tracking view based on the first discovered input image: selected compact native
channel, usable-field boundary, dynamic bright-region mask, reference-image
detections, and complete bootstrap track histories.

The second page is deliberately temporal. It plots the actual consecutive
exposure intervals, detections per track, track temporal spans, and epoch
coverage. Summary statistics include minimum, median, 90th-percentile, and
maximum cadence intervals plus track-length and duration statistics. These
measurements make track fragmentation visible without assuming that the input
images are evenly spaced in time.
