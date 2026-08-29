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

The static report background defaults to the original reference image recorded
in ``detections.npz``.  Any retained sequence image can instead be selected
without invalidating cached scientific products::

   gonet-astrometry run /path/to/night \
       --algorithm sep \
       --reference-image /path/to/night/frame-350.jpg

``--reference-image`` affects PDF rendering only.  Detection, tracking, sidereal,
and absolute-orientation products remain reusable.

Detection settings
------------------

Every field of :class:`~gonet_astrometry.detection.config.DetectionConfig` is
available as a command-line option. The default detection threshold is ``3.5``
sigma, selected from catalog-backed completeness tests on representative Adler
night-sky data. For example::

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

For the Grid-assisted independent-channel SEP path, the default search field is
the automatically inferred illuminated fisheye footprint rather than a fixed
angular-radius cut. ``--field-edge-threshold-fraction`` controls the
center-to-corner intensity threshold used to infer that footprint (default
``0.2``). The historical ``--footprint-threshold-fraction`` spelling remains an
alias. ``--field-edge-keep-margin-px`` then optionally requires source centroids
to lie an additional full-sensor distance inside the inferred footprint before
they are retained. Its default is ``0``, so the established
``--footprint-erosion-px`` conservative erosion defines the accepted edge unless
an extra guard band is requested. Optional ``--grid-search-radius-deg`` and
``--grid-acceptance-radius-deg`` settings intersect those image-driven masks with
Grid-calibrated angular-radius caps. For an exact reproduction of the historical
75-degree search / 70-degree acceptance field, use::

   --no-use-provisional-field-mask \
   --grid-search-radius-deg 75 \
   --grid-acceptance-radius-deg 70

Static ``--field-mask`` exclusions are combined with both regions when supplied.
``--field-mask-keep-margin-px`` can additionally require retained centroids to
lie a full-sensor distance inside the static-mask boundary. This is an acceptance
erosion only: background estimation and source search may still use the complete
static allowed region.

Parallel multichannel detection
-------------------------------

Grid-assisted source extraction can distribute independent images across worker
processes with ``--workers N``. The default ``--workers 1`` preserves the serial
reference path. Each worker loads the Grid calibration once and then processes
whole images; the four native Bayer channels remain sequential within an image.
Results are reordered deterministically before the detection sequence is written,
so changing worker count does not change detection-product provenance.

Start with a modest worker count because each worker holds its own image, local
background maps, and Grid calibration state.

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
       sidereal_rotation.npz      # when --grid-calibration is supplied
       bright_star_catalog.npz     # after the first orientation solve
       absolute_orientation.npz    # when --solve-orientation is supplied
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


Grid-calibrated sidereal rotation
---------------------------------

Supplying the portable ``*_calibration.npz`` artifact from the Grid Calibration
package enables the first physical astrometry stage::

   gonet-astrometry run /path/to/night \
       --algorithm sep \
       --output-dir gonet_astrometry_output \
       --grid-calibration /path/to/202_250116_204846_calibration.npz

The Grid package is imported only when this option is used. The artifact sensor
dimensions and explicit upper-left ``x=column, y=row`` convention are validated
against the full native Bayer coordinates stored in ``detections.npz``. Pixel
coordinates are inverted through the complete Grid harmonic model without
extrapolating beyond its calibrated angular radius. Track points outside that
domain are marked unavailable rather than aborting the complete fit.

For every sufficiently long bootstrap track, the calibrated directions are
converted to unit rays in the Grid frame. A robust optimizer then requires the
rays to follow one common rotation axis at the fixed sidereal angular rate. No
uniform image cadence is assumed; every rotation angle is computed from the
actual exposure midpoint. The resulting track classes are
``sidereal-consistent``, ``sidereal-rejected``, and ``insufficient``.

The common-axis fit is deliberately robust to the large number of fragmented or
non-stellar bootstrap tracks that can occur in a full-night sequence.  The
solver first estimates an unsigned small-circle axis from each sufficiently
curved ``candidate`` track, finds the largest antipodal consensus of those
per-track axes, uses the timestamps and fixed sidereal rate to choose the
rotation sense, and then refines only tracks that are already broadly
consistent with that preliminary physical model.  ``low-motion`` and
``poor-fit`` bootstrap tracks are still scored by the final solution but do not
normally determine the fitted pole.

The fitted product is written as ``sidereal_rotation.npz`` and, like the other
mid-level products, contains no pickle/object arrays. Its provenance depends on
the tracking product, Grid-calibration file fingerprint, solver configuration,
and a manual solver revision. Compatible solutions are reused automatically.
Use ``--overwrite-solution`` to rerun only this stage while preserving cached
detections and tracks. ``--overwrite-products`` still forces the complete
workflow from detection onward.

The principal fitting controls are ``--sidereal-min-track-points``,
``--sidereal-min-duration-minutes``, ``--sidereal-min-span-deg``,
``--sidereal-robust-scale-deg``, and ``--sidereal-consistent-rms-deg``. Long
runs can limit optimization cost with ``--sidereal-max-fit-tracks`` and
``--sidereal-max-points-per-track``; all tracks are scored after the global axis
has been fitted.

PDF diagnostic
--------------

Without a Grid calibration the generated PDF contains two pages. The first is
the static image-plane
tracking view based on the first discovered input image: selected compact native
channel, usable-field boundary, dynamic bright-region mask, reference-image
detections, and complete bootstrap track histories.

The second page is deliberately temporal. It plots the actual consecutive
exposure intervals, detections per track, track temporal spans, and epoch
coverage. Summary statistics include minimum, median, 90th-percentile, and
maximum cadence intervals plus track-length and duration statistics. These
measurements make track fragmentation visible without assuming that the input
images are evenly spaced in time.


When ``--grid-calibration`` is supplied, a third page summarizes the shared
sidereal fit. It overlays consistent/rejected/insufficient tracks, marks the
fitted apparent rotation pole when it lies inside Grid coverage, plots the
per-track de-rotated residual distribution and residual versus track duration,
and records the Grid-frame pole vector and global RMS/median/P95 residuals.

Absolute camera orientation
---------------------------

The common sidereal rotation axis fixes only two of the camera attitude's three
rotational degrees of freedom.  One exact rotation about the celestial pole
remains unconstrained until at least one absolute celestial direction is
identified.  GONet Astrometry resolves that remaining twist by matching
sidereal-consistent, de-rotated stellar tracks to a bright-star catalog.

Enable this stage together with a portable Grid calibration::

   gonet-astrometry run /path/to/night \
       --algorithm sep \
       --output-dir gonet_astrometry_output \
       --grid-calibration /path/to/camera_calibration.npz \
       --solve-orientation

The first run uses the VizieR Bright Star Catalogue when no local cache exists
and writes a portable ``bright_star_catalog.npz`` beside the other workflow
products.  Later runs reuse that cache and do not require another catalog
query.  A different cache location can be supplied with ``--catalog-cache``.
Fetching a missing cache requires the optional ``catalog`` dependency (included
in the development extra).

The solver first de-rotates every sufficiently long sidereal-consistent track
to one common reference epoch.  Fragmented tracklets that collapse to the same
stellar direction are de-duplicated.  Before catalog matching, every valid
point in each candidate track is also assigned an independent declination from
its angle to the fitted NCP.  Tracks with unstable declination are excluded from
the orientation anchors; both a robust scatter limit and a P95 absolute
deviation limit are used so that a few bad associations cannot hide behind a
small MAD.

The fitted sidereal axis identifies each surviving anchor's declination,
reducing catalog matching to one unknown angular twist about the North
Celestial Pole.  The default declination gate is deliberately tight because the
real multi-image tracks constrain declination at the arcminute level.  Rather
than accumulating independent observed/catalog phase pairs, the solver scans
the complete 0--360 degree twist range and scores how many anchors can be
registered simultaneously to a sparse bright-star subset.  The strongest
separated candidate twists are refined, checked with a one-to-one assignment,
then expanded to the full configured catalog for the final attitude fit.

Catalog directions are transformed to the local geometric horizon with Astropy.
The local frame is right-handed ENU: ``+x`` east, ``+y`` north, and ``+z`` up.
The resulting matrix obeys::

   ray_enu = grid_to_enu @ ray_grid

The saved ``absolute_orientation.npz`` product contains that full rotation
matrix, the reference epoch and observing location, the NCP in both coordinate
frames, the fitted pole-axis twist, catalog matches, and residual statistics.
It is pickle-free and provenance-checked against the sidereal product, catalog
cache, and orientation settings.  Use ``--overwrite-orientation`` to recompute
only this stage while preserving detections, tracks, and the sidereal solution.

The default catalog matching excludes stars below 10 degrees geometric altitude
to reduce sensitivity to unmodeled atmospheric refraction.  The current
bright-star cache uses catalog positions as stored and does not yet propagate
individual stellar proper motions; this is appropriate for the initial coarse
attitude solve but can be upgraded later for higher-precision catalog fitting.

With ``--solve-orientation`` the PDF gains a fourth page showing matched stellar
anchors, the recovered NCP/zenith/north/east directions in image coordinates,
catalog-match residuals, the Grid-to-ENU matrix, the optical-axis azimuth and
altitude, and orientation-fit summary statistics.
