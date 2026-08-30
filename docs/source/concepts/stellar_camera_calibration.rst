Direct stellar camera calibration
=================================

Purpose
-------

The production geometric calibration is fitted directly from stars rather than
as a correction to the portable Grid calibration.  The Grid remains valuable as
an initial registration tool: it places catalog stars close enough to raw
source detections that robust stellar identities can be established.  Once
those identities exist, however, the calibration target is the sky itself.

For each matched stellar measurement the final geometric problem is therefore

.. code-block:: text

   catalog RA/Dec + exposure midpoint + GPS
                      |
                      v
             geometric local ENU ray
                      |
              rigid camera attitude
                      |
                      v
                camera-frame ray
                      |
            direct intrinsic model
                      |
                      v
             raw full-sensor (x, y)

No Grid ray or Grid angular coordinate enters the fitted intrinsic mapping.

Validation context
------------------

The empirical choices in this document come from the cleaned Adler-roof
validation sequence used during development: 695 open-shutter frames spanning
about 11.5 hours in Chicago. Source detection used the production 3.5-sigma SEP
threshold, the static Adler field mask, and a 32 px static-mask keep margin. The
stellar catalog was retained to V<=4.5 so the production calibration keeps some
sensitivity headroom for darker sites.

The catalog-identification stage found 147 catalog-labelled stellar tracks. In
hybrid tracking, the unmatched population added 38 generic temporal tracks, but
those fallback tracks are deliberately not treated as calibration stars: they
may include planets, satellites, aircraft, or other coherent non-sidereal
objects. Direct geometric calibration therefore uses only positive catalog-star
identities.

Why the Grid is not the final camera model
------------------------------------------

The physical Grid experiment measures the combination of at least two
geometries: the camera/lens projection and imperfections in the calibration
Grid itself (placement, shape, printing, alignment, and measurement error).  A
residual correction layered on the Grid mapping would therefore retain some
structure that does not belong to the camera.

The stellar experiment demonstrated this directly.  When the first direct
stellar sweep accidentally retained the Grid frame's in-plane axis convention,
compact radial models failed by hundreds of pixels while a generic two-
dimensional polynomial could absorb the missing rotation.  Re-estimating the
sensor-plane orientation from stars recovered an in-plane rotation of
``+57.42 deg`` with positive parity.  After that alignment, compact radial
models immediately reached approximately two-pixel held-out performance.  The
Grid was therefore an excellent bootstrap geometry, but not the simplest
intrinsic description of the camera.

Chosen intrinsic model
----------------------

The selected production model is a radial cubic (``poly3``):

.. math::

   t = \frac{\theta}{\pi/2},

.. math::

   r(\theta) = c_1 t + c_3 t^3.

``theta`` is angular distance from the fitted optical axis and ``r`` is raw
full-sensor pixel radius about the fitted optical center ``(c_x, c_y)``.  A
single proper 3-D rotation maps the camera frame to local east/north/up (ENU).
The complete fit therefore has seven parameters:

* three camera-attitude degrees of freedom;
* ``c_x`` and ``c_y``;
* radial coefficients ``c_1`` and ``c_3``.

The model sweep deliberately compared this compact form with textbook fisheye
projections, higher radial orders, affine/tangential terms, and generic 2-D
polynomials.  On Grid-free direct stellar labels, representative grouped-star
cross-validation results were:

============================  ======  ===============  ============
Model                         Params  Median [arcmin]  P90 [arcmin]
============================  ======  ===============  ============
``poly5_affine_tangential``       12            2.836         6.637
``poly5_tangential``              10            2.792         6.645
``poly3``                          7            2.982         6.807
``xy_poly3``                      23            3.062         6.828
``poly5``                          8            2.794         6.884
``equisolid``                      6            7.853        14.278
``equidistant``                    6           15.607        23.779
============================  ======  ===============  ============

The nominal 12-parameter winner improves P90 over ``poly3`` by only about
``0.17 arcmin`` (roughly ``0.045`` full-sensor pixel in this data set), while
``poly3`` is substantially simpler and has comparable held-out behavior.  The
production choice therefore favors the seven-parameter model rather than the
small numerical advantage of the more flexible fit.

Grid-independent rematching
---------------------------

Initial stellar identities are treated only as bootstrap labels.  Production
calibration performs the following conservative transition:

#. retain primary Grid-assisted matches with residual <=15 px, geometric
   altitude >=25 deg, and at least five observations per star;
#. perform grouped-by-star ``poly3`` cross-validation and temporarily exclude
   seed stars with P90 residual >20 arcmin;
#. fit the initial direct ``poly3`` camera model;
#. project the V<=4.5 catalog directly through that stellar camera model;
#. rematch raw detections with an 8 px one-to-one gate;
#. refit and tighten to a 5 px gate for two iterations;
#. fit the final camera calibration from the conservative dark/high-altitude
   subset.

On the Adler validation night this transition produced 13,213 direct stellar
matches belonging to 143 catalog stars.  Compared with the original
Grid-assisted assignments, 12,976 detection epochs retained the same identity,
57 were newly matched, 180 were relabeled, and 641 old assignments were not
retained.  The large 40--60 arcmin star-specific residuals seen before direct
rematching disappeared, showing that they were association errors rather than
missing camera-distortion terms.

Calibration sample and observational error floor
-------------------------------------------------

Catalog positions are transformed with Astropy at the exposure midpoint and GPS
location using zero pressure.  They are therefore *geometric/topocentric*
directions with no atmospheric refraction model baked into the camera
calibration.

The final production fit uses:

* Sun altitude <= ``-18 deg``;
* stellar geometric altitude >= ``30 deg``.

These limits were selected empirically rather than only by astronomical
convention.  With the Grid-free ``poly3`` model, grouped-star validation gave:

==================  ============  =====  ================  =============
Selection           Measurements  Stars  Median [arcmin]  P90 [arcmin]
==================  ============  =====  ================  =============
All retained              13,110     81             2.982          6.807
Sun <= -12 deg            12,548     80             2.962          6.680
Sun <= -18 deg            11,222     77             2.923          6.438
Sun <= -24 deg             9,886     72             2.844          6.104
Sun <= -18, Alt >=30      10,942     73             2.795          6.417
==================  ============  =====  ================  =============

The monotonic twilight improvement is consistent with reduced centroid
precision in bright sky: higher sky background increases photon noise in the
stellar profile, so the same source has a less precise measured center.  The
stellar-altitude sweep, by contrast, showed little improvement above 30 deg;
unmodeled atmospheric refraction is therefore not the dominant term at the
current several-arcminute validation floor.

The calibration artifact records the maximum camera-frame angular radius
actually represented by the final calibration sample.  Public projection and
inverse methods reject extrapolation beyond that range by default.  Callers
must opt into extrapolation explicitly.

Portable artifact
-----------------

``stellar_camera_calibration.npz`` is pickle-free and stores:

* image shape and explicit coordinate convention;
* the camera-to-ENU rotation matrix;
* ``c_x``, ``c_y``, ``c_1``, and ``c_3``;
* the validated angular range;
* direct-rematching statistics;
* grouped-star cross-validation median/P90 residuals;
* bootstrap rejection statistics;
* the complete fitting configuration and upstream product identities.

The artifact is geometrically independent of the Grid calibration even though
its provenance records the bootstrap stellar-identification product used to
start the first fit. The intrinsic model is recorded explicitly as
``radial-poly3``. Its radial mapping must remain positive and monotonic over the
calibrated angular range so the public forward and inverse transformations are
well defined.

Using the stellar calibration on later observations
---------------------------------------------------

Once ``stellar_camera_calibration.npz`` exists, the Grid is no longer required
for observations made with the same fixed camera/lens geometry.  The portable
artifact supplies both pieces that the initial bootstrap had to infer through
the Grid: the intrinsic pixel-to-ray mapping and the absolute camera-to-ENU
attitude.  A later observing sequence can therefore follow the shorter path

.. code-block:: text

   raw images
       |
       v
   multichannel SEP detections
       |
       +---- stellar_camera_calibration.npz
       |                 |
       |                 v
       |       catalog RA/Dec + time + GPS
       |                 |
       |                 v
       +----------> predicted raw (x, y)
                         |
                    5 px one-to-one match
                         |
                         v
              stellar_camera_identifications.npz
                         |
                  catalog-labelled tracks

The direct matcher uses the calibration's validated ``poly3`` geometry; it does
not solve another frame-wide attitude and does not consult any Grid coordinate.
Only catalog directions inside the stellar-calibrated angular range are
projected.  The static field mask and its keep margin remain pixel-space
constraints and may be reused unchanged when their sensor convention matches.

The command-line entry point is ``--stellar-calibration``.  Normal operation
uses catalog tracking, while hybrid mode additionally runs the existing
spherical temporal tracker on detections left unmatched by the catalog.  In
hybrid mode the fallback ray geometry is also supplied by the stellar camera
model; no Grid inversion is involved.  Fallback tracks remain generic coherent
objects and are not promoted to calibration stars merely because they track
well.

For example::

   gonet-astrometry run /data/night/*.jpg \
       --algorithm sep \
       --stellar-calibration /calibration/stellar_camera_calibration.npz \
       --field-mask /calibration/site_field_mask.npz \
       --field-mask-keep-margin-px 32 \
       --workers 8 \
       --tracking-mode catalog \
       --output-dir /data/night_astrometry

The first run writes ``detections.npz`` and
``stellar_camera_identifications.npz`` in the new output directory.  Compatible
reruns reuse both products.  A Bright Star Catalogue cache is still loaded once
per output directory (or from ``--catalog-cache``) and propagated to each
exposure time.

This makes the Grid a one-time bootstrap instrument rather than an operational
dependency of the calibrated camera.  A new Grid bootstrap is needed only when
no stellar calibration exists or when the physical camera/lens geometry has
changed enough that the existing stellar artifact is no longer applicable.

Current limitations
-------------------

* The Bright Star Catalogue coordinates are currently used as stored; proper
  motion propagation is not yet applied.
* Atmospheric refraction is intentionally excluded from the intrinsic camera
  geometry.  A future apparent-sky layer may model it separately.
* Creating the *first* stellar camera calibration still requires a bootstrap
  correspondence set.  The production bootstrap currently comes from the Grid;
  later observing sequences use the stellar calibration directly and do not
  require the Grid.
* Validation metrics are grouped by catalog star so a held-out star is absent
  from its fold's fit.  This is intentionally stricter than random row-level
  validation of repeatedly observed stellar tracks.
