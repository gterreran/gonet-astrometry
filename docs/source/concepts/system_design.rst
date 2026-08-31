System design
=============

Problem formulation
-------------------

Stars follow rotations about the celestial pole on the unit celestial sphere.
A fisheye projection does not generally preserve those trajectories as
Euclidean circles in raw pixel coordinates. A conservative image-plane linker
may bootstrap short source associations before an approximate pixel-to-ray map
is available. That bootstrap linker uses the actual exposure midpoints for
motion gates, track lifetime, and extrapolation; it never assumes a uniform
cadence or a fixed burst pattern. The physical temporal model then converts
those tracklets to camera-frame unit rays before fitting the common celestial
rotation.

The production stellar geometry compares an observed raw pixel with a catalog
direction transformed through:

#. catalog celestial coordinates;
#. geometric local coordinates at the exposure midpoint and GPS location;
#. one rigid camera-to-ENU attitude;
#. a compact direct intrinsic camera model;
#. native full-sensor coordinates.

The Grid calibration is used only to bootstrap the first reliable stellar
identities.  It is deliberately excluded from the final intrinsic geometry,
because the physical Grid measurement contains both camera distortion and
calibration-target imperfections. Once a portable stellar calibration exists,
new observing sequences use it directly for multichannel detection geometry,
per-frame catalog projection, and optional spherical fallback tracking; the Grid
package is not loaded. See :doc:`stellar_camera_calibration` for the empirical
model-selection evidence, production fitting policy, and the final post-hoc
Grid-versus-stellar comparison.  That comparison removes the best rigid frame
alignment before evaluating residual geometry, so it is a diagnostic of
non-rigid mapping disagreement rather than coordinate-convention differences.

Architecture boundaries
-----------------------

The numerical core does not depend on a GUI. GONet Wizard reuse is confined to
compatibility adapters, allowing Wizard internals to evolve without spreading
coupling across the astrometry package.

The initial boundaries are:

``io``
   Native image and portable calibration-product loading protocols.
``adapters``
   GONet Wizard-specific compatibility code.
``detection``
   Interchangeable source-detection backends.
``geometry``
   Coordinate conventions and pixel/ray transformations.
``tracking``
   Temporal association and common-axis estimation.
``catalogs``
   Local or remote catalog access.
``calibration``
   Direct intrinsic camera models and stellar geometric fitting.
``solving``
   Sidereal/orientation matching, robust optimization, and validation.
``diagnostics``
   Overlays, residual maps, and calibration reports.

Modeling constraints
--------------------

The final camera solver uses the seven-parameter radial ``poly3`` model selected
by grouped-star cross-validation.  Higher radial orders, affine/tangential
terms, and generic two-dimensional polynomials were retained as research
comparators but did not improve held-out performance enough to justify their
complexity.  Flexible residual fields should not be added unless independent
data demonstrate spatial structure that the compact model cannot explain.
