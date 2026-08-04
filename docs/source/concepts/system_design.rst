System design
=============

Problem formulation
-------------------

Stars follow rotations about the celestial pole on the unit celestial sphere.
A fisheye projection does not generally preserve those trajectories as
Euclidean circles in raw pixel coordinates. The calibration pipeline therefore
converts detections from sensor coordinates to camera-frame unit rays before
fitting temporal motion.

The eventual joint model will compare an observed pixel with a projected
catalog direction transformed through:

#. catalog celestial coordinates;
#. apparent local coordinates at the exposure midpoint and GPS location;
#. camera attitude relative to the local frame;
#. lens and Grid-calibration geometry;
#. native sensor coordinates.

Architecture boundaries
-----------------------

The numerical core does not depend on a GUI. GONet Wizard reuse is confined to
compatibility adapters, allowing Wizard internals to evolve without spreading
coupling across the astrometry package.

The initial boundaries are:

``io``
   Native image and Grid-calibration loading protocols.
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
``solving``
   Matching, robust optimization, and validation.
``diagnostics``
   Overlays, residual maps, and calibration reports.

Modeling constraints
--------------------

The first solver should use a compact, interpretable geometric model. Flexible
residual fields may be introduced only after correct source associations and
basic lens geometry are demonstrated, and they must be regularized to avoid
absorbing matching errors.
