Coordinate systems
==================

Every transformation must document axis direction, handedness, origin, pixel
centers, and array indexing. Silent convention changes are considered critical
calibration defects.

Native sensor coordinates
-------------------------

Scientific frames remain in the original two-dimensional BGGR Bayer mosaic.
The Wizard compact channels are expanded through the Wizard's own Bayer-plane
mapping and combined without demosaicing or interpolation. Coordinates are
zero-indexed sensor positions: ``x`` increases with array column and ``y``
increases with array row. Portal channel previews use compact per-channel
coordinates and are therefore display products rather than scientific frames.

Camera rays
-----------

Grid calibration converts sensor positions to normalized three-dimensional
camera-frame vectors. The definitive camera-axis convention will be selected
after inspecting the current Grid calibration implementation and serialized
output.

Local and celestial frames
--------------------------

Apparent catalog positions will be evaluated at the exposure midpoint and at
the GPS-derived observing location. Atmospheric refraction, clock offsets, and
per-frame camera motion will be added incrementally and recorded explicitly in
the solution provenance.
