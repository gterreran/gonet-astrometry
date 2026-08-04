Coordinate systems
==================

Every transformation must document axis direction, handedness, origin, pixel
centers, and array indexing. Silent convention changes are considered critical
calibration defects.

Native sensor coordinates
-------------------------

Images remain in their original two-dimensional Bayer mosaic. Coordinates are
zero-indexed floating-point sensor positions. ``x`` increases with array column
and ``y`` increases with array row unless the inspected GONet Wizard contract
requires an explicitly documented conversion.

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
