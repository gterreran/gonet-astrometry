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

The direct stellar camera calibration defines a right-handed camera frame with
``+z`` along the optical axis and ``+x``/``+y`` aligned with increasing native
sensor ``x``/``y``.  The portable camera-to-ENU matrix maps camera-frame column
vectors into the local east/north/up frame.

The intrinsic ``poly3`` mapping uses angular distance ``theta`` from ``+z`` and
raw full-sensor radius about the fitted optical center.  Grid-frame axes are not
part of this final convention; their in-plane orientation was found empirically
to differ from the native sensor frame by about 57 degrees in the Adler
bootstrap data.

Local and celestial frames
--------------------------

Catalog positions are evaluated at the exposure midpoint and GPS-derived
observing location in a zero-pressure Astropy ``AltAz`` frame.  They therefore
represent geometric/topocentric directions. Atmospheric refraction is not
absorbed into the intrinsic camera model; any future apparent-sky correction
will be modeled as a separate physical layer and recorded explicitly in
provenance.
