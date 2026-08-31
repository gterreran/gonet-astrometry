Development roadmap
===================

Milestone 1: data foundation
----------------------------

* inspect current GONet Wizard image and Grid-calibration APIs;
* implement compatibility adapters;
* validate metadata, exposure midpoint, and sensor orientation;
* visualize the pixel-to-ray mapping;
* establish synthetic round-trip geometry tests.

Milestone 2: native Bayer detection
------------------------------------

* estimate background and noise by Bayer channel;
* benchmark segmentation and point-source detection backends;
* preserve uncertain detections with quality flags;
* model channel-dependent amplitudes around a common centroid;
* produce detection overlays and quality diagnostics.

Milestone 3: temporal tracking and sidereal axis
------------------------------------------------

* bootstrap conservative image-plane tracklets across timestamps;
* convert associated detections to Grid-frame camera rays;
* estimate the common celestial rotation axis in ray space;
* reject static lights, aircraft, clouds, and inconsistent transients;
* diagnose timestamp, cadence, and camera-motion errors.

Milestone 4: absolute camera orientation
----------------------------------------

* de-rotate sidereal-consistent tracks to one common epoch;
* match the resulting stellar anchors to a bright-star catalog;
* resolve the final attitude twist about the celestial pole;
* recover and validate the complete Grid-to-local-horizon rotation matrix;
* generate catalog-match and local-direction diagnostics.

Milestone 5: direct stellar camera calibration
----------------------------------------------

* use the Grid solution only to bootstrap high-confidence stellar identities;
* rematch catalog stars directly against raw detections with the stellar model;
* fit a compact radial ``poly3`` intrinsic camera model and 3-D attitude;
* validate by grouped catalog-star cross-validation rather than random rows;
* restrict the production fit to dark-time, high-altitude stellar measurements;
* export a versioned, pickle-free stellar camera calibration with provenance.

This milestone is complete.  Once ``stellar_camera_calibration.npz`` exists,
``--stellar-calibration`` replaces Grid geometry entirely for normal catalog
projection, one-to-one stellar identification, catalog tracking, and optional
hybrid fallback tracking.  The Grid remains only as a commissioning/bootstrap
tool for a camera that does not yet have a stellar calibration.

Milestone 6: calibration-workflow portal
-----------------------------------------

* redesign the portal around user workflows rather than exposing CLI flags one
  control at a time;
* distinguish camera commissioning (Grid bootstrap -> first stellar calibration)
  from normal operation with an existing stellar calibration;
* expose calibration-product selection, cache/provenance state, catalog
  identification, and validation diagnostics coherently;
* keep all numerical calibration and matching logic in the GUI-independent core.
