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

Milestone 5: joint nightly calibration
---------------------------------------

* fit all matched frames with robust losses;
* refine attitude and geometric calibration;
* quantify uncertainty and parameter degeneracies;
* export a versioned astrometric calibration with provenance.
