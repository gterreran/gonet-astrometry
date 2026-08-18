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

Milestone 3: Grid-assisted absolute solution
---------------------------------------------

* transform a bright-star catalog to local apparent coordinates;
* bootstrap camera attitude from the Grid calibration;
* match detections and generate residual maps;
* validate single-frame solution limits.

Milestone 4: temporal tracking
------------------------------

* bootstrap conservative image-plane tracklets across timestamps;
* convert associated detections to camera rays;
* estimate the common celestial rotation axis in ray space;
* reject static lights, aircraft, clouds, and inconsistent transients;
* diagnose timestamp and camera-motion errors.

Milestone 5: joint nightly calibration
---------------------------------------

* fit all matched frames with robust losses;
* refine attitude and geometric calibration;
* quantify uncertainty and parameter degeneracies;
* export a versioned calibration with provenance.
