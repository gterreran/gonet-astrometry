GONet Astrometry Calibrator
===========================

``gonet-astrometry`` is an independent, GONet Wizard-adjacent package for
astrometrically calibrating wide-field GONet images from stellar detections and
temporal tracks.

The project is currently pre-alpha but now includes validated native Bayer
detection, catalog-first/hybrid stellar tracking, and a Grid-independent direct
stellar camera calibration. The physical Grid remains supported as a bootstrap
registration tool; the production intrinsic camera geometry is fitted directly
from catalog stars and raw sensor centroids.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   concepts/system_design
   concepts/coordinate_systems
   concepts/stellar_camera_calibration
   concepts/development_roadmap
   user_guide/portal
   user_guide/cli_run
   developer_guide/contributing
   api/index
