.. Added: for new features.
.. Changed: for changes in existing functionality.
.. Deprecated: for soon-to-be removed features.
.. Removed: for now removed features.
.. Fixed: for any bug fixes.
.. Security: in case of vulnerabilities.

###########
Change Log
###########

All notable changes to this project will be documented in this file.
This project adheres to `Semantic Versioning <http://semver.org/>`_.

v0.1-beta
************

Added
-----

- Equalization assay analysis
- Registration module (for co-acquisitions to generate mLSR-3D DL training data)
- Deep learning integration for nuclei (StarDist, 3dunet) and membranes (PlantSeg, 3dunet)
- BigDataViewer
- Anisotropic compact watershed (dependency on forked scikit-image).
- Median intensities from segments (dependency on forked scikit-image).
- Functionality for parallelization with multiprocessing.
- Membrane enhancement module.
- Non-proprietary software for all pipeline steps.


Changed.
-----

- CLI
- downsampling automated.
- Forked and adapted scikit-image for anisotropic compact watershed.
- Module names.
- Default directory structure and file naming.
- Simplification of pipelines.
- Homogeneous function calls with image path and parameter filepath.
- Demo data downloads from within notebooks on-the-go.


v0.1-alpha.1 2020-07-06
************

Added
-----

 - feature extraction demo notebook


v0.1-alpha 2020-06-19
************

Added
-----

Core STAPL-3D functionality, i.e. modules for
 - preprocessing: shading correction
 - preprocessing: mask generation
 - preprocessing: bias field correction
 - segmentation: segmentation
 - segmentation: zipping
 - segmentation: feature extraction
 - channel operations
 - report generation
 - result backprojection
 - imaris file manipulation
 - general image io

Scripts for usage on HPC

Pipeline for usage from Python

Demos for preprocessing and segmentation
