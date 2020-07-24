.. list-table::
   :widths: 25 25
   :header-rows: 1

   * - fair-software.nl recommendations
     - Badges
   * - \1. Code repository
     - |GitHub Badge|
   * - \2. License
     - |License Badge|

.. |GitHub Badge| image:: https://img.shields.io/badge/github-repo-000.svg?logo=github&labelColor=gray&color=blue
   :target: https://github.com/RiosGroup/stapl3d
   :alt: GitHub Badge

.. |License Badge| image:: https://img.shields.io/github/license/RiosGroup/stapl3d
   :target: https://github.com/RiosGroup/STAPL3D
   :alt: License Badge

################################################################################
STAPL-3D
################################################################################
Overview
--------
STAPL-3D (SegmenTation Analysis by ParaLlelization of 3D Datasets) is an analysis tool for mLSR-3D or other high dimensional 3D fluorescency microscopy datasets.

In brief, the STAPL-3D pipeline is characterized by:
 -	distributed processing with deployment on computing clusters or workstations;
 -	inclusion of quality assurance reports;
 -	modularity of processing steps;
 -	maximal data retrieval;
 -	focus on spatial aspects of the data.

What STAPL-3D does to allow accurate analysis can be summarized as:
 -	correction of inhomogeneity artefacts;
 -	segmentation with subcellular resolution;
 -	zipping of blocks for lossless analysis;
 -	feature extraction for spatio-phenotypic patterning;
 -	backprojection for 3D exploration of analysis results in native imaging space.

.. A STAPL-3D legacy repository with potentially useful additional code can be found here: https://github.com/michielkleinnijenhuis/segmentation

System requirements
-------------------
Hardware requirements
*********************

STAPL-3D is created for deployment on High Performance Computing clusters and tested with SGE and SLURM job schedulers. STAPL-3D can also be run on (high-end) workstations. Our lab uses HP Z8 G4 with dual Intel Xeon Gold 5122 3.6 GHz processors, 1 TB RAM, 3x2TB SSD + 6x5TB HDD in RAID-5, and an Nvidia Quadro P4000 8GB graphics card. This workstation setup can process a 100 x 15000 x 15000 x 8 voxel dataset within a day. More modest systems are also feasible: the RAM requirements scale with the chosen parallelization blocksize and is approximately 40GB for a blocksize of 100x1280x1280 voxels for the membrane enhancement step. We would recommend a minimum of 16GB of RAM.

Software requirements
*********************
This software runs on Linux, MacOS and Windows. It has been tested on:

 - Linux: CentOS7
 - MacOS: Catalina 10.15.5, Mojave 10.14.1
 - Windows: 10 Pro

dependencies:
Python (auto-installed through with conda or pip)

 - python=3
 - scikit-image>=0.16.2
 - h5py>=2.10.0
 - matplotlib>=3.1.3
 - numpy>=1.18.1
 - pyyaml>=5.3.1
 - scipy>=1.4.1
 - pandas>=1.0.4
 - pypdf2>=1.26.0

 - czifile>=2019.7.2
 - nibabel>=3.1.0
 - simpleitk>=2.0.0rc1.post285

Other software packages:

 - `ACME <https://wiki.med.harvard.edu/SysBio/Megason/ACME>`_
 - `Ilastik <https://www.ilastik.org/documentation/basics/installation.html>`_ (optional)


Installation
------------

STAPL-3D is most easily installed (< 5 min) with `conda <https://docs.conda.io/en/latest>`_ by pasting the following code in the terminal:

.. code-block:: console

  git clone https://github.com/RiosGroup/STAPL3D.git
  cd STAPL3D

  conda env create -f environment.yml
  conda activate stapl3d

  python setup.py install
  pip install --pre SimpleITK --find-links https://github.com/SimpleITK/SimpleITK/releases/tag/latest

To use membrane enhancement in the pipeline, `ACME <https://wiki.med.harvard.edu/SysBio/Megason/ACME>`_ needs to be installed. To use the machine-learning channel clean-up procedures, `Ilastik <https://www.ilastik.org/documentation/basics/installation.html>`_ needs to be installed. **To run the demos, these two external packages are not required** as we provide these steps.

Getting started
---------------

Demos
*****
`Jupyter notebooks <https://jupyter-notebook-beginner-guide.readthedocs.io/en/latest/index.html>`_ are provided in the demos directory to showcase the basic functionality of STAPL-3D. The data for the demo (~6GB) can be downloaded `here <https://surfdrive.surf.nl/files/index.php/s/Q9wRT5cyKGERxI5>`_.
 - preprocessing_demo (runtime: 20 min)
 - segmentation_demo (runtime: 20 min)
 - feature_extraction_demo (runtime: 10 min)

If you installed STAPL-3D using conda, you can start a notebook in the stapl3d enviroment as follows:

.. code-block:: console

  conda activate stapl3d
  conda install jupyter
  python -m ipykernel install --user --name=stapl3d
  jupyter notebook

HPC deployment
**************

 - copy the file stapl3d/pipelines/.stapl3d.ini to your HPC home directory
 - adapt the paths in .stapl3d.ini for:
    - STAPL3D: stapl3d package directory
    - ACME: directory with the ACME binaries
    - FIJI: path to fiji executable
    - ILASTIK: path to run_ilastik.sh

Basic instruction for running STAPL3D on your own data
******************************************************

 - create a directory <datadir> for the <dataset>
 - generate a parameter file <datadir>/<dataset>.yml for your dataset: use stapl3d/pipelines/params.yml as a template
 - upload the datafile <datadir>/<dataset>.czi
 - an example pipeline for HPC usage is provide in stapl3d/pipelines/pipeline.sh
 - an example pipeline for python usage is provided in stapl3d/pipelines/pipeline.py

Contributing
------------

If you want to contribute to the development of STAPL3D,
have a look at the `contribution guidelines <CONTRIBUTING.rst>`_.

License
-------

Copyright (c) 2020,

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Credits
-------

This package was created with `Cookiecutter <https://github.com/audreyr/cookiecutter>`_ and the `NLeSC/python-template <https://github.com/NLeSC/python-template>`_.
