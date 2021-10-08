#!/usr/bin/env python

"""
# STAPL3D equalization assay analysis.

> NOTE: STAPL3D development branch (dev1.0.0) required to run this pipeline.

- File organization (recommended):

  > NB: ensure each filename is unique, even if they are in separate directories.

  - organize files in a directory structure according to:
    <datadir>/<species>/<antibody>/<repetition>.<ext>

  - Primaries should be included as follows:
    <datadir>/primaries/<antibody>/<repetition>.<ext>

- Run equalization analysis:

  - Option 1: from terminal with default options:

    >>> conda activate stapl3d
    >>> python <path_to_stapl3d>/stapl3d/equalization.py -i <path_to_datadir>

  - Option 2: from python with default options:

    - in terminal:

      >>> conda activate stapl3d
      >>> ipython

    - in python:

      >>> from stapl3d import equalization

      >>> datadir = '.'

      >>> equaliz3r = equalization.Equaliz3r(datadir)
      >>> equaliz3r.run()

"""
