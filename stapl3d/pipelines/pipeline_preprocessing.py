#!/usr/bin/env python

"""Run STAPL3D preprocessing.

- Organize files in a directory structure according to:
    <projectdir>/<dataset>/<dataset>.<ext>

- Run shading correction:
>>> conda activate stapl3d-dev1.0.0
>>> python <path_to_stapl3d>/stapl3d/preprocessing/shading.py -i <path_to_raw_data> -x <dataset> -s estimate process postprocess

- Stitch in Zen according to the procedure in <path_to_stapl3d>/stapl3d/pipelines/stitching_zen.md

- Optional: convert the stitched file to Imaris 5.5 file format in Imaris File Converter.

- Run 3D inhomogeneity correction:
>>> conda activate stapl3d-dev1.0.0
>>> python <path_to_stapl3d>/stapl3d/preprocessing/biasfield.py -i <path_to_stitched_data> -x <dataset>

"""
