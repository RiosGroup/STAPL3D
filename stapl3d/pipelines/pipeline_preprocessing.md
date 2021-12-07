# STAPL3D preprocessing.

- Recommended: organize files in a directory structure according to:
    *\<projectdir\>/\<dataset\>/\<dataset\>.\<ext\>*

- Run STAPL3D shading estimation:

  > Note: the final step of *STAPL3D shading correction* `apply` is not required when stitching in Zen and is avoided here.

  - Option 1: from terminal with default options:

    ```
    conda activate stapl3d
    python <path_to_stapl3d>/stapl3d/preprocessing/shading.py -i <path_to_raw_data> -x <dataset> -s estimate process postprocess
    ```

  - Option 2: from python with default options:
    - in terminal:
      ```
      conda activate stapl3d
      ipython
      ```
    - in python:
      ```
      from stapl3d.preprocessing import shading

      dataset = 'subset_tiled'
      filepath_raw = f'{dataset}.czi'
      steps = ['estimate', 'process', 'postprocess']

      deshad3r = shading.Deshad3r(filepath_raw, prefix=dataset)
      deshad3r.run(steps)
      ```

  - Option 3: from terminal with parameter file:

    ```
    conda activate stapl3d
    python <path_to_stapl3d>/stapl3d/preprocessing/shading.py -i <path_to_raw_data> -p <path_to_yaml_file> -x <dataset> -s estimate process postprocess
    ```

    yaml-file setup:
      ```
      shading:
        estimate:
          # Threshold to discard background, default 1000.
          noise_threshold: 2000
          # Metric for creating profiles, default 'median', else 'mean'.
          metric: mean
        process:
          # Quantile at which planes are discarded, default 0.8.
          quantile_threshold: 0.6
          # Order of the polynomial to fit the profile, default 3.
          polynomial_order: 3
      ```

  - Option 4: from python with attribute specification:
    - in terminal:
      ```
      conda activate stapl3d
      ipython
      ```
    - in python:
      ```
      from stapl3d.preprocessing import shading

      dataset = 'subset_tiled'
      filepath_raw = f'{dataset}.czi'
      steps = ['estimate', 'process', 'postprocess']

      deshad3r = shading.Deshad3r(filepath_raw, prefix=dataset)

      deshad3r.channels = [1, 3, 5, 7]  # process a subset of channels
      deshad3r.planes = list(range(20, 50))  # process a subset of planes
      deshad3r.noise_threshold = 500  # reduced noise threshold
      deshad3r.quantile_threshold = 0.7  # reduced quantile threshold

      deshad3r.run(steps)
      ```

- Stitch in Zen according to the procedure in [stitching_zen](pipeline_stitching_zen.md)

- Optional: convert the stitched file to Imaris 5.5 file format in Imaris File Converter.

- Run 3D inhomogeneity correction:
  ```
  conda activate stapl3d-dev1.0.0
  python <path_to_stapl3d>/stapl3d/preprocessing/biasfield.py -i <path_to_stitched_data> -x <dataset>
  ```
