# Stitching in Zen for STAPL3D.

This manual details the procedure for data processing in Zen Blue, when used for stitching as part of the STAPL3D preprocessing pipeline.

It consists of 

1. splitting the raw data into channels;
2. applying the channel-specific shading profiles (as estimated by STAPL-3D's shading correction module);
3. merging the corrected channels;
4. stitching the corrected z-stacks.

Required input:

- Raw data file: .czi format
- STAPL3D shading profile for each channel: .tiff format, found in the 'shading' subdirectory.

Output:
  - Shading-corrected and stitched .czi or .ims files.

## 1. Split raw data into channels.

- Open the czi-file in Zen Blue.
- Wait for the Image pyramid creation to finish.
- Go to 'Processing' > 'Method' > 'Create Image Subset and Split'.
  - Choose the czi-file as input.
  - *[Choose 'MK_Stitch_01' as the settings]*.
  - Click apply: this will generate separate images with single-channels.
  - Cancel the Image pyramid creation of each single-channel image.
- Wait for the splitting to finish.
- Close the original czi-file. No need to save it.

## 2. Apply STAPL-3D shading profile.

> **NB**: Zen-channels are numbered *1-8* but STAPL3D shading profiles are numbered *0-7*!!

For each channel (can be done simultaneously):

- Open the STAPL3D shading profile (tiff-file).
- Go to 'Processing' > 'Method' > 'Shading Correction'.
  - Choose the single-channel image as input.
  - Choose the STAPL3D shading profile as shading reference.
  - *[Choose 'MK_Stitching_02' as settings preset]*.
  - Click apply: this will generate shading-corrected-single-channel images.
  - Cancel the Image pyramid creation of each shading-corrected-single-channel image.

- Wait for shading correction to finish (for all channels).
-	Close (uncorrected) single-channel files and STAPL3D shading profiles. No need to save them.

## 3. Merge corrected channels.

For each single-channel pair (simultaneously):

- Go to 'Processing' > 'Method' > 'Add Channel'.
  - Choose two shading-corrected-single-channels as input:
    - ch1 + ch2 => ch1-2
    - ch3 + ch4 => ch3-4
    - ch5 + ch6 => ch5-6
    - ch7 + ch8 => ch7-8
  - Click Apply: this with generate double-channel images.
  - Cancel the Image pyramid creation for each double-channel image.
- Wait for the merging to finish (for all pairs).
- Close each shading-corrected-single-channel image. No need to save them.

For each double-channel pair (simultaneously):

- Go to 'Processing' > 'Method' > 'Add Channel'. 
  - Choose two double-channel images as input:
    - ch1-2 + ch3-4 => ch1-4
    - ch5-6 + ch7-8 => ch5-8
  - Click Apply: this with generate quad-channel images.
  - Cancel the Image pyramid creation for each quad-channel image.
- Wait for the merging to finish (for all quads).
- Close each double-channel image. No need to save them.

For the quad-channel pair:

- Go to 'Processing' > 'Method' > 'Add Channel'.
  - Choose the two quad-channel images as input:
    - ch1-4 + ch5-8 => ch1-8
  - Click Apply: this with generate an 8-channel image.
  - Cancel the Image pyramid creation for the 8-channel image.
- Wait for the merging to finish.
- Close quad-channel images. No need to save them.

## 4. Stitch corrected channels.
- Go to 'Processing' > 'Method' > 'Stitching'.
  - Choose the 8-channel shading-corrected file as input.
  - *[Choose 'MK_Stitching_03' as settings]*.
  - Find the best Z-plane for stitching: the deepest plane where you still have ample DAPI signal is recommended.
  - Click apply: this will generate a stitched 8-channel image.
  - Cancel the Image pyramid creation for the stitched file.
- Wait for stitching to finish.
- Save shading-corrected stitched file as czi.
