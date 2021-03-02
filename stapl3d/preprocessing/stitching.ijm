// read dataset path, number of tiles as commandline arguments
args = getArgument()
args = split(args, " ");

stitch_step = args[0];
channel = args[1];
tif_path = args[2];
outputdir = args[3];
basename = args[4];
cfg_path = args[5];
elsize_z = args[6];
elsize_y = args[7];
elsize_x = args[8];

downsample_in_z = args[9];
downsample_in_y = args[10];
downsample_in_x = args[11];

min_r = args[12];
max_r = args[13];
max_shift_in_z = args[14];
max_shift_in_y = args[15];
max_shift_in_x = args[16];
max_displacement = args[17];

relative = args[18];
absolute = args[19];

xml_file = basename + "_stacks.xml";
xml_path = outputdir + File.separator + xml_file;
h5_stem = outputdir + File.separator + basename + "_stacks";
h5_fused = outputdir + File.separator + basename + ".xml";

if (stitch_step==1) {

    // define dataset
    // creates dataset.h5 and dataset.xml
    run("Define dataset ...",
        " define_dataset=[Automatic Loader (Bioformats based)]" +
        " project_filename=" + xml_file +
        " path=" + tif_path +
        " exclude=10" +
        " pattern_0=Channels pattern_1=Tiles" +
        " modify_voxel_size? voxel_size_x=" + elsize_x + " voxel_size_y=" + elsize_y + " voxel_size_z=" + elsize_z + " voxel_size_unit=µm " +
        " how_to_load_images=[Re-save as multiresolution HDF5]" +
        " dataset_save_path=" + outputdir +
        " subsampling_factors=[{ {1,1,1}, {2,2,2}, {4,4,4} }]" +
        " hdf5_chunk_sizes=[{ {16,16,16}, {16,16,16}, {16,16,16} }]" +
        " timepoints_per_partition=1" +
        " setups_per_partition=0" +
        " use_deflate_compression" +
        " export_path=" + h5_stem );

    }
else if (stitch_step==2) {

    // load tile positions from file
    // creates additional affine transforms in ViewRegistrations tag of dataset.xml
    run("Load TileConfiguration from File...",
        " browse=" + xml_path +
        " select=" + xml_path +
        " browse=" + cfg_path +
        " tileconfiguration=" + cfg_path +
        " use_pixel_units" +
        " keep_metadata_rotation");

    }
else if (stitch_step==3) {

    // calculate pairwise shifts
    // creates PairwiseResult entries in StitchingResults tag of dataset.xml
    run("Calculate pairwise shifts ...",
        " select=" + xml_path +
        " process_angle=[All angles]" +
        " process_channel=[All channels]" +
        " process_illumination=[All illuminations]" +
        " process_timepoint=[All Timepoints]" +
    	" process_tile=[All Tiles]" +
        " method=[Phase Correlation]" +
    	" how_to_treat_timepoints=[treat individually]" +
    	" how_to_treat_channels=[treat individually]" +
    	" how_to_treat_illuminations=[treat individually]" +
    	" how_to_treat_angles=[treat individually]" +
    	" how_to_treat_tiles=compare" +
        " channels=[use Channel 0]" +
        " downsample_in_x=" + downsample_in_x +
        " downsample_in_y=" + downsample_in_y +
        " downsample_in_z=" + downsample_in_z +
        " subpixel");

    }
else if (stitch_step==4) {

    // filter shifts with 0.7 corr. threshold
    // removes PairwiseResult entries from StitchingResults tag in dataset.xml
    run("Filter pairwise shifts ...",
        " select=" + xml_path +
        " filter_by_link_quality" +
        " min_r=" + min_r +
        " max_r=" + max_r +
        " max_shift_in_x=" + max_shift_in_x +
        " max_shift_in_y=" + max_shift_in_y +
        " max_shift_in_z=" + max_shift_in_z +
        " max_displacement=" + max_displacement);

    }
else if (stitch_step==5) {

    // do global optimization
    // creates additional affine transforms in ViewRegistrations tag of dataset.xml
    run("Optimize globally and apply shifts ...",
        " select=" + xml_path +
        " process_angle=[All angles]" +
        " process_channel=[All channels]" +
        " process_illumination=[All illuminations]" +
        " process_tile=[All tiles]" +
        " process_timepoint=[All Timepoints]" +
        " relative=" + relative +
        " absolute=" + absolute +
        " global_optimization_strategy=[Two-Round using Metadata to align unconnected Tiles]" +
        " fix_group_0-0,");

    }
else if (stitch_step==6) {

    // fuse dataset, save as hdf5
    // creates output dataset-f0.h5 and dataset-f0.xml (new:fused_image=[Save as new XML Project (HDF5)]); OR
    // creates output dataset-m0.h5 and dataset-f1.h5 (append: fused_image=[Append to current XML Project (HDF5)])
    run("Fuse dataset ...",
        " select=" + xml_path +
        " process_angle=[All angles]" +
        " process_channel=[All channels]" +
        " process_illumination=[All illuminations]" +
        " process_tile=[All tiles]" +
        " process_timepoint=[All Timepoints]" +
        " bounding_box=[All Views]" +
        " downsampling=1" +
        " pixel_type=[16-bit unsigned integer]" +
        " interpolation=[Linear Interpolation]" +
        " image=Cached" +
        " interest_points_for_non_rigid=[-= Disable Non-Rigid =-]" +
        " blend" +
        " preserve_original" +
        " produce=[Each timepoint & channel]" +
        " fused_image=[Save as new XML Project (HDF5)]" +
        " subsampling_factors=[{ {1,1,1}, {2,2,1}, {4,4,2}, {8,8,4}, {16,16,8} }]" +
        " hdf5_chunk_sizes=[{ {32,32,4}, {16,16,16}, {16,16,16}, {16,16,16}, {16,16,16} }]" +
        " timepoints_per_partition=1" +
        " setups_per_partition=0" +
        " use_deflate_compression" +
        " export_path=" + h5_fused );

    }

print("  Finished: " + xml_path);

// quit after we are finished
eval("script", "System.exit(0);");
