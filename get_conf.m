function conf = get_conf
%% Paths
% Caching flow/detections/Caffe models/whatever
conf.cache_dir = 'cache/';
% For data set code and data sets themselves
conf.dataset_dir = 'datasets/';
% For third-party deps
conf.ext_dir = 'ext/';

%% CNN-related props
% Size of CNN crop necessary
conf.cnn.window = [227 227];

%% Augmentation stuff
% Rotations for data augmentation (degrees from non-rotated)
conf.aug.rots = -35:5:35;
% Scales for data augmentation (2.0 = double-scale, 0.5 = half-scale)
conf.aug.scales = 2.^(-1:0.25:1);

%% Other training junk
% Each sample is 227 * 227 * 8 * 4 bytes (singles, 2*RGB layers, 1*flow
% layer), so we can fit ~650 of them per GiB. Round down to 512 samples
% so that we have a multiple of the batch size (256 by default).
conf.hdf5_samples = 512;