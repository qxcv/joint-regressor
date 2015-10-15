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

%% Augmentation stuff (this is 70x augmentation ATM; probably too much)

% Total number of augmentations is given by
%   length(conf.aug.rots) * length(conf.aug.flips)
%    * (sum(conf.aug.scales < 1) * conf.aug.randtrans
%       + sum(conf.aug.scales >= 1)),
% which doesn't count random translations on images which aren't sub-scale.

% Rotations for data augmentation (degrees from non-rotated)
conf.aug.rots = -20:10:20;
% Scales for data augmentation (2.0 = one quarter of a skeleton per frame, 0.5 = four skeletons per frame)
conf.aug.scales = [0.6, 0.85, 1.0];
% 3 random translations at each scale where it's possible to translate
% whilst keeping the pose in-frame.
conf.aug.randtrans = 3;
% Normal orientation plus one flip
conf.aug.flips = [0, 1];

%% Other training junk
% How many HDF5 files should we split our data set across? When writing out
% samples, a HDF5 file will be chosen at random and written to (this will
% work out in the long run).
conf.num_hdf5s = 100;