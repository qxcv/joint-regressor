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

%% Augmentation stuff (this is 196x augmentation, last I checked)

% I think total number of augmentations is given by
%   length(conf.aug.rots) * length(conf.aug.flips)
%    * (sum(conf.aug.scales < 1) * conf.aug.randtrans
%       + sum(conf.aug.scales >= 1)),
% which doesn't count random translations on images which aren't sub-scale.

% Rotations for data augmentation (degrees from non-rotated)
conf.aug.rots = -30:10:30;
% Scales for data augmentation (2.0 = one quarter of a skeleton per frame, 0.5 = four skeletons per frame)
conf.aug.scales = [0.6, 0.75, 0.9, 1.0, 1.1];
% 4 random translations at each scale where it's possible to translate
% whilst keeping the pose in-frame.
conf.aug.randtrans = 4;
% Normal orientation plus one flip
conf.aug.flips = [0, 1];

%% Other training junk
% Each sample is 227 * 227 * 8 * 4 bytes (singles, 2*RGB layers, 1*flow
% layer), so we can fit ~650 of them per GiB. Round down to 512 samples
% so that we have a multiple of the batch size (256 by default).
conf.hdf5_samples = 512;