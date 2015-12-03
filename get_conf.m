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
% Deploy prototxt
conf.cnn.deploy_prototxt = 'models/deploy.prototxt';
% Trained net
conf.cnn.model = fullfile(conf.cache_dir, 'regressor.caffemodel');
% GPU ID for testing (negative to disable)
conf.cnn.gpu_id = 0;

%% Augmentation stuff (this is 70x augmentation ATM; probably too much)

% Total number of augmentations is given by
%   length(conf.aug.rots) * length(conf.aug.flips)
%    * (sum(conf.aug.scales < 1) * conf.aug.randtrans
%       + sum(conf.aug.scales >= 1)),
% which doesn't count random translations on images which aren't sub-scale.

% Rotations for data augmentation (degrees from non-rotated)
conf.aug.rots = -30:15:30;
% Scales for data augmentation (2.0 = one quarter of a skeleton per frame, 0.5 = four skeletons per frame)
conf.aug.scales = [0.8, 0.85, 0.9];
% 3 random translations at each scale where it's possible to translate
% whilst keeping the pose in-frame.
conf.aug.randtrans = 3;
% Normal orientation plus one flip
conf.aug.flips = [0, 1];

% Validation augmentations are basically nonexistent (scaling is just to
% keep the pose in a box)
conf.val_aug.rots = 0;
conf.val_aug.scales = 0.8;
conf.val_aug.randtrans = 0;
conf.val_aug.flips = 0;

%% Other training junk
% How many HDF5 files should we split our data set across? When writing out
% samples, a HDF5 file will be chosen at random and written to (this will
% work out in the long run).
conf.num_hdf5s = 100;
% Number of hdf5s to use for validation
conf.num_val_hdf5s = 4;
% Fraction of pairs to use for validation
conf.val_pairs_frac = 0.2;