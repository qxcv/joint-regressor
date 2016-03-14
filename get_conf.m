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
conf.cnn.window = [224 224];
% Fully convolutional network definition for Keras
conf.cnn.deploy_json = fullfile(conf.cache_dir, 'cnn_model.json');
% Trained net weights
conf.cnn.deploy_weights = fullfile(conf.cache_dir, 'cnn_model.h5');
% GPU ID for testing (negative to disable)
conf.cnn.gpu_id = 0;
% Stride at which fully convolutional network slides over the input
% XXX: Need to confirm that my step actually is 32!
conf.cnn.step = 32;

%% Inference stuff
% Levels of the feature pyramid in each octave. Might have to turn this
% down, since 10 is insane.
conf.interval = 10;

%% Augmentation stuff (this is 70x augmentation ATM; probably too much)

% Total number of augmentations is given by
%   length(conf.aug.rots) * length(conf.aug.flips)
%    * (sum(conf.aug.scales < 1) * conf.aug.randtrans
%       + sum(conf.aug.scales >= 1)),
% which doesn't count random translations on images which aren't sub-scale.

% Range of random rotations
conf.aug.rot_range = [-50, 50];
% Number of random rotations for each datum
conf.aug.rand_rots = 5;
% Scales for data augmentation (2.0 = one quarter of a skeleton per frame, 0.5 = four skeletons per frame)
conf.aug.scales = [0.8, 0.9];
% Random translations at each scale where it's possible to translate whilst
% keeping the pose in-frame; 0 to disable.
conf.aug.rand_trans = 0;
% Choose a single flip type at random
% Values: 'random' (choose only one at random), 'both' (do both flips),
% 'none' (don't flip)
conf.aug.flip_mode = 'random'; % Other values: "both", "none"
% Include 30 randomly cropped negative samples for each datum
conf.aug.negs = 30;

% Validation augmentations are less aggressive (24x instead)
conf.val_aug.rot_range = [-30, 30];
conf.val_aug.rand_rots = 2;
conf.val_aug.scales = [0.75 0.8];
conf.val_aug.rand_trans = 0;
conf.val_aug.flip_mode = 'random';
conf.val_aug.negs = 8;

%% Other training junk
% How many HDF5 files should we split our data set across? When writing out
% samples, a HDF5 file will be chosen at random and written to (this will
% work out in the long run).
% NOTE: I changed this down to 1 so that the resultant data would be easier
% to work with in Keras. This only works because h5py is smart enough to
% lazily load data from disk; if you try this with Caffe, which naively
% attempts to load *entire datasets* from disk, you will end up crashing
% the program.
conf.num_hdf5s = 1;

% HDF5 chunk sizes for training and validation, respectively. Training data
% is accessed randomly, so smaller is better. Validation data, on the other
% hand, is accessed sequentially, so long chunks are advantageous.
conf.train_chunksz = 4;
conf.val_chunksz = 4;

% Number of hdf5s to use for validation
conf.num_val_hdf5s = 1;

% Fraction of pairs to use for validation (XXX is this used?)
conf.val_pairs_frac = 0.2;

% Use K-means to cluster 2 * length(conf.poselet)-dimensional
% poselet-per-frame vectors, then use the resulting centroids as classes
% for biposelet prediction.
conf.biposelet_classes = 100;