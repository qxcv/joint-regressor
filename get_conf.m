function conf = get_conf
% Global configuration
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
% GPU ID for testing
conf.cnn.gpu = 2;
% lib.cnmem flag for Theano
conf.cnn.cnmem = 0.5;
% Stride at which fully convolutional network slides over the input
conf.cnn.step = 32;

%% Inference stuff
% Scales used in each level of the feature pyramid (reversed so that
% biggest is at the beginning)
conf.pyra.scales = sort(1.12 .^ (-3:3), 'ascend');
% This defines the maximum size for the QP solver's support vector cache
% (in GiB).
conf.memsize = 0.5;
% Number of biposes to fetch for each frame pair
conf.num_dets = 300;
% Weights for stitching biposelet sequences. rscore is root score of
% biposelet, dist is L2 distance between neighbouring poses
conf.stitch_weights.rscore = 1;
conf.stitch_weights.dist = 1;

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
% Random translations at each scale where it's possible to translate whilst
% keeping the pose in-frame; 0 to disable. Should probably pad all images
% by step size and then randomly translate by [-step/2, step/2] (both axes)
% in training code; that should ensure that learnt biposelet clusters
% capture something interesting about pose structure.
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

% Use K-means to cluster 2 * length(conf.poselet)-dimensional
% poselet-per-frame vectors, then use the resulting centroids as classes
% for biposelet prediction.
conf.biposelet_classes = 100;

% Multiply bounding box sides by this factor to get a CNN crop size
% (ensures that entire bounding box is in view)
conf.template_scale = 1.15;

%% Evaluation config
conf.pck_thresholds = 0:2.5:100;
