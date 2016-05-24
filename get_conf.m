function conf = get_conf
% Global configuration
%% Paths
% For data set code and data sets themselves
conf.dataset_dir = 'datasets/';
% For third-party deps
conf.ext_dir = 'ext/';

%% CNN-related props
% Size of CNN crop necessary
conf.cnn.window = [224 224];
% GPU ID for testing
conf.cnn.gpu = 2;
% lib.cnmem flag for Theano. I have this set to a really low value so that
% I can exploit cnmem's never-release-anything "feature", which stops other
% GPU users from staking out GPU memory in the few seconds in which it
% would normally be released between forward/back prop runs. Apparently low
% values can cause fragmentation, so I may have to change this back up
% later or disable cnmem entirely.
conf.cnn.cnmem = 0;
% Stride at which fully convolutional network slides over the input
conf.cnn.step = 32;

%% Inference stuff
% Scales used in each level of the feature pyramid (reversed so that
% biggest is at the beginning)
conf.pyra.scales = sort(1.12 .^ (-1:1), 'ascend');
% This defines the maximum size for the QP solver's support vector cache
% (in GiB).
conf.memsize = 0.5;
% Number of biposes (pose across two frames) to fetch for each frame pair
conf.num_dets = 10000;
% How many of those biposes to actually use during recombination
conf.num_stitch_dets = 300;
% Weights for stitching biposelet sequences. rscore is root score of
% biposelet, dist is L2 distance between neighbouring poses
conf.stitch_weights.rscore = 1;
conf.stitch_weights.dist = 0.005;

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
% ...and only one for validation.
conf.num_val_hdf5s = 1;

% HDF5 chunk sizes for training and validation, respectively. Training data
% is accessed randomly, so smaller is better. Validation data, on the other
% hand, is accessed sequentially, so long chunks are advantageous.
conf.train_chunksz = 4;
conf.val_chunksz = 4;

% Use K-means to cluster 2 * length(conf.poselet)-dimensional
% poselet-per-frame vectors, then use the resulting centroids as classes
% for biposelet prediction.
conf.biposelet_classes = 100;

% Multiply bounding box sides by this factor to get a CNN crop size
% (ensures that entire bounding box is in view)
conf.template_scale = 1.15;

%% Evaluation config
conf.pck_thresholds = 0:2.5:100;
