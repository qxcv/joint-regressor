function conf = get_conf_mpii
%GET_CONF_MPII MPII-specific config (extends generic conf from get_conf)
conf = get_conf;

% Cache directory (per-dataset)
conf.cache_dir = 'cache/mpii-cooking/';
% Fully convolutional network definition for Keras
conf.cnn.deploy_json = fullfile(conf.cache_dir, 'cnn_model.json');
% Trained net weights
conf.cnn.deploy_weights = fullfile(conf.cache_dir, 'cnn_model.h5');

conf.num_hdf5s = 1;
% Number of hdf5s to use for validation
conf.num_val_hdf5s = 1;
% right_parts and left_parts are used to ensure that the meanings of "left"
% and "right" are preserved when doing flip augmentations.
conf.right_parts = [3, 5, 7, 9];
conf.left_parts = [4, 6, 8, 10];
% Number of joints in the model; we don't use all of these
conf.num_joints = 12;

%% STUFF FOR MULTI-POSELET CODE BELOW HERE
% Rough guide for MPII
% 3  -> right shoulder
% 4  -> left shoulder
% 5  -> right elbow
% 6  -> left elbow
% 7  -> right wrist
% 8  -> left wrist
% 9  -> right hand
% 10 -> left hand
% 11 -> head upper point
% 12 -> head lower point
% This means that left and right are shoulder->elbow->wrist->hand, and head
% is left shoulder->right shoulder->upper head->lower head.
subpose_indices = {[3 4 11 12], [4 6 8 10], [3 5 7 9]};
subpose_names = {'head', 'left', 'right'};
conf.subposes = struct('name', subpose_names, 'subpose', subpose_indices);
conf.valid_parts = unique([subpose_indices{:}]);
% Tells us which subpose is the parent of which (0 for root)
conf.subpose_pa = [0 1 1];
% shared_parts{c} is a two-element cell array in which the first element is
% a vector naming parts from the biposelet associated with subpose c and
% the second element is a vector naming equivalent parts from the bipose
% associated with its parent. The fact that we're dealing with biposes
% rather than just subposes means that some indices will be greater than
% the number of joints in a subpose.
conf.shared_parts = make_shared_parts(conf.subposes, conf.subpose_pa);

% Throw out pairs for which the mean distance between corresponding joints
% (between the two frames) is beyond this pixel threshold. Empirically, this
% retains ~99% of the data. The rest are probably mislabelled or
% incorrectly classified as being in the same scene.
conf.pair_mean_dist_thresh = 50;

% List of limbs for PCP calculation
conf.limbs = struct(...
    'indices', {[3 5],   [5 7],   [7 9],   [4 6],   [6 8],   [8 10],  [11 12]}, ...
    'names',   {'ruarm', 'rfarm', 'rhand', 'luarm', 'lfarm', 'lhand', 'face'});
conf.limb_combinations = containers.Map(...
    {'uarm', 'farm', 'hand', 'face'}, ...
    {{'ruarm', 'luarm'}, {'rfarm', 'lfarm'}, {'rhand', 'lhand'}, {'face'}}...
);
