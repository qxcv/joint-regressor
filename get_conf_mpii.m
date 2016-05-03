function conf = get_conf_mpii
%GET_CONF_MPII MPII-specific config (extends generic conf from get_conf)
conf = get_conf;

% Cache directory (per-dataset)
conf.cache_dir = 'cache/mpii-cooking/';
% Fully convolutional network definition for Keras
conf.cnn.deploy_json = fullfile(conf.cache_dir, 'cnn_model.json');
% Trained net weights
conf.cnn.deploy_weights = fullfile(conf.cache_dir, 'cnn_model.h5');
% Different for each dataset, I guess
conf.cnn.gpu = 1;

%% Augmentation stuff (this is 70x augmentation ATM; probably too much)

% Total number of augmentations is given by
%   length(conf.aug.rots) * length(conf.aug.flips)
%    * (sum(conf.aug.scales < 1) * conf.aug.randtrans
%       + sum(conf.aug.scales >= 1)),
% which doesn't count random translations on images which aren't sub-scale.

% Range of random rotations
conf.aug.rot_range = [-50, 50];
% Number of random rotations for each datum
conf.aug.rand_rots = 4;
% Random translations at each scale where it's possible to translate whilst
% keeping the pose in-frame; 0 to disable. Should probably pad all images
% by step size and then randomly translate by [step, step] (both axes)
% in training code; that should ensure that learnt biposelet clusters
% capture something interesting about pose structure.
conf.aug.rand_trans = 2;
% Choose a single flip type at random
% Values: 'random' (choose only one at random), 'both' (do both flips),
% 'none' (don't flip)
conf.aug.flip_mode = 'random'; % Other values: "both", "none"
% Include this many randomly cropped patches from the background for each
% datum (so no parts at all in the image)
conf.aug.easy_negs = 10;
% Also include this many challenging negatives for *each subpose* in each
% datum. Challenging negatives are those where the real subpose appears
% (partially) in the frame, but might will far enough off that it can't
% reasonably be assigned a type.
conf.aug.hard_negs = 2;

% Validation augmentations are less aggressive (24x instead)
conf.val_aug.rot_range = [-30, 30];
conf.val_aug.rand_rots = 2;
conf.val_aug.rand_trans = 1;
conf.val_aug.flip_mode = 'random';
conf.val_aug.easy_negs = 2;
conf.val_aug.hard_negs = 1;

%% General writing stuff

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
