function conf = get_conf_flic_piw
%GET_CONF_FLIC_PIW Configuration for FLIC and PIW
conf = get_conf;

% Cache directory (per-dataset)
conf.cache_dir = 'cache/flic-piw/';
% Directory for final results (for accum_stats)
conf.results_dir = 'results/piw/';
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
conf.aug.rot_range = [-30, 30];
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
conf.aug.easy_negs = 0;
% Also include this many challenging negatives for *each subpose* in each
% datum. Challenging negatives are those where the real subpose appears
% (partially) in the frame, but might will far enough off that it can't
% reasonably be assigned a type.
conf.aug.hard_negs = 10;
% We need to use inria_negs (number of negatives to take from each INRIA
% frame) instead of easy_negs because easy_negs might catch unlabelled
% people on FLIC-full
conf.aug.inria_negs = 20;

% Validation augmentations are less aggressive (24x instead)
conf.val_aug.rot_range = [-30, 30];
conf.val_aug.rand_rots = 2;
conf.val_aug.rand_trans = 1;
conf.val_aug.flip_mode = 'random';
conf.val_aug.easy_negs = 0;
conf.val_aug.hard_negs = 5;
conf.val_aug.inria_negs = 10;

%% General writing stuff
conf.num_hdf5s = 1;
% Number of hdf5s to use for validation
conf.num_val_hdf5s = 1;

%% STUFF FOR MULTI-POSELET CODE BELOW HERE
% We need to transformation specs: one for FLIC and one for PIW. The result
% is an identical skeleton for each dataset.
conf.flic_trans_spec = struct(...
    'indices', {...
        ... MIDDLE:
        [1 4], ... Mid-shoulders            #1
        ... LEFT:
        1,     ... Left shoulder            #2
        [1 2], ... Left upper arm           #3
        2,     ... Left elbow               #4
        [2 3], ... Left forearm             #5
        3,     ... Left wrist               #6
        ... RIGHT:
        4,     ... Right shoulder           #7
        [4 5], ... Right upper arm          #8
        5,     ... Right elbow              #9
        [5 6], ... Right forearm            #10
        6,     ... Right wrist              #11
    }, ...
    'weights', {...
        [1/2 1/2], ... Mid-shoulders        #1
        1,         ... Left shoulder        #2
        [1/3 2/3], ... Left upper arm       #3
        1,         ... Left elbow           #4
        [2/3 1/3], ... Left forearm         #5
        1,         ... Left wrist           #6
        1,         ... Right shoulder       #7
        [1/3 2/3], ... Right upper arm      #8
        1,         ... Right elbow          #9
        [2/3 1/3], ... Right forearm        #10
        1,         ... Right wrist          #11
    });
conf.piw_trans_spec = struct(...
    'indices', {...
        ... MIDDLE:
        [2 5], ... Mid-shoulders            #1
        ... LEFT:
        2,     ... Left shoulder            #2
        [2 3], ... Left upper arm           #3
        3,     ... Left elbow               #4
        [3 4], ... Left forearm             #5
        4,     ... Left wrist               #6
        ... RIGHT:
        5,     ... Right shoulder           #7
        [5 6], ... Right upper arm          #8
        6,     ... Right elbow              #9
        [6 7], ... Right forearm            #10
        7,     ... Right wrist              #11
    }, ...
    'weights', {...
        [1/2 1/2], ... Mid-shoulders        #1
        1,         ... Left shoulder        #2
        [1/3 2/3], ... Left upper arm       #3
        1,         ... Left elbow           #4
        [2/3 1/3], ... Left forearm         #5
        1,         ... Left wrist           #6
        1,         ... Right shoulder       #7
        [1/3 2/3], ... Right upper arm      #8
        1,         ... Right elbow          #9
        [2/3 1/3], ... Right forearm        #10
        1,         ... Right wrist          #11
    });
% Used for reconstructing original joint positions
conf.test_trans_spec = conf.piw_trans_spec;

% right_parts and left_parts are used to ensure that the meanings of "left"
% and "right" are preserved when doing flip augmentations.
conf.right_parts = 7:11;
conf.left_parts = 2:6;
% Number of joints in the model; we don't use all of these
conf.num_joints = 11;

subpose_indices = {[2 1 7], ...
    [2 3], [3 4 5], [5 6], ...
    [7 8], [8 9 10], [10 11]};
subpose_names = {'shols', ... 1 Mid
    ...  2       3        4
    'luarm', 'lelb', 'lfarm', ... Left
    ...  5       6        7
    'ruarm', 'relb', 'rfarm'}; % Right
conf.subposes = struct('name', subpose_names, 'subpose', subpose_indices);
conf.valid_parts = unique([subpose_indices{:}]);
conf.subpose_pa = [0 ... Mid
    1 2 3 ... Left
    1 5 6]; % Right
conf.shared_parts = make_shared_parts(conf.subposes, conf.subpose_pa);

% List of limbs for PCP calculation
conf.limbs = struct(...
    'indices', {[7 9],   [9 11],  [2 4],   [4 6],   [2 7]}, ...
    'names',   {'ruarm', 'rfarm', 'luarm', 'lfarm', 'shoul'});
conf.limb_combinations = containers.Map(...
    {'uarm', 'farm', 'shoul'}, ...
    {{'ruarm', 'luarm'}, {'rfarm', 'lfarm'}, {'shoul'}}...
);
