function conf = get_conf_h36m_upper
%GET_CONF_H36M_UPPER Upper-body-only Human3.6M-specific config
conf = get_conf;

% Directory for temporary files
conf.cache_dir = 'cache/h36m-upper/';
% Directory for final results (for accum_stats)
conf.results_dir = 'results/h36m-upper/';
% Fully convolutional network definition for Keras
conf.cnn.deploy_json = fullfile(conf.cache_dir, 'cnn_model.json');
% Trained net weights (fully convolutional)
conf.cnn.deploy_weights = fullfile(conf.cache_dir, 'cnn_model.h5');
% Use different GPU
conf.cnn.gpu = 2;

conf.h36m_keep_frac = 0.15;
% How many test sequences should we choose? There is something like 120
% sequences normally, so we need to randomly choose a subset of them.
conf.h36m_test.seqs = 60;
% How many frames should we trim each test sequence to? Subsequence will be
% randomly chosen for each sequence modulo this constraint.
conf.h36m_test.seq_size = 15;
% Random seed used for test sequence selection
conf.h36m_test.seq_seed = 42;

%% STUFF FOR MULTI-POSELET CODE BELOW HERE
% Actual skeleton, adapted from 32-point H3.6M skeleton
conf.trans_spec = struct(...
    ... .indices indicates which source joints should be combined to produce
    ... each new "super joint"
    'indices', {...
        ... MIDDLE:
        [18 26], ... Mid-shoulders          #1
        ... LEFT SIDE:
        18, ... Left shoulder               #2
        [19 18], ... Left upper arm         #3
        19, ... Left elbow                  #4
        [20 19], ... Left forearm           #5
        20, ... Left wrist                  #6
        ... RIGHT SIDE:
        26, ... Right shoulder              #7
        [27 26], ... Right upper arm        #8
        27, ... Right elbow                 #9
        [28 27], ... Right forearm          #10
        28, ... Right wrist                 #11
    }, ...
    ... .weights specifies the proportion in which each source joint should
    ... be used.
    'weights', {...
        [1/2 1/2], ... Mid-shoulders        #1
        1.0, ... Left shoulder              #2
        [2/3 1/3], ... Left upper arm       #3
        1.0, ... Left elbow                 #4
        [1/3 2/3], ... Left forearm         #5
        1.0, ... Left wrist                 #6
        1.0, ... Right shoulder             #7
        [2/3 1/3], ... Right upper arm      #8
        1.0, ... Right elbow                #9
        [1/3 2/3], ... Right forearm        #10
        1.0, ... Right wrist                #11
    });
conf.test_trans_spec = conf.trans_spec;

% Rest of this is basically the same as MPII (or the same as MPII used to be)
conf.right_parts = 7:11;
conf.left_parts = 2:6;
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
conf.limbs = struct(...
    'indices', {[7 9],   [9 11],  [2 4],   [4 6],   [2 7]}, ...
    'names',   {'ruarm', 'rfarm', 'luarm', 'lfarm', 'shoul'});
conf.limb_combinations = containers.Map(...
    {'uarm', 'farm', 'shoul'}, ...
    {{'ruarm', 'luarm'}, {'rfarm', 'lfarm'}, {'shoul'}}...
);

% conf.stitching.* holds parameters for producing sequence of poses from
% per-pair pose sets
conf.stitching.app_weights = ones([15 1]);
conf.stitching.dist_weights = ones([15 1]);
% Grab the best poses_per_pair biposelets from each pair of frames for
% stitching. Remember stitching is O(poses_per_pair^2), so this can't be
% too large.
conf.stitching.poses_per_pair = 100;
    
%% Augmentation stuff
conf.aug.rot_range = [-30, 30];
conf.aug.rand_rots = 1;
conf.aug.rand_trans = 1;
conf.aug.flip_mode = 'random'; % Other values: "both", "none"
conf.aug.easy_negs = 3;
conf.aug.hard_negs = 3;
conf.aug.inria_negs = 0;
conf.val_aug.rot_range = [-15, 15];
conf.val_aug.rand_rots = 1;
conf.val_aug.rand_trans = 0;
conf.val_aug.flip_mode = 'random';
conf.val_aug.easy_negs = 2;
conf.val_aug.hard_negs = 2;
conf.val_aug.inria_negs = 0;
