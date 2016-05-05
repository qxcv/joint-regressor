function conf = get_conf_h36m
%GET_CONF_H36M Human3.6M-specific config (extends get_conf)
conf = get_conf;

% Directory for temporary files
conf.cache_dir = 'cache/h36m/';
% Fully convolutional network definition for Keras
conf.cnn.deploy_json = fullfile(conf.cache_dir, 'cnn_model.json');
% Trained net weights (fully convolutional)
conf.cnn.deploy_weights = fullfile(conf.cache_dir, 'cnn_model.h5');
% Use different GPU
conf.cnn.gpu = 2;

% right_parts and left_parts are used to ensure that the meanings of "left"
% and "right" are preserved when doing flip augmentations.
conf.right_parts = 2:12;
conf.left_parts = 13:23;
% Number of joints in the model; we don't use all of these
conf.num_joints = 23;

%% STUFF FOR MULTI-POSELET CODE BELOW HERE
% Actual skeleton, adapted from 32-point H3.6M skeleton
conf.trans_spec = struct(...
    ... .indices indicates which source joints should be combined to produce
    ... each new "super joint"
    'indices', {...
        ... MIDDLE:
        [18 26], ... Mid-shoulders          #1
        ... RIGHT SIDE:
        28, ... Right wrist                 #2
        [28 27], ... Right forearm          #3
        27, ... Right elbow                 #4
        [27 26], ... Right upper arm        #5
        26, ... Right shoulder              #6
        [26 2], ... Right lower-torso       #7
        2, ... Right hip                    #8
        [2 3], ... Right mid-upper-leg      #9
        3, ... Right knee                   #10
        [3 4], ... Right below-knee         #11
        4, ... Right ankle                  #12
        ... LEFT SIDE:
        20, ... Left wrist                  #13
        [20 19], ... Left forearm           #14
        19, ... Left elbow                  #15
        [19 18], ... Left upper arm         #16
        18, ... Left shoulder               #17
        [18 7], ... Left lower-torso        #18
        7, ... Left hip                     #19
        [7 8], ... Left mid-upper-leg       #20
        8, ... Left knee                    #21
        [8 9], ... Left below-knee          #22
        9, ... Left ankle                   #23
    }, ...
    ... .weights specifies the proportion in which each source joint should
    ... be used.
    'weights', {...
        [1/2 1/2], ... Mid-shoulders        #1
        1.0, ... Right wrist                #2
        [1/3 2/3], ... Right forearm        #3
        1.0, ... Right elbow                #4
        [2/3 1/3], ... Right upper arm      #5
        1.0, ... Right shoulder             #6
        [1/4 3/4], ... Right lower-torso    #7
        1.0, ... Right hip                  #8
        [1/2 1/2], ... Right mid-upper-leg  #9
        1.0, ... Right knee                 #10
        [3/4 1/4], ... Right below-knee     #11
        1.0, ... Right ankle                #12
        1.0, ... Left wrist                 #13
        [1/3 2/3], ... Left forearm         #14
        1.0, ... Left elbow                 #15
        [2/3 1/3], ... Left upper arm       #16
        1.0, ... Left shoulder              #17
        [1/4 3/4], ... Left lower-torso     #18
        1.0, ... Left hip                   #19
        [1/2 1/2], ... Left mid-upper-leg   #20
        1.0, ... Left knee                  #21
        [3/4 1/4], ... Left below-knee      #22
        1.0, ... Left ankle                 #23
    });

% Subposes. llarm = left lower arm, lutor = left upper torso, etc.
% TODO: Figure out how to fit the head in somewhere. It's probably a really
% useful landmark.
% Indices:       1
subpose_names = {'shoul', ...
    ...  2   3           4        5        6           7           8
    'ruarm', 'rmarm',    'rlarm', 'rutor', 'rltor',    'ruleg',    'rlleg', ... Right side
    ...  9   10          11       12       13          14          15
    'luarm', 'lmarm',    'llarm', 'lutor', 'lltor',    'luleg',    'llleg', ... Left side
};
% Joints which make up each subpose
subpose_indices = {[6 1 17], ...
    [6 5],   [5 4 3],    [3 2],   [6 7],   [7 8 9],    [9 10 11],  [11 12], ...
    [17 16], [16 15 14], [14 13], [17 18], [18 19 20], [20 21 22], [22 23]};
% Tells us which subpose is the parent of which (0 for root)
conf.subpose_pa = [0 ...
    1        2           3        1         5          6           7 ...
    1        9          10        1        12         13          14];
conf.subposes = struct('name', subpose_names, 'subpose', subpose_indices);
conf.valid_parts = unique([subpose_indices{:}]);
% See get_conf_mpii for .shared_parts explanation. tl;dr it identifies
% joints shared by different subposes, which is good for finding ideal
% displacements.
conf.shared_parts = make_shared_parts(conf.subposes, conf.subpose_pa);

% conf.stitching.* holds parameters for producing sequence of poses from
% per-pair pose sets
conf.stitching.app_weights = ones([15 1]);
conf.stitching.dist_weights = ones([15 1]);
% Grab the best poses_per_pair biposelets from each pair of frames for
% stitching. Remember stitching is O(poses_per_pair^2), so this can't be
% too large.
conf.stitching.poses_per_pair = 100;

% List of limbs for PCP calculation
conf.limbs = struct(...
    'indices', {[6 17], ...
        [2 4],   [4 6],   [6 8],   [8 10],  [10 12], ...
        [13 15], [15 17], [17 19], [19 21], [21 23] ...
     }, ...
    'names',   {'shols', ...
        'rlarm', 'ruarm', 'rtor', 'ruleg', 'rlleg', ...
        'llarm', 'luarm', 'ltor', 'luleg', 'llleg'});
    
%% Augmentation stuff
conf.aug.rot_range = [-15, 15];
conf.aug.rand_rots = 1;
conf.aug.rand_trans = 0;
conf.aug.flip_mode = 'random'; % Other values: "both", "none"
conf.aug.negs = 2;
conf.val_aug.rot_range = [-15, 15];
conf.val_aug.rand_rots = 1;
conf.val_aug.rand_trans = 0;
conf.val_aug.flip_mode = 'random';
conf.val_aug.negs = 1;
