function conf = get_conf_mpii
conf = get_conf;

conf.num_hdf5s = 1;
% Number of hdf5s to use for validation
conf.num_val_hdf5s = 1;
% right_parts and left_parts are used to ensure that the meanings of "left"
% and "right" are preserved when doing flip augmentations.
conf.right_parts = [3, 5, 7, 9];
conf.left_parts = [4, 6, 8, 10];

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
subpose_indices = {[4 6 8 10], [3 5 7 9], [3 4 11 12]};
subpose_names = {'left', 'right', 'head'};
conf.subposes = struct('name', subpose_names, 'subpose', subpose_indices);
% Tells us which subpose is the parent of which (0 for root)
conf.subpose_pa = [3 3 0];
% subpose_shared_parts{c}
conf.shared_parts = {
    % Left side shoulder coordinates in the left arm subpose have indices 1
    % (first frame) and 5 (second frame), which correspond to indices 2
    % (first frame) and 6 (second frame) in the head subpose.
    {[1 5], [2 6]}
    % Right side shoulder coordinates in the right arm subpose have indices
    % 1 (first frame) and 5 (second frame), which correspond to indices 2
    % (first frame) and 5 (second frame) in the head subpose.
    {[1 5], [1 5]}
    % Don't worry about the head because it has no parents
    {}
 };