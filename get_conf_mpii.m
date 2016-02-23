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
poselet_names = {'left', 'right', 'head'};
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
poselet_indices = {[4 6 8 10], [3 5 7 9], [3 4 11 12]};
conf.poselets = struct('name', poselet_names, 'poselet', poselet_indices);