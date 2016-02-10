function conf = get_conf_mpii
conf = get_conf;

conf.num_hdf5s = 1;
% Number of hdf5s to use for validation
conf.num_val_hdf5s = 1;
% Use only parts with these indices
conf.poselet = [4 6 8]; % That's MPII left side
conf.right_parts = [3, 5, 7, 9];
conf.left_parts = [4, 6, 8, 10];