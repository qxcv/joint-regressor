% Like main_dual, but for MPII

startup;
conf = get_conf;
[mpii_data, train_pairs, val_pairs] = get_mpii_cooking(conf.dataset_dir, ...
                                                       conf.cache_dir);

fprintf('Writing validation set\n');
val_patch_dir = fullfile(conf.cache_dir, 'val-patches-mpii');
write_dset(mpii_data, val_pairs, conf.cache_dir, val_patch_dir, ...
    conf.num_val_hdf5s, conf.cnn.window, conf.poselet, conf.val_aug);

fprintf('Writing training set\n');
train_patch_dir = fullfile(conf.cache_dir, 'train-patches-mpii');
write_dset(mpii_data, train_pairs, conf.cache_dir, train_patch_dir, ...
    conf.num_hdf5s, conf.cnn.window, conf.poselet, conf.aug);

fprintf('Removing mean pixel\n');
adjust_for_mean_pixel(train_patch_dir, val_patch_dir, conf.cache_dir);