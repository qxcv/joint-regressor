% Like main_dual, but for MPII

startup;
conf = get_conf_mpii;
[train_data, val_data, train_pairs, val_pairs] = get_mpii_cooking(conf.dataset_dir, ...
                                                       conf.cache_dir);

fprintf('Writing validation set\n');
val_patch_dir = fullfile(conf.cache_dir, 'val-patches-mpii');
write_dset(val_data, val_pairs, conf.cache_dir, val_patch_dir, ...
    conf.num_val_hdf5s, conf.cnn.window, conf.poselet, ...
    conf.left_parts, conf.right_parts, conf.val_aug, conf.val_chunksz);
write_negatives(val_data, val_pairs, conf.cache_dir, val_patch_dir, ...
    conf.cnn.window, conf.val_aug.negs, 2 * 2 * numel(conf.poselet), ...
    conf.val_chunksz);

fprintf('Writing training set\n');
train_patch_dir = fullfile(conf.cache_dir, 'train-patches-mpii');
% write_dset(train_data, train_pairs, conf.cache_dir, train_patch_dir, ...
%     conf.num_hdf5s, conf.cnn.window, conf.poselet, ...
%     conf.left_parts, conf.right_parts, conf.aug, conf.train_chunksz);
write_negatives(train_data, train_pairs, conf.cache_dir, train_patch_dir, ...
    conf.cnn.window, conf.aug.negs, 2 * 2 * numel(conf.poselet), ...
    conf.train_chunksz);

fprintf('Writing cluster information\n');
cluster_h5s(conf.biposelet_classes, train_patch_dir, val_patch_dir);

fprintf('Calculating mean pixel\n');
store_mean_pixel(train_patch_dir, conf.cache_dir);