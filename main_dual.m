% Run this, sit back and watch the blinkenlights.

startup;
conf = get_conf;
[flic_data, train_pairs, val_pairs] = get_flic(conf.dataset_dir, ...
                                               conf.cache_dir);

fprintf('Writing validation set\n');
val_patch_dir = fullfile(conf.cache_dir, 'val-patches');
write_dset(flic_data, val_pairs, conf.cache_dir, val_patch_dir, ...
    conf.num_val_hdf5s, conf.cnn.window, conf.poselet, conf.val_aug);

fprintf('Writing training set\n');
train_patch_dir = fullfile(conf.cache_dir, 'train-patches');
write_dset(flic_data, train_pairs, conf.cache_dir, train_patch_dir, ...
    conf.num_hdf5s, conf.cnn.window, conf.poselet, conf.aug);

% Get biposelet clusters using training data
% cluster_path = fullfile(conf.cache_dir, 'centroids.mat');
% if ~exist(cluster_path, 'file')
%     % centroids = cluster_biposelets(flic_data, train_pairs, conf.biposelet_classes);
%     centroids = [];
%     save(cluster_path, 'centroids');
% else
%     load(cluster_path, 'centroids');
% end