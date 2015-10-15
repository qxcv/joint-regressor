% Run this, sit back and watch the blinkenlights.

function demo

startup;
conf = get_conf;
[flic_data, pairs] = get_flic(conf.dataset_dir, conf.cache_dir);

pairs_cache = fullfile(conf.cache_dir, 'pairs.mat');
try
    loaded_pairs = load(pairs_cache);
    train_pairs = loaded_pairs.train_pairs;
    val_pairs = loaded_pairs.val_pairs;
catch
    val_num = round(size(pairs, 1) * conf.val_pairs_frac);
    val_pairs = pairs(1:val_num);
    train_pairs = pairs(val_num+1:end);
    save(pairs_cache, 'train_pairs', 'val_pairs');
end

fprintf('Writing validation set\n');
val_patch_dir = fullfile(conf.cache_dir, 'val-patches');
write_dset(flic_data, val_pairs, conf.cache_dir, conf.num_val_hdf5s, ...
    conf.val_aug, val_patch_dir);
fprintf('Writing training set\n');
train_patch_dir = fullfile(conf.cache_dir, 'train-patches');
write_dset(flic_data, train_pairs, conf.cache_dir, conf.num_hdf5s, ...
    conf.aug, train_patch_dir);