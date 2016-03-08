% Like main_dual, but for MPII

startup;
conf = get_conf_mpii;
[train_data, val_data, train_pairs, val_pairs] = get_mpii_cooking(...
    conf.dataset_dir, conf.cache_dir);
% INRIAPerson data is only used for training the graphical model; I used
% person-free crops of MPII cooking to train the CNN to recognise
% background.
[neg_data, neg_pairs] = get_inria_person(conf.dataset_dir, conf.cache_dir);

fprintf('Writing validation set\n');
val_patch_dir = fullfile(conf.cache_dir, 'val-patches-notrans');
write_dset(val_data, val_pairs, conf.cache_dir, val_patch_dir, ...
    conf.num_val_hdf5s, conf.cnn.window, conf.poselets, ...
    conf.left_parts, conf.right_parts, conf.val_aug, conf.val_chunksz);
write_negatives(val_data, val_pairs, conf.cache_dir, val_patch_dir, ...
    conf.cnn.window, conf.val_aug.negs, conf.val_chunksz, conf.poselets);

fprintf('Writing training set\n');
train_patch_dir = fullfile(conf.cache_dir, 'train-patches-notrans');
write_dset(train_data, train_pairs, conf.cache_dir, train_patch_dir, ...
    conf.num_hdf5s, conf.cnn.window, conf.poselets, ...
    conf.left_parts, conf.right_parts, conf.aug, conf.train_chunksz);
write_negatives(train_data, train_pairs, conf.cache_dir, train_patch_dir, ...
    conf.cnn.window, conf.aug.negs, conf.train_chunksz, conf.poselets);

fprintf('Writing cluster information\n');
cluster_h5s(conf.biposelet_classes, conf.poselets, train_patch_dir, ...
    val_patch_dir, conf.cache_dir);

fprintf('Calculating mean pixel\n');
store_mean_pixel(train_patch_dir, conf.cache_dir);

% TODO: Make training automatic. I can do this manually, but people who
% want to reproduce my results can't.
if ~(exist(conf.cnn.deploy_json, 'file') && exist(conf.cnn.deploy_weights, 'file'))
    error('jointregressor:nocnn', ...
        ['You need to run train.py to train a network, then use the ' ...
         'provided notebook to convert it to an FC net. This should ' ...
         'give you a model definition (%s) and a weights file (%s).'], ...
         conf.cnn.deploy_json, conf.cnn.deploy_weights);
end

fprintf('Computing ideal poselet displacements\n');
save_centroid_pairwise_means(...
    conf.cache_dir, conf.subpose_pa, shared_parts, conf.cnn_window);

fprintf('Training graphical model');
graphical_model = train_model(conf.cache_dir, conf.subpose_pa, val_data, ...
    neg_data, tsize);

% TODO: remember that I need to copy the evaluation code out of my own
% project, since that does the stitching thing properly (much easier here
% since I don't really need flow consistency).
%
% This is really tricky, since the training process for the graphical model
% requires working detection code as well (for hard negative mining, I
% think), which I will have to re-adapt to work with all of Chen & Yuille's
% parts.
%
% Since I'm not learning weights for recombination, I don't have to port
% the recombination code to work with C&Y's stuff. Just my own fully
% convolutional network code (in detect_fast?).