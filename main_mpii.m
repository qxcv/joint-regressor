% Like main_dual, but for MPII

startup;
conf = get_conf_mpii;
[train_dataset, val_dataset] = get_mpii_cooking(conf.dataset_dir, ...
    conf.cache_dir, conf.pair_mean_dist_thresh);
% INRIAPerson data is only used for training the graphical model; I used
% person-free crops of MPII cooking to train the CNN to recognise
% background.
neg_dataset = get_inria_person(conf.dataset_dir, conf.cache_dir);

% TODO: Should derive scale from both train dataset and validation dataset
% at same time
[train_dataset, ~] = mark_scales(train_dataset, conf.subposes, ...
    conf.cnn.step, conf.template_scale);
[val_dataset, tsize] = mark_scales(val_dataset, conf.subposes, ...
    conf.cnn.step, conf.template_scale, [train_dataset.pairs.scale]);

assert(false, 'Everything below here is broken');

fprintf('Writing validation set\n');
val_patch_dir = fullfile(conf.cache_dir, 'val-patches-mpii');
write_dset(val_dataset, conf.cache_dir, val_patch_dir, ...
    conf.num_val_hdf5s, conf.cnn.window, conf.subposes, ...
    conf.left_parts, conf.right_parts, conf.val_aug, conf.val_chunksz);
write_negatives(val_dataset, conf.cache_dir, val_patch_dir, ...
    conf.cnn.window, conf.val_aug.negs, conf.val_chunksz, conf.subposes);

fprintf('Writing training set\n');
train_patch_dir = fullfile(conf.cache_dir, 'train-patches-mpii');
write_dset(train_dataset, conf.cache_dir, train_patch_dir, ...
    conf.num_hdf5s, conf.cnn.window, conf.subposes, ...
    conf.left_parts, conf.right_parts, conf.aug, conf.train_chunksz);
write_negatives(train_dataset, conf.cache_dir, train_patch_dir, ...
    conf.cnn.window, conf.aug.negs, conf.train_chunksz, conf.subposes);

fprintf('Writing cluster information\n');
cluster_h5s(conf.biposelet_classes, conf.subposes, train_patch_dir, ...
    val_patch_dir, conf.cache_dir);

fprintf('Calculating mean pixel\n');
store_mean_pixel(train_patch_dir, conf.cache_dir);

% TODO: Make training automatic. I can do this manually, but people who
% want to reproduce my results can't. This should just be a call to
% train.py with the appropriate arguments, although there will be some
% additional effort involved in activating the virtualenv.
if ~(exist(conf.cnn.deploy_json, 'file') && exist(conf.cnn.deploy_weights, 'file'))
    error('jointregressor:nocnn', ...
        ['You need to run train.py to train a network, then use the ' ...
         'provided notebook to convert it to an FC net. This should ' ...
         'give you a model definition (%s) and a weights file (%s).'], ...
         conf.cnn.deploy_json, conf.cnn.deploy_weights);
end

fprintf('Computing ideal poselet displacements\n');
subpose_disps = save_centroid_pairwise_means(...
    conf.cache_dir, conf.subpose_pa, conf.shared_parts);

fprintf('Training graphical model\n');
% XXX: This is woefully unscientific and needs to be changed as soon as I
% can figure out a uniform-scale training protocol
[~] = train_model(conf, val_dataset, neg_dataset, subpose_disps, tsize);

assert(false, 'You need to write the rest of this');

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