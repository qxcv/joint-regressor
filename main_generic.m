function main_generic(conf, train_dataset, val_dataset, test_seqs)
%MAIN_GENERIC Called by main_{h36m,mpii} to sequence most of the work.
% Don't run this directly; run main_h36m or main_mpii instead if you just
% want to test the code.
sizes_okay = @(ds) all(cellfun(@length, {ds.data.joint_locs}) == conf.num_joints);
assert(sizes_okay(train_dataset) && sizes_okay(val_dataset) && sizes_okay(test_seqs));
neg_dataset = get_inria_person(conf.dataset_dir, conf.cache_dir);

fprintf('Setting GPU for OpenCV\n');
% This is ridiculous, but I guess this is what it takes to run something on
% every worker and the client.
gpu_run_cmd = sprintf('setOpenCVGPU(%i);', conf.cnn.gpu);
fprintf('Command: %s\n', gpu_run_cmd);
gcp;
pctRunOnAll(gpu_run_cmd);

fprintf('Writing validation set\n');
val_patch_dir = fullfile(conf.cache_dir, 'val-patches');
write_dset(val_dataset, val_patch_dir, conf.num_val_hdf5s, ...
    conf.cnn.window, conf.cnn.step, conf.subposes, conf.left_parts, ...
    conf.right_parts, conf.val_aug, conf.val_chunksz);

fprintf('Writing training set\n');
train_patch_dir = fullfile(conf.cache_dir, 'train-patches');
write_dset(train_dataset, train_patch_dir, conf.num_hdf5s, ...
    conf.cnn.window, conf.cnn.step, conf.subposes, conf.left_parts, ...
    conf.right_parts, conf.aug, conf.train_chunksz);

fprintf('Writing cluster information\n');
biposelets = cluster_h5s(conf.biposelet_classes, conf.subposes, ...
    train_patch_dir, val_patch_dir, conf.cache_dir);

fprintf('Writing training negatives (from training set)\n');
write_negatives(train_dataset, biposelets, train_patch_dir, ...
    conf.cnn.window, conf.aug, conf.train_chunksz, conf.subposes, ...
    'negatives.h5');
fprintf('Writing training negatives (from INRIA)\n');
write_negatives(neg_dataset, biposelets, train_patch_dir, ...
    conf.cnn.window, conf.aug, conf.train_chunksz, conf.subposes, ...
    'inria-negatives.h5');

fprintf('Writing validation negatives (from validation set)\n');
write_negatives(val_dataset, biposelets, val_patch_dir, ...
    conf.cnn.window, conf.val_aug, conf.val_chunksz, conf.subposes, ...
    'negatives.h5');
fprintf('Writing validation negatives (from INRIA)\n');
write_negatives(neg_dataset, biposelets, val_patch_dir, ...
    conf.cnn.window, conf.val_aug, conf.val_chunksz, conf.subposes, ...
    'inria-negatives.h5');

% We need to re-write clusters (poselets) so that the negatives have
% poselets too (obviously those will be "null poselets" in a sense, but
% poselets all the same).
fprintf('Re-writing cluster information\n');
biposelets = cluster_h5s(conf.biposelet_classes, conf.subposes, ...
    train_patch_dir, val_patch_dir, conf.cache_dir);

fprintf('Calculating mean pixel\n');
store_mean_pixel(train_patch_dir, conf.cache_dir);

fprintf('Training CNN\n');
train_h5s = files_with_extension(train_patch_dir, '.h5');
val_h5s = files_with_extension(train_patch_dir, '.h5');
cnn_train(conf.cnn, conf.cache_dir, train_h5s, val_h5s);

fprintf('Computing ideal poselet displacements\n');
subpose_disps = save_centroid_pairwise_means(...
    conf.cache_dir, conf.subpose_pa, conf.shared_parts);

fprintf('Caching flow for positive validation pairs\n');
cache_all_flow(val_dataset, conf.cache_dir);
fprintf('Caching flow for negative pairs\n');
cache_all_flow(neg_dataset, conf.cache_dir);

fprintf('Training graphical model\n');
ssvm_model = train_model(conf, val_dataset, neg_dataset, subpose_disps);

fprintf('Running bipose detections on validation set\n');
pair_dets = get_pair_dets(conf.cache_dir, test_seqs, ssvm_model, ...
    biposelets, conf.subposes, conf.num_joints, conf.num_dets);

fprintf('Stitching detections into sequence\n');
pose_dets = stitch_all_seqs(pair_dets, conf.stitch_weights, ...
    conf.valid_parts, conf.cache_dir);
pose_gts = get_gts(test_seqs);

fprintf('Calculating statistics\n');
flat_dets = cat(2, pose_dets{:});
flat_gts = cat(2, pose_gts{:});
pck_thresholds = conf.pck_thresholds;
all_pcks = pck(flat_dets, flat_gts, pck_thresholds); %#ok<NASGU>
limbs = conf.limbs;
all_pcps = pcp(flat_dets, flat_gts, {limbs.indices}); %#ok<NASGU>
save(fullfile(conf.cache_dir, 'final-stats.mat'), 'all_pcks', ...
    'all_pcps', 'pck_thresholds', 'limbs');

fprintf('Done!\n');
end

