function best_weights_evar = hacky_find_weights
%HACKY_FIND_WEIGHTS Find stitching weights. You really only need to run
%this once.
fprintf('Loading stuff\n');
conf = get_conf_mpii;
[~, ~, test_seqs] = get_mpii_cooking(...
    conf.dataset_dir, conf.cache_dir, conf.pair_mean_dist_thresh, ...
    conf.subposes, conf.cnn.step, conf.template_scale, conf.trans_spec);
pair_dets = parload(fullfile(conf.cache_dir, 'pair_dets_for_wf.mat'), 'pair_dets');
pose_gts = get_gts(test_seqs);
fprintf('Finding weights\n');
best_weights_evar = stitching_weight_search(pair_dets, conf.valid_parts, pose_gts, conf.limbs);
fprintf('Got them!\n');
end

