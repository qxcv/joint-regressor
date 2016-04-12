function pair_dets = get_pair_dets(cache_dir, test_seqs, ssvm_model, ...
    biposelets, subposes, num_joints, num_dets)
%GET_PAIR_DETS Get biposelet detections for each frame pair in test seqs
pd_path = fullfile(cache_dir, 'pair_dets.mat');
try
    fprintf('Using old (possibly stale) detections\n');
    l = load(pd_path);
    pair_dets = l.pair_dets;
catch
    fprintf('Regenerating detections\n');
    pair_dets = get_test_detections(test_seqs, ssvm_model, biposelets, ...
        subposes, num_joints, num_dets, cache_dir);
    save(pd_path, 'pair_dets');
end
end
