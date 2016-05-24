function pair_dets = get_pair_dets(cache_dir, test_seqs, ssvm_model, ...
    biposelets, subposes, num_joints, num_dets)
%GET_PAIR_DETS Get biposelet detections for each frame pair in test seqs
pd_path = fullfile(cache_dir, 'pair_dets.mat');
try
    l = load(pd_path);
    pair_dets = l.pair_dets;
    fprintf('Using old (possibly stale) detections\n');
catch
    fprintf('Regenerating detections\n');
    pair_dets = cell([1 length(test_seqs.seqs)]);
    num_seqs = length(test_seqs.seqs);
    for seq_num=1:num_seqs
        fprintf('Working in seq %i/%i\n', seq_num, num_seqs);
        pair_dets{seq_num} = get_seq_detections(test_seqs, seq_num, ssvm_model, biposelets, ...
            subposes, num_joints, num_dets, cache_dir);
    end
    save(pd_path, 'pair_dets');
end
end
