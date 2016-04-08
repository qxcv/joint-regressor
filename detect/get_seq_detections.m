function detections = get_seq_detections(dataset, seq_num, ssvm_model, thresh)
%GET_SEQ_DETECTIONS Evaluate on a single test sequence
% Differs from get_test_detections in that this only works on a single
% sequence, whereas get_test_detections works on a whole set of detections
seq = dataset.seqs{seq_num};
num_pairs = length(seq) - 1;
detections = cell([1 num_pairs]);
for pair_idx = 1:num_pairs
    fprintf('Working on pair %i/%i...', pair_idx, num_pairs);
    idx1 = seq(pair_idx);
    idx2 = seq(pair_idx+1);
    im1_info = dataset.data(idx1);
    im2_info = dataset.data(idx2);
    start = tic;
    [b, ~, ~] = detect(im1_info, im2_info, [], [], ssvm_model, thresh);
    time_taken = toc(start);
    fprintf(' took %fs\n', time_taken);
    % TODO: Need to recover pose from boxes
    detections{pair_idx} = b;
end
end

