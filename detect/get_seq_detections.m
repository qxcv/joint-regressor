function detections = get_seq_detections(dataset, seq_num, ssvm_model, ...
    biposelets, subposes, num_joints, num_results)
%GET_SEQ_DETECTIONS Evaluate on a single test sequence
% Differs from get_test_detections in that this only works on a single
% sequence, whereas get_test_detections works on a whole set of detections
seq = dataset.seqs{seq_num};
num_pairs = length(seq) - 1;
empt = @() cell([1 num_pairs]);
detections = struct('raw', empt, 'recovered', empt);
for pair_idx = 1:num_pairs
    fprintf('Working on pair %i/%i...', pair_idx, num_pairs);
    idx1 = seq(pair_idx);
    idx2 = seq(pair_idx+1);
    im1_info = dataset.data(idx1);
    im2_info = dataset.data(idx2);
    start = tic;
    [boxes, ~, ~] = detect(im1_info, im2_info, ssvm_model, 'NumResults', num_results);
    assert(length(boxes) == num_results, ...
        'Expected %i detections, got %i', num_results, length(boxes));
    time_taken = toc(start);
    fprintf(' took %fs\n', time_taken);
    detections(pair_idx).raw = boxes;
    recovered = cell([1 length(boxes)]);
    for det=1:length(boxes)
        recovered{det} = boxes2pose(boxes(det), biposelets, ...
            ssvm_model.cnn.window, subposes, num_joints);
    end
    detections(pair_idx).recovered = recovered; 
end
end
