function detections = get_test_detections(test_seqs, varargin)
%GET_TEST_DETECTIONS Run over a sequence and get detections for each pair
detections = cell([1 length(test_seqs.seqs)]);
num_seqs = length(test_seqs.seqs);
for seq_num=1:num_seqs
    fprintf('Working in seq %i/%i\n', seq_num, num_seqs);
    detections{seq_num} = get_seq_detections(test_seqs, seq_num, varargin{:});
end
end
