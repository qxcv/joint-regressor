function seq_gts = get_gts(test_seqs)
%GET_GTS Fetch ground truth pose locations from a test dataset
seq_gts = cell([1 length(test_seqs.seqs)]);
for i=1:length(seq_gts)
    seq = test_seqs.seqs{i};
    seq_gts{i} = {test_seqs.data(seq).joints};
end
end

