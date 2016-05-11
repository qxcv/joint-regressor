function seq_gts = get_gts(test_seqs)
%GET_GTS Fetch ground truth pose locations from a test dataset
% Invalid joints will be set to NaN
seq_gts = cell([1 length(test_seqs.seqs)]);
for i=1:length(seq_gts)
    seq = test_seqs.seqs{i};
    seq_gts{i} = cell([1 length(seq)]);
    for j=1:length(seq)
        datum = test_seqs.data(seq(j));
        locs = datum.joint_locs;
        if hasfield(datum, 'visible')
            locs(~datum.visible, :) = nan;
        end
        seq_gts{i}{j} = locs;
    end
end
end

