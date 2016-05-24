function seq_gts = get_gts(test_seqs, orig)
%GET_GTS Fetch ground truth pose locations from a test dataset
% Invalid joints will be set to NaN
if ~exist('orig', 'var')
    orig = false;
end
seq_gts = cell([1 length(test_seqs.seqs)]);
for i=1:length(seq_gts)
    seq = test_seqs.seqs{i};
    seq_gts{i} = cell([1 length(seq)]);
    for j=1:length(seq)
        datum = test_seqs.data(seq(j));
        % orig is handy when you don't want transformed joints (e.g. in
        % accum_stats, as I've tried to stick to a common, original format
        % for predictions on each data set).
        if orig
            locs = datum.orig_joint_locs;
            if hasfield(datum, 'orig_visible')
                locs(~datum.orig_visible, :) = nan;
            end
        else
            locs = datum.joint_locs;
            if hasfield(datum, 'visible')
                locs(~datum.visible, :) = nan;
            end
        end
        seq_gts{i}{j} = locs;
    end
end
end

