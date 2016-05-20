function accum_stats(test_seqs, preds, save_dir, pck_thresholds, ...
    pcp_thresholds, limbs, limb_combos)
%ACCUM_STATS Compute statistics for detections over some test sequences
% This should work even with detections I've accumulated from third party
% code (suitably modified by me to produce desired output format, of
% course).
assert(isstruct(test_seqs) && iscell(preds));
assert(length(test_seqs.seqs) == length(preds));

if ~any(0.5 == pcp_thresholds)
    warning('JointRegressor:accum_stats:quirkyPCP', ...
        ['You didn''t include 0.5 in your list of PCP thresholds, even ' ...
         'though it is the standard threshold. Was that a mistake?']);
end

pose_gts = get_gts(test_seqs);
flat_dets = cat(2, preds{:});
flat_gts = cat(2, pose_gts{:});

% Start with PCKs
all_pcks = pck(flat_dets, flat_gts, pck_thresholds);
pck_table = ;

% Now do PCPs
for thresh=pcp_thresholds
    assert(thresh >= 1e-3, 'PCP ridiculously low!');
    assert(thresh < 1, 'PCP of 1 or more is probably too big');
    all_pcps = pcp(flat_dets, flat_gts, {limbs.indices}, thresh);
    pcp_table = pcp_summary_table(all_pcps, limbs, limb_combos);
    dest_fn = sprintf('pcp_at_%0.3f.csv', thresh);
    dest_path = fullfile(save_dir, dest_fn);
    writetable(pcp_table, dest_path);
end
end

