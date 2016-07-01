function accum_stats(test_seqs, preds, save_dir, ...
    pck_thresholds, pck_joints, pck_norm_joints, ...       For PCK
    pcp_thresholds, limbs, limb_combos) % For PCP
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

mkdir_p(save_dir);
pose_gts = get_gts(test_seqs, true);
flat_dets = cat(2, preds{:});
flat_gts = cat(2, pose_gts{:});

% Start with PCKs
% TODO: Need to normalise PCK (using hips -> shoulders) for H3.6M
if isempty(pck_norm_joints)
    all_pcks = pck(flat_dets, flat_gts, pck_thresholds);
else
    all_pcks = pck(flat_dets, flat_gts, pck_thresholds, pck_norm_joints);
end
pck_table = format_pcks(all_pcks, pck_thresholds, pck_joints);
dest_path = fullfile(save_dir, 'pcks.csv');
writetable(pck_table, dest_path);

% Now do PCPs
pcps_at_thresh = cell([1 length(pcp_thresholds)]);
for thresh_idx=1:length(pcp_thresholds)
    thresh = pcp_thresholds(thresh_idx);
    assert(thresh >= 0, 'Threshold shouldn''t be negative');
    assert(thresh <= 1, 'PCP of >=1 is probably too big');
    all_pcps = pcp(flat_dets, flat_gts, {limbs.indices}, thresh);
    pcps_at_thresh{thresh_idx} = all_pcps;
end
pcp_table = format_pcps(pcps_at_thresh, pcp_thresholds, limbs, limb_combos);
dest_path = fullfile(save_dir, 'pcps.csv');
writetable(pcp_table, dest_path);
end

function pcp_table = format_pcps(all_pcps, pcp_thresholds, limbs, limb_combinations)
limb_names = limb_combinations.keys;
combined_pcps = cell([1, length(limb_names)]);
for limb_idx=1:length(limb_names)
    str_names = limb_combinations(limb_names{limb_idx});
    [~, limb_indices] = intersect({limbs.names}, str_names);
    limb_pcps = cellfun(@(pcps) mean(pcps(limb_indices)), all_pcps);
    combined_pcps{limb_idx} = limb_pcps';
end
safe_limb_names = cellfun(@genvarname, limb_names, 'UniformOutput', false);
pcp_table = table(pcp_thresholds', combined_pcps{:}, ...
    'VariableNames', [{'Threshold'} safe_limb_names]);
end
