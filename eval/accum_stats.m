function accum_stats(test_seqs, preds, save_dir, ...
    pck_thresholds, pck_joints, ...       For PCK
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
all_pcks = pck(flat_dets, flat_gts, pck_thresholds);
pck_table = format_pcks(all_pcks, pck_thresholds, pck_joints);
dest_path = fullfile(save_dir, 'pcks.csv');
writetable(pck_table, dest_path);

% Now do PCPs
for thresh=pcp_thresholds
    assert(thresh >= 0, 'Threshold shouldn''t be negative');
    assert(thresh <= 1, 'PCP of >=1 is probably too big');
    all_pcps = pcp(flat_dets, flat_gts, {limbs.indices}, thresh);
    pcp_table = format_pcps(all_pcps, limbs, limb_combos);
    dest_fn = sprintf('pcp_at_%0.3f.csv', thresh);
    dest_path = fullfile(save_dir, dest_fn);
    writetable(pcp_table, dest_path);
end
end

function pck_table = format_pcks(all_pcks, pck_thresholds, pck_joints)
joint_names = pck_joints.keys;
num_out_joints = length(joint_names);
accs = cell([1 num_out_joints]);
for joint_idx=1:num_out_joints
    these_joints = pck_joints(joint_names{joint_idx});
    these_pcks = cellfun(@(acc) mean(acc(these_joints)), all_pcks);
    accs{joint_idx} = these_pcks';
end
pck_table = table(pck_thresholds', accs{:}, ...
    'VariableNames', ['Threshold' joint_names]);
end

function pcp_table = format_pcps(all_pcps, limbs, limb_combinations)
limb_names = limb_combinations.keys;
combined_pcps = zeros([length(limb_names), 1]);
for limb_idx=1:length(limb_names)
    str_names = limb_combinations(limb_names{limb_idx});
    [~, limb_indices] = intersect({limbs.names}, str_names);
    limb_pcps = all_pcps(limb_indices);
    combined_pcps(limb_idx) = mean(limb_pcps);
end
pcp_table = table(limb_names', combined_pcps, ...
    'VariableNames', {'Limb', 'PCP'});
end
