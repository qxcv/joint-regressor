function [ pck_table, pcp_table ] = format_stats(final_stats, ...
    limb_combinations, pck_joints)
%FORMAT_STATS Put PCPs and PCKs in appropriate format for paper
% Will give back tables for PCPs and PCKs
pcp_table = format_pcps(final_stats, limb_combinations);
pck_table = format_pcks(final_stats, pck_joints);
end

function pck_table = format_pcks(final_stats, pck_joints)
joint_names = pck_joints.keys;
num_out_joints = length(joint_names);
accs = cell([1 num_out_joints]);
for joint_idx=1:num_out_joints
    these_joints = pck_joints(joint_names{joint_idx});
    these_pcks = cellfun(@(acc) mean(acc(these_joints)), final_stats.all_pcks);
    accs{joint_idx} = these_pcks';
end
pck_table = table(final_stats.pck_thresholds', accs{:}, ...
    'VariableNames', ['Threshold' joint_names]);
end

function pcp_table = format_pcps(final_stats, limb_combinations)
limbs = final_stats.limbs;
limb_names = limb_combinations.keys;
combined_pcps = zeros([length(limb_names), 1]);
for limb_idx=1:length(limb_names)
    str_names = limb_combinations(limb_names{limb_idx});
    [~, limb_indices] = intersect({limbs.names}, str_names);
    limb_pcps = final_stats.all_pcps(limb_indices);
    combined_pcps(limb_idx) = mean(limb_pcps);
end
pcp_table = table(limb_names', combined_pcps, ...
    'VariableNames', {'Limb', 'PCP'});
end
