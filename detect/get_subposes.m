function [rv_types, rv_locs, rv_dist] = get_subposes(j1, j2, subposes, uf_biposelets)
%GET_SUBPOSES Estimate subpose types and locations for some ground truth
% Algorithm is a simple greedy which chooses the "best" type and location
% for each subpose independently, where "best" means "lowest L2 distance".
num_sp = length(subposes);
rv_locs = nan([num_sp 2]);
rv_types = nan([1 num_sp]);
sp_dists = nan([1 num_sp]);
for sp_idx=1:num_sp
    sp_biposelets = uf_biposelets{sp_idx};
    sp_joints = subposes(sp_idx).subpose;
    sp_joints = cat(1, j1(sp_joints, :), j2(sp_joints, :));
    [rv_types(sp_idx), rv_locs(sp_idx, :), sp_dists(sp_idx)] = ...
        get_subpose(sp_joints, sp_biposelets);
end
rv_dist = sum(sp_dists);

% Sanity checks
assert(~any(isnan(rv_locs(:))));
assert(~any(isnan(rv_types(:))));
assert(~isnan(rv_dist));
end
