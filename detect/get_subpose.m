function [rv_type, rv_loc, rv_dist] = get_subpose(true_joints, uf_sp_biposelets)
%GET_SUBPOSE Gets a single "best" subpose type and location for the GT
% get_subposes wraps this function to get subpose types and locations for
% each subpose. Locations will be in image coordinates. true_joints and
% sp_biposelets should be pre-scaled, and true_joints should additionally
% be translated as necessary if the source image has been cropped.
offsets = mean(bsxfun(@minus, true_joints, uf_sp_biposelets), 1);
pred_joints_all = bsxfun(@plus, uf_sp_biposelets, offsets);
dists = mean_dists(true_joints, pred_joints_all);
[rv_dist, rv_type] = min(dists);
rv_loc = offsets(:, :, rv_type);
assert(isvector(rv_loc) && size(rv_loc, 2) == 2);
end
