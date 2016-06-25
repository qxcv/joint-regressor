function [rv_type, rv_loc, rv_dist] = get_subpose(true_joints, sp_biposelets)
%GET_SUBPOSE Gets a single "best" subpose type and location for the GT
% get_subposes wraps this function to get subpose types and locations for
% each subpose. Locations will be in image coordinates. true_joints and
% sp_biposelets should be pre-scaled, and true_joints should additionally
% be translated as necessary if the source image has been cropped.
rv_dist = inf;
rv_type = [];
rv_loc = [];
for type=1:length(sp_biposelets)
    centroid = unflatten_coords(sp_biposelets(type, :));
    offset = mean(true_joints - centroid, 1);
    pred_joints = centroid + offset;
    dist = mean_dists(true_joints, pred_joints);
    if dist < rv_dist
        rv_dist = dist;
        rv_type = type;
        rv_loc = offset;
    end
end
end

