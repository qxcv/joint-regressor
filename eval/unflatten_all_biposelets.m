function uf_bp_cells = unflatten_all_biposelets(bp_cells)
%UNFLATTEN_ALL_BIPOSELETS "Unflatten" centroids used as biposelets
% Turns the flat vectors produced by K-means into actual J*2 matrices of
% joint coordinates for the relevant subposes. Can be applied to the
% biposelets returned by cluster_h5s (for instance).
assert(isvector(bp_cells) && iscell(bp_cells));
uf_bp_cells = cell([1 length(bp_cells)]);
for sp_idx=1:length(bp_cells)
    flat = bp_cells{sp_idx};
    num_types = size(flat, 1);
    num_joints = size(flat, 2) / 2;
    unflat = nan([num_joints, 2, num_types]);
    for bp_idx=1:num_types
        unflat(:, :, bp_idx) = unflatten_coords(flat(bp_idx, :));
    end
    assert(~any(isnan(unflat(:))));
    uf_bp_cells{sp_idx} = unflat;
end
end
