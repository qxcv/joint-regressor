function normed_locs = norm_subpose(d1, d2, scale, cnn_window, subpose_idxs)
%NORM_SUBPOSE Norm joint coords to lie in CNN window and be well scaled
assert(isscalar(scale));
assert(isvector(cnn_window));
assert(isvector(subpose_idxs) && isnumeric(subpose_idxs));

joints = cat(1, d1.joint_locs(subpose_idxs, :), d2.joint_locs(subpose_idxs, :));
assert(ismatrix(joints) && ~isvector(joints));
assert(size(joints, 2) == 2);
maxes = max(joints, [], 1);
mins = min(joints, [], 1);
assert(numel(maxes) == 2 && numel(mins == 2));
midpoint = mins + (maxes - mins) ./ 2;
anchor = midpoint - scale;
trans_joint_locs = bsxfun(@minus, joints, reshape(anchor, [1 2]));
scale_factors = cnn_window ./ scale;
normed_locs = bsxfun(@times, trans_joint_locs, reshape(scale_factors, [1 2]));
assert(all(normed_locs >= 0));
assert(all(bsxfun(@lt, normed_locs, reshape(cnn_window, [1 2]))));
assert(all(size(normed_locs) == size(joints)));
end