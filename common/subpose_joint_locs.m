function sp_joint_locs = subpose_joint_locs(im_data, pair, subpose)
%SUBPOSE_JOINT_LOCS Extract locations of joints in subpose
assert(numel(pair) == 2);
assert(isvector(subpose));
j1 = im_data(pair(1)).joint_locs;
j2 = im_data(pair(2)).joint_locs;
assert(size(j1, 2) == 2 && size(j1, 2) == 2);
assert(ismatrix(j1) && ismatrix(j2));
sp_joint_locs = cat(1, j1(subpose, :), j2(subpose, :));
assert(ismatrix(sp_joint_locs) && size(sp_joint_locs, 2) == 2);
end

