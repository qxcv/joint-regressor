function vec = joints2vec(f1_locs, f2_locs)
%JOINTS2VEC Convert a pair of joint locations from adjacent frames to a
%vector for biposelet clustering.
combined = [f1_locs; f2_locs];
% Subtract out location of first joint in first frame from subsequent joint
% locations; strip location of first joint.
subbed = bsxfun(@minus, combined, combined(1, :));
without_zero = subbed(2:end, :);
% Return row vector
vec = without_zero(:)';
end

