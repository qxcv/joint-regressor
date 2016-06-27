function mu = mean_dists(j1, j2)
%MEAN_DISTS Compute mean Euclidean distance between sets of joints
assert(ismatrix(j1) && ndims(j2) >= 2 && ndims(j2) <= 3);
assert(size(j1, 2) == 2);
mu = squeeze(mean(sqrt(sum(bsxfun(@minus, j1, j2).^2, 2))));

% We support vectorisation, but for now only on the second joint. I found
% that not vectorising this code made it really slow :(
if ndims(j2) == 3
    assert(isvector(mu) && numel(mu) == size(j2, 3));
else
    assert(isscalar(mu));
end
end
