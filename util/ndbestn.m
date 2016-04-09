function [varargout] = ndbestn(in_mat, N)
%NDBESTN Return ND subscripts of maximum N results
assert(nargout == ndims(in_mat), 'Need an output for each dimension');
assert(numel(in_mat) >= N, 'Need at least N inputs to produce N outputs');
varargout = cell([1 nargout]);
[~, sorted_inds] = sort(in_mat(:), 'Descend');
best_inds = sorted_inds(1:N);
[varargout{:}] = ind2sub(size(in_mat), best_inds);
end