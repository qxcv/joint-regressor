function [varargout] = ndfind(mask)
%NDFIND N-dimensional find()
varargout = cell([1 nargout]);
assert(nargout == ndims(mask), 'Need an output for each dimension');
[varargout{:}] = ind2sub(size(mask), find(mask));
end