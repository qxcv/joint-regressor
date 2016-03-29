function flat = flatten_coords(coords)
%FLATTEN_COORDS Flatten marix of 2D coords into column vector (row-major)
assert(ismatrix(coords) && size(coords, 2) == 2);
flat = reshape(coords', [numel(coords), 1]);
end

