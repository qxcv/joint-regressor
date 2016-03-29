function coords = unflatten_coords(flat)
%UNFLATTEN_COORDS Inverse of flatten_coords
assert(isvector(flat) && mod(numel(flat), 2) == 0);
coords = reshape(flat, [2 numel(flat) / 2])';
end