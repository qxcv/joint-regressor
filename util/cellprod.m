function cprod = cellprod(c1, c2)
%CELLPROD Cartesian product of cell arrays
assert(iscell(c1) && iscell(c2));
c1_inds = 1:length(c1);
c2_inds = 1:length(c2);
[c1_trans, c2_trans] = meshgrid(c1_inds, c2_inds);
c1_cells = flat(c1(c1_trans(:)));
c2_cells = flat(c2(c2_trans(:)));
cmat = cat(2, c1_cells, c2_cells);
cprod = mat2cell(cmat, ones([1 size(cmat, 1)]), 2);
end

