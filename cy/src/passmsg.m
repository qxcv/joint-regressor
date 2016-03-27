function [score, Ix, Iy, Im] = passmsg(child, parent)
% Pass a message from child component to parent componoent, returning four
% H*W*K matrices. In each matrix, the (h, w, k)-th entry corresponds to a
% parent of type k at location (h, w). The matrices can be interpreted as
% follows:
% - score gives score of best subpose in which parent has specified
%   location and type
% - Ix gives the x location of the current child in the best subpose that
%   has parent in specified configuration
% - Iy is the same but for child y location
% - Im is the same but for child type
height = size(parent.score, 1);
width = size(parent.score, 2);
% Number of parent and child types, respectively
parent_K = size(parent.score, 3);
child_K = size(child.score, 3);
assert(child_K == parent_K);

[score, Ix0, Iy0] = deal(zeros(height, width, parent_K, child_K));
for parent_type = 1:parent_K
    for child_type = 1:child_K
        fixed_score_map = double(child.score(:, :, child_type));
        % this is child_center - parent_center, IIRC
        mean_disp = child.subpose_disps{child_type}{parent_type};
        assert(isvector(mean_disp) && length(mean_disp) == 2);
        % TODO: Two problems here:
        % (1) Displacements are at CNN-receptive-field-relative scale, but
        %     heatmap is clearly not at the same scale due to inherent
        %     downsampling. Probably need to use step argument of shiftdt
        %     to fix this.
        % (2) Not sure whether mean_disp is pointing in the right
        %     direction for shiftdt.
        [score(:, :, parent_type, child_type), Ix0(:, :, parent_type, child_type), ...
            Iy0(:, :, parent_type, child_type)] = shiftdt(fixed_score_map, ...
            child.gauw, int32(mean_disp), int32([width, height]), 1);
        
        % If there was a prior-of-deformation (like the image evidence in
        % Chen & Yuille's model), then I would add it in here.
    end
end
[score, Im] = max(score, [], 4);
assert(ndims(score) == 3 && ndims(Im) == 3);
[Ix, Iy] = deal(zeros(height, width, parent_K));
for row = 1:height
    for col = 1:width
        for ptype = 1:parent_K
            ctype = Im(row, col, ptype);
            assert(isscalar(ctype));
            Ix(row, col, ptype) = Ix0(row, col, ptype, ctype);
            Iy(row, col, ptype) = Iy0(row, col, ptype, ctype);
        end
    end
end
% "score" is a message that will be added to the parent's total score in
% detect.m. Hence, we need it to be the right size.
assert(all(size(score) == [height width parent_K]));