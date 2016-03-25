function [score, Ix, Iy, Imc, Imp] = passmsg(child, parent)
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
        
%   double *vals = (double *)mxGetPr(prhs[0]);
%   int sizx  = mxGetN(prhs[0]);
%   int sizy  = mxGetM(prhs[0]);
%   double ax = -mxGetScalar(prhs[1]);
%   double bx = -mxGetScalar(prhs[2]);
%   double ay = -mxGetScalar(prhs[3]);
%   double by = -mxGetScalar(prhs[4]);
%   int offx  = (int)mxGetScalar(prhs[5])-1;
%   int offy  = (int)mxGetScalar(prhs[6])-1;
%   int lenx  = (int)mxGetScalar(prhs[7]);
%   int leny  = (int)mxGetScalar(prhs[8]);
%   double step = mxGetScalar(prhs[9]);
        
        % If there was a prior-of-deformation (like the image evidence in
        % Chen & Yuille's model), then I would add it in here.
    end
end
[score, Imcp] = max(score, [], 4);
assert(ndims(score) == 3 && ndims(Imcp) == 3);
% XXX: Everything below is hopelessly broken. Firstly, I don't think it
% works at all (the ind2sub call simply doesn't make sense). Secondly, I'm
% not sure how it will let me recover type information.
% Some basic experiments suggest that Imp will be all ones. Why is this
% even useful?
[Imc, Imp] = ind2sub([child_K, parent_K], Imcp);
[Ix, Iy] = deal(zeros(height, width));
for row = 1:height
    for col = 1:width
        Ix(row, col) = Ix0(row, col, Imc(row, col), Imp(row, col));
        Iy(row, col) = Iy0(row, col, Imc(row, col), Imp(row, col));
    end
end
% "score" is a message that will be added to the parent's total score in
% detect.m. Hence, we need it to be the right size.
assert(all(size(score) == [height width parent_K]));