% Compute the deformation feature given child locations, parent locations
% and the child part
function res = defvector(part, x_c, y_c, x_p, y_p, child_type, ...
    parent_type, downsample_factor)
disp = part.subpose_disps{child_type}{parent_type} / downsample_factor;

% This should be correct. Work through displacement calculation code to
% check
dx = x_p - x_c - disp(1);
dy = y_p - y_c - disp(2);
res = -[dx^2, dx, dy^2, dy]'; % gaussian normalization + variances
