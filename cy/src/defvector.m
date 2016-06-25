% Compute the deformation feature given child locations, parent locations
% and the child part displacements
function res = defvector(child_disps, x_c, y_c, x_p, y_p, child_type, parent_type, scale)
% {x,y}{1,2} are in heatmap coordinates, so I need to divide by CNN stride
% (scale)
disp = child_disps{child_type}{parent_type} / scale;

% This should be correct. Work through displacement calculation code to
% check
dx = x_p - x_c - disp(1);
dy = y_p - y_c - disp(2);
res = -[dx^2, dx, dy^2, dy]'; % gaussian normalization + variances
