% Compute the deformation feature given parent locations,
% child locations, and the child part
function res = defvector(part, x1, y1, x2, y2, child_type, parent_type, scale)
% {x,y}{1,2} are in heatmap coordinates, so I need to divide by CNN stride
% (scale)
disp = part.subpose_disps{child_type}{parent_type} / scale;

% This should be correct. Work through displacemenet calculation code to
% check
dx = (x1 - x2 - disp(1));
dy = (y1 - y2 - disp(2));
res = -[dx^2, dx, dy^2, dy]'; % gaussian normalization + variances
