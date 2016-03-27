% Compute the deformation feature given parent locations,
% child locations, and the child part
function res = defvector(part, x1, y1, x2, y2, child_type, parent_type)
disp = part.subpose_disps{child_type}{parent_type};
mean_y = disp(1);
mean_x = disp(2);

% XXX: This could be totally wrong! I really need to walk through all of my
% mean displacements and understand exactly what the components of
% subpose_disps{}{} are and exactly what passmsg and functions like this
% are doing with them.
dx = (x2 - x1 - mean_x);
dy = (y2 - y1 - mean_y);
res = -[dx^2, dx, dy^2, dy]'; % gaussian normalization + variances
