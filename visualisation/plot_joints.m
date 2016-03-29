function plot_joints(locs, colour)
% Plot an N * 2 array of joints
if nargin < 2
    extra_args = {};
else
    extra_args = {'MarkerEdgeColor', colour, 'MarkerFaceColor', colour};
end

for i=1:size(locs, 1);
    x = locs(i, 1);
    y = locs(i, 2);
    h = plot(x, y, 'LineStyle', 'none', 'Marker', '+', 'MarkerSize', 15, extra_args{:});
    text(double(x+8), double(y+8), cellstr(num2str(i)), 'Color', h.MarkerEdgeColor, 'FontSize', 15);
end
end

