function plot_joints(locs)
% Plot an N * 2 array of joints
for i=1:size(locs, 1);
    x = locs(i, 1);
    y = locs(i, 2);
    h = plot(x, y, 'LineStyle', 'none', 'Marker', '+', 'MarkerSize', 15);
    text(double(x+8), double(y+8), cellstr(num2str(i)), 'Color', h.Color, 'FontSize', 15);
end
end

