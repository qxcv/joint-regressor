function show_datum(datum)
%SHOW_DATUM Visualise a specific datum
im = readim(datum);
imshow(im);
hold on;
for i=1:size(datum.joint_locs, 1);
    x = datum.joint_locs(i, 1);
    y = datum.joint_locs(i, 2);
    h = plot(x, y, 'LineStyle', 'none', 'Marker', '+', 'MarkerSize', 15);
    text(x+8, y+8, cellstr(num2str(i)), 'Color', h.Color, 'FontSize', 15);
end
hold off;
end

