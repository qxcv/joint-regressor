function show_datum(datum)
%SHOW_DATUM Visualise a specific datum
im = readim(datum);
imshow(im);
hold on;
plot_joints(datum.joint_locs);
hold off;
end

