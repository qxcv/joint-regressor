function plot_detection_seq(seq_data, seq_poses)
%PLOT_DETECTION_SEQ Plot pose sequence on related test data
num_poses = length(seq_poses);
assert(num_poses >= 1);
grid_size = ceil(sqrt(num_poses));
actual_rows = max(ceil((num_poses - 1) / grid_size), 1);
actual_cols = max(~~actual_rows * grid_size, mod(num_poses - 1, grid_size) + 1);
fprintf('Plotting %i rows and %i columns (GS=%i)\n', ...
    actual_rows, actual_cols, grid_size);
for pnum=1:num_poses
    row = floor((pnum - 1) / grid_size) + 1;
    column = mod(pnum - 1, grid_size) + 1;
    plot_num = (row-1) * grid_size + column;
    subplot(actual_rows, actual_cols, plot_num);
    im = readim(seq_data(pnum));
    imagesc(im);
    axis image off;
    axis equal;
    hold on;
    plot_joints(seq_poses{pnum});
    hold off;
end
end

