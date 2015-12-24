function adjust_for_mean_pixel(train_patch_dir, val_patch_dir, cache_dir)
%ADJUST_FOR_MEAN_PIXEL Calculate mean pixel and remove it from patches.

% Collect paths
train_h5s = files_with_extension(train_patch_dir, '.h5');
val_h5s = files_with_extension(val_patch_dir, '.h5');

% Get mean pixel using only training data
mean_pixel_path = fullfile(cache_dir, 'mean_pixel.mat');
if exist(mean_pixel_path, 'file')
    fprintf('Reading mean pixel from file\n');
    load(mean_pixel_path, 'mean_pixel');
else
    fprintf('Calculating mean pixel\n');
    mean_pixel = compute_mean_pixel(train_h5s);
    save(mean_pixel_path, 'mean_pixel');
end

display(mean_pixel);

% Now apply cluster labels to both training and validation data
all_fns = cat(2, train_h5s, val_h5s);
num_fns = length(all_fns);
for i=1:num_fns
    fprintf('Subtracting mean pixel from sample %d/%d\n', i, num_fns);
    sub_mp(all_fns{i}, mean_pixel);
end
end

