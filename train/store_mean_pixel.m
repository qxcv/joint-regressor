function store_mean_pixel(train_patch_dir, cache_dir)
%STORE_MEAN_PIXEL Calculate mean pixel and store it in the cache.

% Collect paths
train_h5s = files_with_extension(train_patch_dir, '.h5');

mean_pixel_path = fullfile(cache_dir, 'mean_pixel.mat');
if ~exist(mean_pixel_path, 'file')
    fprintf('Calculating mean pixel\n');
    mean_pixel = compute_mean_pixel(train_h5s);
    save(mean_pixel_path, 'mean_pixel');
else
    fprintf('Mean pixel already exists');
end

display(mean_pixel);
end