function store_mean_pixel(train_patch_dir, cache_dir)
%STORE_MEAN_PIXEL Calculate mean pixel and store it in the cache.

% Collect paths
train_h5s = files_with_extension(train_patch_dir, '.h5');

mean_pixel_path = fullfile(cache_dir, 'mean_pixel.mat');
if ~exist(mean_pixel_path, 'file')
    fprintf('Calculating mean pixel\n');
    image_mean_pixel = compute_mean_pixel(train_h5s, '/images');
    flow_mean_pixel = compute_mean_pixel(train_h5s, '/flow');
    save(mean_pixel_path, 'flow_mean_pixel', 'image_mean_pixel');
else
    fprintf('Mean pixels already exists');
end

display(image_mean_pixel);
display(flow_mean_pixel);
end