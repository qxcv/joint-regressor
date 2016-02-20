function store_mean_pixel(train_patch_dir, cache_dir)
%STORE_MEAN_PIXEL Calculate mean pixel and store it in the cache.

% Collect paths
train_h5s = files_with_extension(train_patch_dir, '.h5');

mean_pixel_path = fullfile(cache_dir, 'mean_pixel.mat');
if ~exist(mean_pixel_path, 'file')
    fprintf('Calculating mean pixel\n');
    % Make sure that saved names match data structure names
    out_struct.images = compute_mean_pixel(train_h5s, '/images');
    out_struct.flow = compute_mean_pixel(train_h5s, '/flow');
    save(mean_pixel_path, '-struct', 'out_struct');
    display(out_struct.images);
    display(out_struct.flow);
else
    fprintf('Mean pixels already exist');
end
end