function cluster_h5s(num_classes, train_patch_dir, val_patch_dir)
%CLUSTER_H5S Add a "biposelet" dataset to each HDF5 in the cache

% Collect paths
train_h5s = files_with_extension(train_patch_dir, '.h5');
val_h5s = files_with_extension(val_patch_dir, '.h5');

% Get centroids using only training data
fprintf('Generating cluster centers\n');
train_labels = extract_joint_loc_labels(train_h5s);
centroids = calculate_centroids(train_labels, num_classes);

% Now apply cluster labels to both training and validation data
all_fns = cat(2, train_h5s, val_h5s);
num_fns = length(all_fns);
for i=1:num_fns
    fprintf('Writing labels on sample %d/%d\n', i, num_fns);
    fn = all_fns{i};
    current_labels = extract_joint_loc_labels({fn});
    % Subtract 1 from labels because Caffe uses zero-based indexing
    clusters = cluster_labels(current_labels, centroids)' - 1;
    assert(size(clusters, 1) == 1);
    % Now write out labels
    h5create(fn, '/poselet', [1 length(clusters)], 'DataType', 'int32');
    h5write(fn, '/poselet', clusters);
end
end