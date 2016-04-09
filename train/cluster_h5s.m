function biposelets = cluster_h5s(num_classes, subposes, train_patch_dir, val_patch_dir, cache_dir)
%CLUSTER_H5S Add a "biposelet" dataset to each HDF5 in the cache

% Collect paths
train_h5s = files_with_extension(train_patch_dir, '.h5');
val_h5s = files_with_extension(val_patch_dir, '.h5');

% Get centroids using only training data
centroid_path = fullfile(cache_dir, 'centroids.mat');
if exist(centroid_path, 'file')
    fprintf('Loading cluster centers from %s\n', centroid_path);
    loaded = load(centroid_path);
    centroids = loaded.centroids;
else
    fprintf('Generating cluster centers\n');
    [train_classes, train_labels] = extract_joint_loc_labels(train_h5s, subposes);
    centroids = cell([1 length(subposes)]);
    parfor subpose_num=1:length(subposes)
        % poselet_num + 1 because class 1 is background
        labels = ...
            train_labels{subpose_num}(train_classes == subpose_num + 1, :);
        centroids{subpose_num} = calculate_centroids(labels, num_classes);
    end
    save(centroid_path, 'centroids');
end

% Now apply cluster labels to both training and validation data
all_fns = cat(2, train_h5s, val_h5s);
num_fns = length(all_fns);
for fn_no=1:num_fns
    % Read data from file
    fn = all_fns{fn_no};
    fprintf('Writing labels on sample %d/%d (%s)\n', fn_no, num_fns, fn);
    
    info = h5info(fn);
    if any(strcmp('poselet', {info.Datasets.Name}))
        fprintf('Poselet labels already exist, skipping\n');
        continue
    end
    
    [classes, current_labels] = extract_joint_loc_labels({fn}, subposes);
    
    % Determine poselet class for each sample; use ones because 1 is the
    % background (default) class *in Matlab*. Once the labels are written
    % out in one-of-K format, we'll see 0 as the default class from Python.
    poselet_classes = ones([1 length(classes)]);
    for subpose_num=1:length(subposes)
        % Subtract 1 from labels because Caffe uses zero-based indexing
        subpose_mask = classes == subpose_num + 1;
        subpose_labels = current_labels{subpose_num}(subpose_mask, :);
        if isempty(subpose_labels)
            % Loop back around to prevent clustering from breaking
            continue;
        end
        poselet_clusters = cluster_labels(subpose_labels, centroids{subpose_num})';
        offset = num_classes * (subpose_num - 1) + 1;
        assert(offset >= 1);
        assert(all(poselet_clusters + offset > 1));
        poselet_classes(subpose_mask) = poselet_clusters + offset;
    end
    
    % Make sure that background is labelled as such
    assert(all(poselet_classes(classes == 1) == 1));
    assert(all(poselet_classes(classes ~= 1) > 1));
    
    % Convert to one-of-K
    total_classes = num_classes * length(subposes) + 1;
    one_of_k_clusters = one_of_k(poselet_classes, total_classes)';
    assert(all(size(one_of_k_clusters) == [total_classes, length(classes)]));

    % Now write out labels
    h5create(fn, '/poselet', size(one_of_k_clusters), 'DataType', 'int32');
    h5write(fn, '/poselet', one_of_k_clusters);
end

% Return value
biposelets = centroids;
end
