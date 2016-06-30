function centroids = generate_centroids_from_h5s(h5_paths, subposes, num_classes)
%GENERATE_CENTROIDS_FROM_H5S Generate centroids from HDF5s. Won't cache.
%(see cluster_h5s for more).

fprintf('Fetching labels from HDF5s\n');
[train_classes, train_labels] = extract_joint_loc_labels(h5_paths, subposes);

fprintf('Clustering, for each subpose\n');
centroids = cell([1 length(subposes)]);
parfor subpose_num=1:length(subposes)
    % poselet_num + 1 because class 1 is background
    labels = ...
        train_labels{subpose_num}(train_classes == subpose_num + 1, :);
    centroids{subpose_num} = calculate_centroids(labels, num_classes);
end
end

