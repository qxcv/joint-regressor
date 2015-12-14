function centroids = calculate_centroids(joint_loc_labels, num_classes)
%CALCULATE_CENTROIDS Get biposelet class centroids using K-means
% Subtract out the first coordinate in each set
sub_mat = repmat(joint_loc_labels(:, 1:2), 1, size(joint_loc_labels, 2) / 2);
X = joint_loc_labels - sub_mat;
[~, centroids] = kmeans(X, num_classes, 'MaxIter', 1000, 'Replicates', 20, ...
    'Options', statset('UseParallel',1));
end