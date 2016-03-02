function centroids = calculate_centroids(joint_loc_labels, num_classes)
%CALCULATE_CENTROIDS Get biposelet class centroids using K-means
% Subtract out the first coordinate in each set
[~, centroids] = kmeans(joint_loc_labels, num_classes, 'MaxIter', 1000, 'Replicates', 20, ...
    'Options', statset('UseParallel',1));
end