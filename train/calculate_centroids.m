function centroids = calculate_centroids(joint_loc_labels, num_classes)
%CALCULATE_CENTROIDS Get biposelet class centroids using K-means
% Subtract out the first coordinate in each set

% K-means notes:
%  - MaxIter of 250 sometimes fails to converge
%  - Originally had replicates at 20, but that's probably overkill
[~, centroids] = kmeans(joint_loc_labels, num_classes, 'MaxIter', 500, ...
    'Replicates', 5, 'Options', statset('UseParallel',1));
end
