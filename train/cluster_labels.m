function indices = cluster_labels(labels, centroids)
%CLUSTER_LABELS Find cluster indices for labels
dists = pdist2(labels, centroids);
[~, indices] = min(dists, [], 2);
end