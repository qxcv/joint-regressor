function [indices, dists] = cluster_labels(labels, centroids)
%CLUSTER_LABELS Find cluster indices for labels
dists = pdist2(labels, centroids);
if nargin > 1
    [dists, indices] = min(dists, [], 2);
else
    [~, indices] = min(dists, [], 2);
end
end
