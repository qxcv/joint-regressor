function centroids = cluster_biposelets(data, pairs, num_classes)
%CLUSTER_BIPOSELETS Get biposelet class centroids using K-means
num_pairs = size(pairs, 1);
joint_data_size = numel(data(1).joint_locs);
% We need to store two lots of joint coordinates for each of the input
% pairs we're given.
X = zeros(num_pairs, 2 * joint_data_size);

for i=1:size(pairs, 1)
    fst = data(pairs(i, 1));
    snd = data(pairs(i, 2));
    X(i, :) = joints2vec(fst.joint_locs, snd.joint_locs);
end

[~, centroids] = kmeans(X, num_classes);
end