function test_clustering
%TEST_CLUSTERING Test joint labels clustering functions
COUNT_PER_DIST = 100;
NUM_DISTS = 100;
PAIRS = 7;
data = zeros(NUM_DISTS * COUNT_PER_DIST, PAIRS * 2);

% Produce NUM_DISTS lots of data, consisting of COUNT_PER_DIST samples each
for j=1:NUM_DISTS
    start = j*COUNT_PER_DIST;
    finish = (j+1)*COUNT_PER_DIST - 1;
    data(start:finish, :) = gen_data(COUNT_PER_DIST, PAIRS);
end

% Now find the centroids associated with each sample, and the "dominant"
% centroid associated with each of the NUM_DISTS lots of data generated
centroids = calculate_centroids(data, NUM_DISTS);
cluster_ids = cluster_labels(data, centroids);
dom_class = zeros(1, NUM_DISTS);

for j=1:NUM_DISTS
    start = j*COUNT_PER_DIST;
    finish = (j+1)*COUNT_PER_DIST - 1;
    relevant_rows = cluster_ids(start:finish, :);
    rel_freqs = sum(one_of_k(relevant_rows, NUM_DISTS)) / COUNT_PER_DIST;
    % Get class and make sure it's dominant (>50% of observations, let's
    % say)
    [prop, cls] = max(rel_freqs);
    assert(prop > 0.5);
    dom_class(j) = cls;
end

% Now make sure there's exactly one class label per lot of data
assert(all(sort(dom_class) == 1:NUM_DISTS));
end

function results = gen_data(count, pairs)
results = zeros(count, 2 * pairs);
centers = zeros(pairs, 2);
centers(1, :) = unifrnd(0, 100, 1, 2);
for j=2:pairs
    centers(j, :) = centers(j-1, :) + unifrnd(0, 10, 1, 2);
end
for j=1:pairs
    start = 2*(j-1)+1;
    results(:, start:start+1) = mvnrnd(centers(j, :), eye(2), count);
end
end