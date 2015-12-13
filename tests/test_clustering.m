function test_clustering
%TEST_CLUSTERING Test joint labels clustering functions
COUNT_PER_DIST = 100;
NUM_DISTS = 100;
PAIRS = 100;
data = zeros(NUM_DISTS * COUNT_PER_DIST, PAIRS * 2);

% Produce NUM_DISTS lots of data, consisting of COUNT_PER_DIST samples each
temp_data = cell(1, NUM_DISTS);
fprintf('Generating data\n');
parfor j=1:NUM_DISTS
    temp_data{j} = gen_data(COUNT_PER_DIST, PAIRS);
end

for j=1:NUM_DISTS
    start = j*COUNT_PER_DIST;
    finish = (j+1)*COUNT_PER_DIST - 1;
    data(start:finish, :) = temp_data{j};
end

% Now find the centroids associated with each sample, and the "dominant"
% centroid associated with each of the NUM_DISTS lots of data generated
centroids = calculate_centroids(data, NUM_DISTS);
cluster_ids = cluster_labels(data, centroids);
dom_class = zeros(1, NUM_DISTS);

fprintf('Producing output vectors\n');
parfor j=1:NUM_DISTS
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

fprintf('Final assertion\n');
% Make sure that we ended up with at least 0.9 * NUM_DISTS unique dominant
% clusters.
dom_class_prop = length(unique(dom_class)) / NUM_DISTS;
fprintf('Unique dominant class proportion is %f\n', dom_class_prop);
assert(dom_class_prop > 0.9);
fprintf('Success!\n');
end

function results = gen_data(count, pairs)
results = zeros(count, 2 * pairs);
centers = zeros(pairs, 2);
centers(1, :) = unifrnd(0, 100, 1, 2);
% Calculate centers with random walk
for j=2:pairs
    centers(j, :) = centers(j-1, :) + unifrnd(0, 100, 1, 2);
end
% Now assign data
for j=1:pairs
    start = 2*(j-1)+1;
    results(:, start:start+1) = mvnrnd(centers(j, :), 0.01 * eye(2), count);
end
end