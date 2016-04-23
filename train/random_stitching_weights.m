function weights = random_stitching_weights(count)
%RANDOM_STITCHING_WEIGHTS Generate some random weights for stitching

% TODO: Will probably need a configurable distribution or something.

% At the moment we have a single degree of freedom, since the stitching
% mechanism is invariant to weight scale. Thus, I'm setting to score weight
% to "1" in each instance and just choosing the distance ratio.
min_dist_exp = -6;
max_dist_exp = 1;
samples = unifrnd(min_dist_exp, max_dist_exp, 1, count);
dist_weights = 10.^samples;
weights = struct('rscore', num2cell(ones([1, count])), ...
    'dist', num2cell(dist_weights));
end

