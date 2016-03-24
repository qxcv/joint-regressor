function labels = assign_label(all_data, pa, clusters, subposes, K, cnn_window)
%ASSIGN_LABEL Associate training data with subpose clusters
% add mix field to imgs
subpose_no = numel(pa);
get_cell = @() cell(all_data.num_pairs, 1);
labels = struct(...
    'global_id', get_cell(),...
    'near', get_cell()...
);
[~, global_IDs, ~] = get_IDs(pa, K);

for pair_idx = 1:all_data.num_pairs
    labels(pair_idx).global_id = int32(zeros([subpose_no 1]));
    labels(pair_idx).near = cell([subpose_no 1]);
    this_pair = all_data.pairs(pair_idx);
    d1 = all_data.data(this_pair.fst);
    d2 = all_data.data(this_pair.snd);
    for subpose_idx = 1:subpose_no
        subpose = subposes(subpose_idx).subpose;
        normed_locs = norm_subpose(d1, d2, this_pair.scale, cnn_window, subpose);
        % Flatten because centroids are flat
        joint_locs = normed_locs(:);
        subpose_centroids = clusters{subpose_idx};
        assert(ismatrix(subpose_centroids) && size(subpose_centroids, 2) == length(joint_locs));
        diffs = bsxfun(@minus, ...
            subpose_centroids, ...
            reshape(joint_locs, [1 size(joint_locs)]));
        squares = diffs.^2;
        assert(all(size(squares) == size(subpose_centroids)));
        dists = sum(squares, 2);
        assert(isvector(dists) && length(dists) == K);
        
        % First find and record the nearest cluster
        [best_dist, best_cluster] = min(dists);
        gid = global_IDs{subpose_idx}(best_cluster);
        labels(pair_idx).global_id(subpose_idx) = gid;
        
        % Now record clusters which are close, but not necessarily the
        % nearest (used for poselet supervision in SVM training code).
        all_near = find(dists <= 1.3 * best_dist);
        assert(~isempty(all_near));
        labels(pair_idx).near{subpose_idx} = all_near;
    end
end