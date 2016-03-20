function labels = assign_label(all_data, pa, clusters, subposes, K)

% add mix field to imgs
subpose_no = numel(pa);
get_cell = @() cell(all_data.num_pairs, 1);
% Should probably change this, since I don't need a struct anymore
labels = struct(...
    'global_id', get_cell()...
);
[~, global_IDs, ~] = get_IDs(pa, K);

for pair_idx = 1:all_data.num_pairs
%     labels(pair_idx).mix_id = cell(subpose_no, 1);
    labels(pair_idx).global_id = int32(zeros([subpose_no 1]));
    this_pair = all_data.pairs(pair_idx);
    pair_idxs = [this_pair.fst this_pair.snd];
    % XXX: This is broken now that I have proper scales. I need to make
    % sure that I'm using pair.scale and cnn_window to correctly rescale
    % (and center) the joints before doing clustering.
    for subpose_idx = 1:subpose_no
        subpose = subposes(subpose_idx).subpose;
        joint_locs_mat = subpose_joint_locs(all_data.data, pair_idxs, subpose);
        assert(ismatrix(joint_locs_mat));
        % Flatten because centroids are flat
        joint_locs = joint_locs_mat(:);
        subpose_centroids = clusters{subpose_idx};
        assert(ismatrix(subpose_centroids) && size(subpose_centroids, 2) == length(joint_locs));
        diffs = bsxfun(@minus, ...
            subpose_centroids, ...
            reshape(joint_locs, [1 size(joint_locs)]));
        squares = diffs.^2;
        assert(all(size(squares) == size(subpose_centroids)));
        dists = sum(squares, 2);
        assert(isvector(dists) && length(dists) == K);
        % If I'm going to insert a 'nearest' check, I should insert it
        % here.
        [~, best_cluster] = min(dists);
        gid = global_IDs{subpose_idx}(best_cluster);
        labels(pair_idx).global_id(subpose_idx) = gid;
    end
end