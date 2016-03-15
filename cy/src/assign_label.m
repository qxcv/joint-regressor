function labels = assign_label(all_data, pa, clusters, subposes, edge_means, tsize)
K = size(edge_means, 2);
assert(all(size(edge_means) == [length(pa) K K 2]));

% add mix field to imgs
subpose_no = numel(pa);
get_cell = @() cell(all_data.num_pairs, 1);
labels = struct(...
    'global_id', get_cell()... , ...
... XXX: I've commented out the near, mix_id and invalid things because I
... haven't figured out how they apply to my model.
... 'mix_id', get_cell(), ...
... 'near', get_cell(), ...
... 'invalid', get_cell()...
);
[~, global_IDs, ~] = get_IDs(pa, K);

for pair_idx = 1:all_data.num_pairs
%     labels(pair_idx).mix_id = cell(subpose_no, 1);
    labels(pair_idx).global_id = int32(zeros([subpose_no 1]));
    this_pair = all_data.pairs(pair_idx);
    pair_idxs = [this_pair.fst this_pair.snd];
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
        % XXX: This was going to be mix_id stuff, but now I'm not sure
        % whether it's even necessary in my case. Same goes for 'nearest'.
%         parent_idx = pa(subpose_no);
%         if parent_idx == 0
%             % Only bother adding mix IDs for child subposes
%             continue
%         end
%         child_locs = subpose_joint_locs(all_data.data, pair, subpose);
%         child_locs = subpose_joint_locs(all_data.data, pair, subpose);
%         labels(pair_idx).mix_id{part_idx} = int32();
    end
end