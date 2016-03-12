function labels = assign_label(all_data, pa, subposes, edge_means, tsize)
K = size(edge_means, 2);
assert(all(size(edge_means) == [length(pa) K K 2]));

% add mix field to imgs
subpose_no = numel(pa);
get_cell = @() cell(all_data.num_pairs, 1);
labels = struct('mix_id', get_cell(), ...
    'global_id', get_cell()... , ...
... XXX: I've commented out the 'near' and 'invalid' things because I
... haven't figured out how they apply to my model.
... 'near', get_cell(), ...
... 'invalid', get_cell()...
);
[~, global_IDs, ~] = get_IDs(pa, K);

for pair_idx = 1:all_data.num_pairs
    labels(pair_idx).mix_id = cell(subpose_no, 1);
    pair = all_data.pairs(pair_idx, :);
    for subpose_idx = 1:subpose_no
        parent_idx = pa(subpose_no);
        if parent_idx == 0
            % Only bother adding mix IDs for child subposes
            continue
        end
        child_locs = subpose_joint_locs(all_data.data, pair, subpose);
        labels(pair_idx).mix_id{part_idx} = int32();
% XXX: Delete the stuff below once you've reviewed it thoroughly enough to
% understand what it does.
%         nbh_N = length(nbh_IDs{part_idx});
%         labels(pair_idx).near{part_idx} = cell(nbh_N, 1);
%         invalid = false;
%         for n = 1:nbh_N
%             % find nearest
%             nbh_idx = nbh_IDs{p}(n);
%             if ( isfield(all_data, 'invalid') && (all_data(pair_idx).invalid(p) || all_data(pair_idx).invalid(nbh_idx)) )
%                 invalid = true;
%             end
%             cluster_num = numel(clusters{part_idx}{n});
%             centers = zeros(cluster_num, 2);
%             for k = 1:cluster_num
%                 centers(k,:) = clusters{part_idx}{n}(k).center;
%             end
%             % XXX: Re-introduce this once scaling is working
%             % rp = norm_rp(all_data(pair_idx), p, nbh_idx, tsize);
%             
%             dists = bsxfun(@minus, centers, rp);
%             dists = sqrt(sum(dists .^ 2, 2));
%             
%             [~,id] = min(dists,[],1);
%             labels(pair_idx).mix_id{part_idx}(n) = int32(id);
%             labels(pair_idx).near{p}{n} = dists < 3 * dists(id);
%         end
%         % invalid
%         labels(pair_idx).invalid(p) = invalid;
    end
    assert(false, 'You need to fix the rest of assign_label');
    labels(pair_idx).global_id = int32(mix2global(labels(pair_idx).mix_id, global_IDs));
end

function global_id = mix2global(mix_id, global_IDs)
p_no = numel(mix_id);
global_id = zeros(p_no, 1);
for p = 1:p_no
    mixs = ones(3, 1);
    for ii = 1:numel(mix_id{p})
        mixs(ii) = mix_id{p}(ii);
    end
    global_id(p) = global_IDs{p}(mixs(1),mixs(2),mixs(3));
end
