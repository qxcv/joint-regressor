function all_distances = get_distances(dataset)
%GET_DISTANCES Get all inter-frame distances in datset.
num_pairs = length(dataset.pairs);
all_distances = zeros([1 num_pairs]);
for pair_idx=1:num_pairs
    pair = dataset.pairs(pair_idx);
    fst = dataset.data(pair.fst);
    snd = dataset.data(pair.snd);
    all_distances(pair_idx) = mean_dists(fst.joint_locs, snd.joint_locs);
end
end

