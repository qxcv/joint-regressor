function unified = unify_dataset(frame_data, pairs, ds_name)
%UNIFY_DATASET Make unified struct representing dataset
% This will probably do more later.
assert(ismatrix(pairs));
assert(size(pairs, 2) == 2);
unified.data = frame_data;
unified.pairs = pairs;
unified.num_pairs = length(pairs);
unified.name = ds_name;
end

