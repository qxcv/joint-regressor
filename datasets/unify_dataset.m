function unified = unify_dataset(frame_data, pairs, ds_name)
%UNIFY_DATASET Make unified struct representing dataset
% This will probably do more later.
assert(ismatrix(pairs));
assert(size(pairs, 2) == 2);
unified.data = frame_data;
fst_cell = num2cell(pairs(:, 1))';
snd_cell = num2cell(pairs(:, 2))';
% XXX: .pairs should have the following attributes:
% 1) fst (gives index of first frame in each pair)
% 2) snd (gives index of second frame in each pair)
% 3) scale_x (see train.m)
% 4) scale_y (again, train.m has the answers)
unified.pairs = struct('fst', fst_cell, 'snd', snd_cell);
unified.num_pairs = length(pairs);
unified.name = ds_name;
end

