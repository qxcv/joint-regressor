function cache_all_flow(dataset, cache_dir)
%CACHE_ALL_FLOW Ensure that cache contains flow for all given pairs.
% This is useful because it means that we don't have to calculate flow when
% we are doing augmentations. It also means that we can interrupt the flow
% calculation process before we get to the augmentation stage and still be
% alright.
pairs = dataset.pairs;
all_data = dataset.data;
parfor i=1:dataset.num_pairs
    idxs = pairs(i);
    fst = all_data(idxs.fst); %#ok<PFBNS>
    snd = all_data(idxs.snd);
    cached_imflow(fst, snd, cache_dir);
end
end
