function cache_all_flow(all_data, pairs, cache_dir)
%CACHE_ALL_FLOW Ensure that cache contains flow for all given pairs.
% This is useful because it means that we don't have to calculate flow when
% we are doing augmentations. It also means that we can interrupt the flow
% calculation process before we get to the augmentation stage and still be
% alright.
fprintf('Filling flow cache\n');
parfor i=1:size(pairs, 1)
    fst_idx = pairs(i, 1);
    snd_idx = pairs(i, 2);
    fst = all_data(fst_idx);
    snd = all_data(snd_idx);
    cached_imflow(fst, snd, cache_dir);
end
end
