function dataset = random_trim_pairs(dataset, pair_frac)
%RANDOM_TRIM_PAIRS Shuffle dataset pairs and keep only  fraction
all_pairs = dataset.pairs;
indices = randperm(length(all_pairs));
to_keep = round(pair_frac * length(indices));
trimmed_indices = indices(1:to_keep);
dataset.pairs = all_pairs(trimmed_indices);
dataset.num_pairs = length(dataset.pairs);
end

