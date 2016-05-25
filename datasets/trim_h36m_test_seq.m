function test_seqs = trim_h36m_test_seq(test_seqs, num_seqs, seq_size, random_seed)
%TRIM_H36M_TEST_SEQ Trim H3.6M test sequences to manageable length

% Use the static random seed for reproducibility
old_seed = rng;
seed_reset = onCleanup(@() rng(old_seed));
rng(random_seed);

assert(length(test_seqs.seqs) >= num_seqs);
seq_inds = randperm(length(test_seqs.seqs));
test_seqs.seqs = test_seqs.seqs(seq_inds(1:num_seqs));
test_seqs.seqs = cellfun(@(s) trim_seq(s, seq_size), test_seqs.seqs, ...
    'UniformOutput', false);

% Sanity checks
assert(length(test_seqs.seqs) == num_seqs);
assert(all(cellfun(@length, test_seqs.seqs) == seq_size));

% Force reset (even if seed_reset is not destroyed for whatever reason)
seed_reset.task();
end

function seq = trim_seq(seq, seq_size)
assert(length(seq) >= seq_size);
start_idx = randi(length(seq) - seq_size + 1);
end_idx = start_idx+seq_size-1;
seq = seq(start_idx:end_idx);
assert(length(seq) == seq_size);
end
