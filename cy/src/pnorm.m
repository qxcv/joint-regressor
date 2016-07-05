function normed = pnorm(in_vec)
%PNORM Normalise nd array of probabilties
if isempty(in_vec)
    normed = in_vec;
    return
end

assert(isnumeric(in_vec));
flat_in = flat(in_vec);
assert(all(flat_in >= 0), 'Can''t have negative probabilities');
total = sum(flat_in);
assert(isscalar(total) && isfinite(total));
if total > 0
    normed = in_vec ./ total;
else
    normed = ones(size(in_vec)) ./ numel(in_vec);
end
assert(abs(sum(flat(normed)) - 1) < 1e-5, 'Output not normalised');
end

