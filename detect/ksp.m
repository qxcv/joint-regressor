function [bp_costs, bp_paths] = ksp(ksp_problem)
%KSP Compute shortest paths through sequence
% ksp_problem is a cell array of distance matrices giving the pairwise
% differences between all types (or whatever you want to call them) at each
% point in the sequence.

% Make sure that input is cell array
assert(iscell(ksp_problem) && length(ksp_problem) >= 1);
real_size = size(ksp_problem{1});

for mat_idx=1:length(ksp_problem)
    mat = ksp_problem{mat_idx};
    % Make sure that it's a single or a double (wouldn't make sense
    % otherwise) and then coerce to double regardless, since that's what
    % unsafe_ksp expects (it will segfault if it doesn't get that...)
    assert(isa(mat_idx, 'double') || isa(mat_idx, 'single'));
    assert(ismatrix(mat) && all(size(mat) == real_size));
    ksp_problem{mat_idx} = double(mat);
end

% Now we can actually call the mex function :P
[bp_costs, bp_paths] = unsafe_ksp(ksp_problem);
end
