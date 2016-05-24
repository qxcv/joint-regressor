function [orig_joints] = spec_trans_back(joints, trans_spec, orig_num_joints)
%SPEC_TRANS_BACK Translate joints back into source form using trans_spec
% Fills joints with no equivalent with nans
single_joints = cellfun(@isscalar, {trans_spec.indices});
orig_idxs = [trans_spec(single_joints).indices];
assert(length(unique(orig_idxs)) == length(orig_idxs), ...
    'trans_spec.indices not injective');
orig_joints = nan([orig_num_joints, 2]);
orig_joints(~~single_joints, :) = joints(orig_idxs, :);
end
