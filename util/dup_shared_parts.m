function shared_parts = dup_shared_parts(old_sp, num_joints)
%DUP_SHARED_PARTS Convert single-frame shared parts to two-frame parts
% I chose a stupid format for conf.shared_parts which requires me to name
% the indices of joints in the cat(1, frame_1.joint_locs,
% frame_2.joint_locs) array instead of just the joint_locs array, so I'm
% using this function instead of rewriting everything that uses
% shared_parts. Ugh.
assert(iscell(old_sp) && isscalar(num_joints));
shared_parts = cell([1 length(old_sp)]);
for sp=1:length(old_sp)
    if isempty(old_sp{sp})
        shared_parts{sp} = old_sp{sp};
        continue
    else
        assert(length(old_sp{sp}) == 2);
        f1_src = old_sp{sp}{1};
        f1_dest = old_sp{sp}{2};
        assert(isvector(f1_src) && isvector(f1_dest) ...
            && length(f1_src) == length(f1_dest));
        f2_dest = f1_dest + num_joints;
        f2_src = f1_src + num_joints;
        shared_parts{sp} = {[f1_src, f2_src], [f1_dest, f2_dest]};
    end
end
end
