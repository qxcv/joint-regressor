function shared_parts = make_shared_parts(subposes, subpose_pa)
%MAKE_SHARED_PARTS Automatically build shared parts array
% conf.shared_parts is used to calculate mean displacements between
% neighbouring subposes of different types before doing message passing.

shared_parts = cell([1 length(subposes)]);

for sp_idx=1:length(subposes)
    pa_idx = subpose_pa(sp_idx);
    assert(pa_idx < sp_idx, 'Nodes should be toposorted (with root=0)');
    
    if ~pa_idx || sp_idx == 1
        assert(~pa_idx && sp_idx == 1, 'Root must be first');
        shared_parts{sp_idx} = {};
        continue
    end
    
    ch_joints = subposes(sp_idx).subpose;
    pa_joints = subposes(pa_idx).subpose;
    
    % Find indices withthin subpose joints array of shared joints
    [inter, ch_shared, pa_shared] = intersect(ch_joints, pa_joints);
    assert(~isempty(inter), 'Parent and child must share at least one joint');
    
    % Need to add indices of shared nodes in first frame and in second
    % frame (assuming that joint locations for each frame are concatenated
    % together row-wise, with first frame on top).
    both_ch = [ch_shared, ch_shared + length(ch_joints)];
    both_pa = [pa_shared, pa_shared + length(pa_joints)];
    
    shared_parts{sp_idx} = {both_ch, both_pa};
end
end
