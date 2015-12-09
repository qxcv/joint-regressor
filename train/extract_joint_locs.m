function locs = extract_joint_locs(paths)
%EXTRACT_JOINT_LOCS Extract the joint locations from a list of files.
locs = []; % This will grow; this is fine.
for i=1:length(paths)
    path = paths{i};
    labels = h5read(path, '/label');
    % Labels should be a matrix like this
    % Row 1 (datum 1): [x^(1)_1 y^(1)_1 x^(1)_2 y^(1)_2 ... x^(1)_j y^(1)_j]
    % Row 2 (datum 2): [x^(2)_1 y^(2)_1 x^(2)_2 y^(2)_2 ... x^(2)_j y^(2)_j]
    % Row 3 (datum 3): ....
    unperm = reshape(labels, [size(labels, 1), 2, size(labels, 2) / 2]);
    all_joints = permute(unperm, [1 3 2]);
    joints_per_frame = size(all_joints, 2) / 2;
    assert(mod(joints_per_frame, 1) == 0);
    per_frame = reshape(all_joints, [size(ajs, 1), 2, joints_per_frame, 2]);
    loc = cat(1, loc, per_frame);
end
end