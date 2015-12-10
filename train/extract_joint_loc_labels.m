function locs = extract_joint_loc_labels(paths)
%EXTRACT_JOINT_LOC_LABELS Extract joint location labels from some files.
locs = []; % This will grow; this is fine.
for i=1:length(paths)
    path = paths{i};
    labels = h5read(path, '/label');
%     unperm = reshape(labels, [size(labels, 1), 2, size(labels, 2) / 2]);
%     all_joints = permute(unperm, [1 3 2]);
%     joints_per_frame = size(all_joints, 2) / 2;
%     assert(mod(joints_per_frame, 1) == 0);
%     per_frame = reshape(all_joints, [size(ajs, 1), 2, joints_per_frame, 2]);
    locs = cat(1, locs, labels);
end
end