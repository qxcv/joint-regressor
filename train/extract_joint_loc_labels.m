function locs = extract_joint_loc_labels(paths)
%EXTRACT_JOINT_LOC_LABELS Extract joint location labels from some H5s.
locs = []; % This will grow; this is fine.
for i=1:length(paths)
    path = paths{i};
    % Remember the transpose!
    labels = h5read(path, '/joints')';
    classes = boolean(h5read(path, '/class'));
    fprintf('[extracting labels] Label size: %i, classes size: %i, valid classes: %i\n', ...
        size(labels, 1), size(classes, 2), sum(classes));
    assert(ismatrix(labels));
    assert(isvector(classes));
    assert(size(labels, 1) == length(classes));
    % Only use locations with a person visible (so class=1)
    locs = cat(1, locs, labels(classes, :));
end
% Some other code which I don't want to delete from this file (even though
% it would be easy to find with git...). This code converts a set of labels
% into a vector of vectors of vectors, where the innermost axis is for (x,
% y), the second-to-innermost indexes specific joints (e.g. left shoulder
% or right wrist) and the outer index indexes the sample number (so axis
% are like [sample no., joint, coordinate axis]).
%     unperm = reshape(labels, [size(labels, 1), 2, size(labels, 2) / 2]);
%     all_joints = permute(unperm, [1 3 2]);
%     joints_per_frame = size(all_joints, 2) / 2;
%     assert(mod(joints_per_frame, 1) == 0);
%     per_frame = reshape(all_joints, [size(ajs, 1), 2, joints_per_frame, 2]);
end