function [all_classes, locs] = extract_joint_loc_labels(paths, poselets)
%EXTRACT_JOINT_LOC_LABELS Extract joint location labels from some H5s.
locs = cell([1 length(poselets)]);
all_classes = [];

for path_idx=1:length(paths)
    path = paths{path_idx};
    % Remember the transpose!
    [some_ones, classes] = max(h5read(path, '/class'));
    classes = classes';
    assert(isvector(classes));
    assert(all(some_ones == 1));
    
    all_classes = cat(1, all_classes, classes);
    assert(isvector(all_classes));
    
    for poselet_num=1:length(poselets)
        labels = h5read(path, ['/' poselets(poselet_num).name]);
        labels = labels';
        assert(ismatrix(labels));
        assert(size(labels, 1) == length(classes));
        
        if isempty(locs{poselet_num})
            locs{poselet_num} = labels;
        else
            locs{poselet_num} = cat(1, locs{poselet_num}, labels);
        end
        
        assert(ismatrix(locs{poselet_num}));
    end
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
%     per_frame = reshape(all_joints, [size(ajs, 1), 2, joints_per_frame, 2]);end