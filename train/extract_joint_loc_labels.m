function [all_classes, locs] = extract_joint_loc_labels(paths, subposes)
%EXTRACT_JOINT_LOC_LABELS Extract joint location labels from some H5s.
locs = cell([1 length(subposes)]);
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
    
    for subpose_num=1:length(subposes)
        labels = h5read(path, ['/' subposes(subpose_num).name]);
        labels = labels';
        assert(ismatrix(labels));
        assert(size(labels, 1) == length(classes));
        
        if isempty(locs{subpose_num})
            locs{subpose_num} = labels;
        else
            locs{subpose_num} = cat(1, locs{subpose_num}, labels);
        end
        
        assert(ismatrix(locs{subpose_num}));
    end
end