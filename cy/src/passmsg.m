function [score, Ix, Iy, Imc, Imp] = passmsg(child, parent, limb_to_parent, limb_to_child)
assert(numel(limb_to_parent) == 1 && numel(limb_to_child) == 1);
height = size(parent.score, 1);
width = size(parent.score, 2);

assert(false, 'Fix passmsg!');

num_child_parent_types = numel(child.gauid{limb_to_parent});
num_parent_child_types = numel(parent.gauid{limb_to_child});

[score0, Ix0, Iy0] = deal(zeros(height, width, num_child_parent_types, num_parent_child_types));
for c_to_p_type = 1:num_child_parent_types
    for p_to_c_type = 1:num_parent_child_types
        % XXX: need to change code so that .appMap is no longer
        % incorporated at a higher level. Instead, it should be passed in
        % here so that this function can get the right score heatmap for
        % each type combination.
        fixed_score_map = double(child.score ...
            + (child.pdw(limb_to_parent) * child.defMap{limb_to_parent}(:, :, c_to_p_type)));
        [score0(:, :, c_to_p_type,p_to_c_type), Ix0(:, :, c_to_p_type,p_to_c_type), ...
            Iy0(:, :, c_to_p_type,p_to_c_type)] = distance_transform(...
            fixed_score_map, ...
            child.gauw{limb_to_parent}(c_to_p_type,:), parent.gauw{limb_to_child}(p_to_c_type,:), ...
            [child.mean_x{limb_to_parent}(c_to_p_type), child.mean_y{limb_to_parent}(c_to_p_type)], ...
            ... XXX: I've ripped the variance thing out of here, but I should rip
            ... it out of distance_transform as well.
            [1 1], ... Constant variance for limb from child to parent
            [parent.mean_x{limb_to_child}(p_to_c_type), parent.mean_y{limb_to_child}(p_to_c_type)], ...
            [1 1], ... Also constant variance for limb from parent to child
            int32(width), int32(height));
        
        score0(:, :, c_to_p_type, p_to_c_type) = ...
            score0(:, :, c_to_p_type, p_to_c_type) ...
            + parent.pdw(limb_to_child)*parent.defMap{limb_to_child}(:, :, p_to_c_type);
    end
end
score = reshape(score0, size(score0, 1), size(score0, 2), num_child_parent_types*num_parent_child_types);
[score, Imcp] = max(score, [], 3);
[Imc, Imp] = ind2sub([num_child_parent_types, num_parent_child_types], Imcp);
[Ix, Iy] = deal(zeros(height, width));
for row = 1:height
    for col = 1:width
        Ix(row, col) = Ix0(row, col, Imc(row, col), Imp(row, col));
        Iy(row, col) = Iy0(row, col, Imc(row, col), Imp(row, col));
    end
end
