function [score, Ix, Iy, Imc, Imp] = passmsg(child, parent) % , limb_to_parent, limb_to_child)
% assert(numel(limb_to_parent) == 1 && numel(limb_to_child) == 1);
height = size(parent.score, 1);
width = size(parent.score, 2);

num_child_types = size(child.score, 3);
num_parent_types = size(parent.score, 3);
assert(num_child_types == num_parent_types);

[score0, Ix0, Iy0] = deal(zeros(height, width, num_child_types, num_parent_types));
for child_type = 1:num_child_types
    for parent_type = 1:num_parent_types
        fixed_score_map = double(child.score(:, :, child.type));
        mean_disp = squeeze(child.subpose_disps(child_type, parent_type, :));
        assert(isvector(mean_disp) && length(mean_disp) == 2);
        [score0(:, :, child_type, parent_type), Ix0(:, :, child_type, parent_type), ...
            Iy0(:, :, child_type, parent_type)] = shiftdt(...
            fixed_score_map, ...
            child.gauw{limb_to_parent}(child_type,:), ...
            mean_disp, ...
            int32(width), int32(height), 1);
        
%         [score0(:, :, child_type, parent_type), Ix0(:, :, child_type, parent_type), ...
%             Iy0(:, :, child_type, parent_type)] = distance_transform(...
%             fixed_score_map, ...
%             child.gauw{limb_to_parent}(child_type,:), ...
%             ... [child.mean_x{limb_to_parent}(child_type), child.mean_y{limb_to_parent}(child_type)], ...
%             mean_disp, ...
%             int32(width), int32(height));
        
        score0(:, :, child_type, parent_type) = ...
            score0(:, :, child_type, parent_type) ...
            + parent.pdw(limb_to_child)*parent.defMap{limb_to_child}(:, :, parent_type);
    end
end
s0_size = size(score0);
score = reshape(score0, [s0_size(1:2), num_child_types * num_parent_types]);
% XXX: Yeah, this maximisation is going to screw me over. I need to
% maximise over another axis and will probably end up making Imc, Imp, Ix
% and Iy 3D arrays (or more? IDK).
[score, Imcp] = max(score, [], 3);
[Imc, Imp] = ind2sub([num_child_types, num_parent_types], Imcp);
[Ix, Iy] = deal(zeros(height, width));
for row = 1:height
    for col = 1:width
        Ix(row, col) = Ix0(row, col, Imc(row, col), Imp(row, col));
        Iy(row, col) = Iy0(row, col, Imc(row, col), Imp(row, col));
    end
end
