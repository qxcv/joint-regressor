function [best_1, best_2] = best_effort_subposes(pair, d1, d2, subposes, ...
    unflat_biposelets, biposelets, scales, cnnpar)
%BEST_EFFORT_SUBPOSES Best-effort approximation of skeleton using model
% Works by splitting the input pose into subpose types and locations in a
% hypothetical heatmap, then reversing the process to recover joint
% locations from those subpose types and locations. The resulting skeleton
% will be good for producing an upper bound on the accuracy of the subpose
% model.
%
% This function emulates multi-scale testing, discretisation due to CNN
% downsampling, etc.

attempts = cell([1 length(scales)]);
attempt_costs = nan([1 length(scales)]);
window = cnnpar.window(1);
% Amount by which the image would have been rescaled to make the "true
% scale" the central one (i.e. the imresize() argument from preprocessing)
scale_factor = window / pair.scale;

for scale_idx=1:length(scales)
    total_scale = scales(scale_idx) * scale_factor;
    assert(isscalar(total_scale));
    to_scale = @(j) (j - 1) * total_scale + 1;
    tp1 = to_scale(d1.joint_locs);
    tp2 = to_scale(d2.joint_locs);
    scaled_bp = cellfun(to_scale, unflat_biposelets, 'UniformOutput', false);

    % sp_locs gives top left corner of biposelet
    [sp_types, sp_locs, ~] = get_subposes(tp1, tp2, subposes, scaled_bp);
    fake_pyra.pad = 0;
    map_locs = image_to_map_loc(sp_locs, fake_pyra, cnnpar);
    boxes = to_boxes(map_locs, sp_types, cnnpar, total_scale);
    both = boxes2pose(boxes, biposelets, window, subposes, size(tp1, 1));
    % TODO: Do I have to rescale this? Will probably have to manually check
    % that this is doing the right thing.
    attempts{scale_idx} = both;
    attempt_costs(scale_idx) = sum(flat((tp1 - both{1}).^2 + (tp2 - both{2}).^2));
end

assert(~any(isnan(attempt_costs)));
assert(~any(cellfun(@isempty, attempts)));

[~, idx] = min(attempt_costs);
best_effort = attempts{idx};
best_1 = best_effort{1};
best_2 = best_effort{2};
end

function rv = to_boxes(map_locs, types, cnnpar, scale_factor)
% Need to turn map locations into boxes, somehow? What should the size of
% the boxes even be? Where should their centres lie?
num_types = length(types);
rv.boxes = cell([1 num_types]);
scale = cnnpar.step / scale_factor;
det_side = cnnpar.window(1) / cnnpar.step;
for sp_idx=1:num_types
    x = map_locs(sp_idx, 1);
    y = map_locs(sp_idx, 2);
    % scale emulates pyra.scale, det_side is cnn window / cnn step
    rv.boxes{sp_idx} = get_subpose_box(x, y, det_side, 0, scale);
end
rv.types = num2cell(types);
end
