function boxes_struct = get_subpose_bboxes(all_gt, subposes, all_vis, backup_boxes)
%GET_SUBPOSE_BBOXES Get [x1 y1 x2 y2] bounding boxes for each subpose
% all_gt is a cell array of j*2 joint arrays, subposes is a struct array of
% subpose specs, all_vis is an (optional) cell array giving joint
% visibility for all_gt, backup_boxes is a set of backup bounding boxes for
% each subpose (in case of joint invisibility).

% Sanity checks
has_vis = exist('backup_boxes', 'var');
assert(~has_vis || isstruct(backup_boxes));
assert(iscell(all_gt) && (~has_vis || iscell(all_vis)), ...
    'all_gt must be a cell array of ground truths, all_vis a cell array of visibilities');
all_valid = cellfun(@(gt) ismatrix(gt) && size(gt, 2) == 2, all_gt);
assert(all(all_valid), 'Some ground truths are not j*2 matrices');

boxes = nan([length(subposes), 4]);
for sp_idx=1:length(subposes)
    joint_idxs = subposes(sp_idx).subpose;
    if has_vis
        cell_vis = cellfun(@(v) v(joint_idxs), all_vis, 'UniformOutput', false);
        joint_vis = cat(1, cell_vis{:});
        if sum(joint_vis) < 2;
            % Invalidate the subpose
            boxes(sp_idx, :) = backup_boxes.xy(sp_idx, :);
            continue;
        end
    end
    cell_locs = cellfun(@(gt) gt(joint_idxs, :), all_gt, 'UniformOutput', false);
    joint_locs = cat(1, cell_locs{:});
    assert(ismatrix(joint_locs) && size(joint_locs, 2) == 2);
    x1 = min(joint_locs(:, 1));
    y1 = min(joint_locs(:, 2));
    x2 = max(joint_locs(:, 1));
    y2 = max(joint_locs(:, 2));
    boxes(sp_idx, :) = [x1 y1 x2 y2];
end

% assert(~any(isnan(boxes(:))));

% For some reason cropscale_pos wants a struct like this. I can't be
% bothered changing it TBH.
boxes_struct = struct('xy', {boxes});
end

