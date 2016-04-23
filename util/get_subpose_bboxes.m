function boxes_struct = get_subpose_bboxes(fst_gt, snd_gt, subposes)
%GET_SUBPOSE_BBOXES Get [x1 y1 x2 y2] bounding boxes for each subpose
assert(ismatrix(fst_gt) && size(fst_gt, 2) == 2);
assert(ismatrix(snd_gt) && size(snd_gt, 2) == 2);
boxes = nan([length(subposes), 4]);
for sp_idx=1:length(subposes)
    joint_idxs = subposes(sp_idx).subpose;
    fst_joint_locs = fst_gt(joint_idxs, :);
    snd_joint_locs = snd_gt(joint_idxs, :);
    joint_locs = cat(1, fst_joint_locs, snd_joint_locs);
    assert(ismatrix(joint_locs) && size(joint_locs, 2) == 2);
    x1 = min(joint_locs(:, 1));
    y1 = min(joint_locs(:, 2));
    x2 = max(joint_locs(:, 1));
    y2 = max(joint_locs(:, 2));
    boxes(sp_idx, :) = [x1 y1 x2 y2];
end
assert(~any(isnan(boxes(:))));

% For some reason cropscale_pos wants a struct like this. I can't be
% bothered changing it TBH.
boxes_struct = struct('xy', {boxes});
end

