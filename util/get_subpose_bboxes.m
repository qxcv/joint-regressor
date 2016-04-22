function boxes = get_subpose_bboxes(fst_gt, snd_gt, subposes)
%GET_SUBPOSE_BBOXES Get [x1 y1 x2 y2] bounding boxes for each subpose
assert(false, 'Broken because I need to handle two frames at once');
assert(ismatrix(gt_locs) && size(gt_locs, 2) == 2);
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
    boxes(sp_idx) = [x1 y1 x2 y2];
end
end

