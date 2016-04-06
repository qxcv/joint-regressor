function pose = boxes2pose(detected_boxes, centroids, centroid_size, subpose_idxs, num_joints)
%BOXES2POSE Recover pose from subpose detections
% Uses type information to get actual joint locations instead of just
% center and type of each subpose
%
% detected_boxes: 1d struct array with .boxes and .types fields (like
%                 detect returns)
% centroids: carefully reformatted centroids from buildmodel (should be
%            saved in model produced by train_model)
% centroid_size: scale at which centroids were clustered. Probably the CNN
%                window size.
% subpose_idxs: cell array giving original joint indices of parts
% num_joints: number of joints in complete skeleton

all_pose_locs = nan([2 * num_joints, 2, length(subpose_idxs)]);
for subpose_idx=1:length(subpose_idxs)
    % box has format [x1 y1 x2 y2]
    bbox = detected_boxes.boxes{subpose_idx};
    bbox_wh = bbox(3:4) - bbox(1:2);
    assert(abs(bbox_wh(1) - bbox_wh(2)) < 1e-5);
    bbox_size = mean(bbox_wh);
    type = detected_boxes.types{subpose_idx};
    centroid = centroids{subpose_ix}{type};
    joint_locs = centroid * bbox_size / centroid_size + bbox(1:2);
    assert(isvector(joint_locs) && size(joint_locs, 2) == 2);
    all_pose_locs(:, :, subpose_idx) = joint_locs;
end

% Now average-out shared joints
mean_pose_locs = nan([2 * num_joints, 2]);
for joint_idx=1:length(num_joints)
    pred_mask = find(~isnan(squeeze(all_pose_locs(joint_idx, 1, :))));
    assert(isvector(pred_mask));
    if isempty(pred_mask)
        continue
    end
    % TODO: Weight mean according to rscore (higher rscore should count
    % more)
    mean_pose_locs(joint_idx, :) = squeeze(mean(...
        all_pose_locs(joint_idx, :, pred_mask), 3));
end

% Split into frames
pose = {mean_pose_locs(1:num_joints, :), mean_pose_locs(num_joints+1:end, :)};
end