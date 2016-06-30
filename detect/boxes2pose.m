function pose = boxes2pose(detection, biposelets, biposelet_scale, ...
    subposes, num_joints)
%BOXES2POSE Recover pose from subpose detections
% Uses type information to get actual joint locations instead of just
% center and type of each subpose
%
% detected_boxes: 1d struct array with .boxes and .types fields (like
%                 detect returns)
% biposelets: carefully reformatted centroids from buildmodel (should be
%             saved in model produced by train_model)
% biposelet_scale: scale at which centroids were clustered. Probably the
%                  CNN window size.
% subpose_idxs: cell array giving original joint indices of parts
% num_joints: number of joints in complete skeleton

assert(all(biposelet_scale == biposelet_scale(1)));
biposelet_scale = biposelet_scale(1);
assert(isscalar(biposelet_scale));

all_pose_locs = nan([2 * num_joints, 2, length(subposes)]);
for subpose_idx=1:length(subposes)
    % Grab bbox (has format [x1 y1 x2 y2])
    bbox = detection.boxes{subpose_idx};
    bbox_wh = bbox(3:4) - bbox(1:2);
    assert(abs(bbox_wh(1) - bbox_wh(2)) < 1e-4);
    bbox_size = mean(bbox_wh);
    
    % Recover centroid and realign to detection center and scale
    type = detection.types{subpose_idx};
    centroid = unflatten_coords(biposelets{subpose_idx}(type, :));
    joint_locs = bsxfun(@plus, centroid * bbox_size / biposelet_scale, bbox(1:2));
    
    % Now save locations into relevant joint blocks
    assert(ismatrix(joint_locs) && size(joint_locs, 2) == 2);
    joint_idxs = subposes(subpose_idx).subpose;
    assert(isrow(joint_idxs));
    pair_idxs = [joint_idxs, joint_idxs + num_joints];
    all_pose_locs(pair_idxs, :, subpose_idx) = joint_locs;
end

% Now average-out shared joints
mean_pose_locs = nan([2 * num_joints, 2]);
for joint_idx=1:2*num_joints
    pred_mask = find(~isnan(squeeze(all_pose_locs(joint_idx, 1, :))));
    assert(isvector(pred_mask));
    if isempty(pred_mask)
        continue
    end
    % TODO: Need to look back at CNN output and figure out what the actual
    % distribution over types was at this location. Should probably average
    % over types (or maybe average over types "near" the detected one) to
    % get a better score. May have to be careful not to average over
    % everything, lest I mix incompatible types for neighbouring parts.
    mean_pose_locs(joint_idx, :) = ...
        squeeze(mean(all_pose_locs(joint_idx, :, pred_mask), 3));
end

% Split into frames
pose = {single(mean_pose_locs(1:num_joints, :)), single(mean_pose_locs(num_joints+1:end, :))};
end
