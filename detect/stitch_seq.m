function pose_detections = stitch_seq(pair_detections, num_stitch_dets, ...
    weights, valid_parts)
%STITCH_SEQ Turn biposelet detection seq into pair seq
num_bps = length(pair_detections);
assert(num_bps > 1, 'Need to implement one-BP trivial case');

% Formulate and solve DP problem to choose which biposelet to use in each
% set of biposelets
ksp_size = num_bps-1;
ksp_problem = cell([1 ksp_size]);
parfor bp_pair_idx=1:ksp_size
    % Distance matrix first
    bp1_poses = pair_detections(bp_pair_idx).recovered;
    bp2_poses = pair_detections(bp_pair_idx+1).recovered;
    first_set = cellfun(@(p) p{2}, bp1_poses, 'UniformOutput', false);
    first_set = first_set(1:min(end, num_stitch_dets));
    first_set = sanitise_poses(first_set, valid_parts);
    second_set = cellfun(@(p) p{1}, bp2_poses, 'UniformOutput', false);
    second_set = second_set(1:min(end, num_stitch_dets));
    second_set = sanitise_poses(second_set, valid_parts);
    lengths = cellfun(@length, {first_set, second_set});
    assert(all(lengths <= num_stitch_dets));
    cost_mat = weights.dist .* pose_distance_matrix(first_set, second_set); %#ok<PFBNS>
    
    % Now add rscores; will be negated, since higher rscore = better but
    % lower dist = better
    % bp{1,2}_rscores will be row vectors
    bp1_rscores = pair_detections(bp_pair_idx).rscores;
    bp1_rscores = bp1_rscores(1:min(end, num_stitch_dets));
    % Add bp1 scores to each row
    cost_mat = bsxfun(@minus, cost_mat, weights.rscore .* bp1_rscores');
    if bp_pair_idx == num_bps-1
        % Add second frame scores to each column if last biposelet
        bp2_rscores = pair_detections(bp_pair_idx+1).rscores; %#ok<PFBNS>
        bp2_rscores = bp2_rscores(1:min(end, num_stitch_dets));
        cost_mat = bsxfun(@minus, cost_mat, weights.rscore .* bp2_rscores);
    end
    
    ksp_problem{bp_pair_idx} = cost_mat;
end

[~, bp_paths] = ksp(ksp_problem);
assert(length(bp_paths) == num_bps);

% Grab poses from chosen sequence of biposelets
out_len = num_bps + 1;
pose_detections = cell([1 out_len]);
for out_idx=1:out_len
    switch out_idx
        case 1
            % In first frame, grab first frame pose of first chosen
            % biposelet
            chosen_bp = bp_paths(1);
            best_pose = pair_detections(1).recovered{chosen_bp}{1};
        case out_len
            % Analogously, in the last frame we get the second frame pose
            % of the last chosen poselet
            chosen_bp = bp_paths(num_bps);
            best_pose = pair_detections(num_bps).recovered{chosen_bp}{2};
        otherwise
            % If not on one of the ends, we grab the two relevant chosen
            % biposelets and average their predictions for the current
            % frame
            bp1 = bp_paths(out_idx-1);
            bp2 = bp_paths(out_idx);
            p1 = pair_detections(out_idx-1).recovered{bp1}{2};
            p2 = pair_detections(out_idx).recovered{bp2}{1};
            % TODO: Weight by rscore
            best_pose = 0.5 .* (p1 + p2);
    end
    pose_detections{out_idx} = best_pose;
end
end

function poses = sanitise_poses(poses, valid_inds)
% Strip out invalid joints. Invalid joints are set to NaN, so they can
% seriously screw up distance calculations.
assert(iscell(poses));
parfor i=1:length(poses)
    assert_inds_consistent(poses{i}, valid_inds);
    poses{i} = poses{i}(valid_inds);
end
end

function assert_inds_consistent(pose, valid_inds)
% Make sure all invalid joints are nan and no invalid ones are
valid_mask = false([1 size(pose, 1)]);
valid_mask(valid_inds) = true;
assert(all(flat(isnan(pose(~valid_mask, :)))));
assert(~any(flat(isnan(pose(valid_mask, :)))));
end
