function distance_matrix = pose_distance_matrix(f1_poses, f2_poses)
%BIPOSELET_DISTANCE_MATRIX Compute distance matrix between possible poses
% Think matrix will be like distance_matrix(f1_pose, f2_pose) = distance
% between f1_poses(f1_pose) and f2_poses(f2_pose)
lf1 = length(f1_poses);
lf2 = length(f2_poses);
% Interestingly, replacing lf1 and lf2 with length(f1_poses) and
% length(f2_poses), respectively, was causing Matlab to die before because
% it "could not classify" the parfor loop.
distance_matrix = zeros([lf1, lf2]);
parfor f1_pose=1:lf1
    for f2_pose=1:lf2
        p1 = f1_poses{f1_pose};
        p2 = f2_poses{f2_pose}; %#ok<PFBNS>
        distance_matrix(f1_pose, f2_pose) = pose_dist_func(p1, p2);
    end
end
end

function dist = pose_dist_func(p1, p2)
% Assuming p1 and p2 are j*2 matrices. This is slow but now you can play
% around with the second argument to norm() easily.
dist = sum(arrayfun(@(rid) norm(p2(rid, :) - p1(rid, :)), 1:size(p1, 1)));
end
