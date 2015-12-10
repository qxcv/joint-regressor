function rvs = get_stacks(d1, d2, poselet, cache_dir, cnn_window, flips, rotations, scales, randtrans)
%GET_STACKS Get the image/flow stacks for a given data pair and set of
%transformations.
% d1: First datum
% d2: Second datum
% flips: List of booleans ([0], [1] or 0:1) telling us whether we should
% includes flips (1) and/or originals (0).
% rotations: List giving how many degrees to rotate by
% scales: magnifications to use (e.g. 2.0 will double the size of the
% skeleton, pushing most of it out of the frame, whilst 0.5 will halve the
% size of the skeleton)
% randtrans: If this is greater than zero, and there's some space between
% the bounding box of the pose and the edge of the image, then the pose box
% will be translated randomly around in that space randtrans times.
% Otherwise, the pose will be centered perfectly in the crop.
% rvs: struct array with .labels (joint locations) and .stack (full input
% data) attributes.
im1 = readim(d1);
im2 = readim(d2);
flow = cached_imflow(d1, d2, cache_dir);
assert(all(size(im1) == size(im2)));
assert(size(im1, 1) == size(flow, 1) && size(im1, 2) == size(flow, 2));
assert(size(flow, 3) == 2);
assert(size(im1, 3) == 3);
% Concatenate along channels
stacked = cat(3, norm_im(im1), ...
                 norm_im(im2), ...
                 norm_flow(flow));
% Now join joints
all_joints = cat(1, d1.joint_locs, d2.joint_locs);
poselet_indices = [poselet, poselet + length(d1.joint_locs)];

% We'll store rvs in a struct array with a "stack" and "joints" attribute
% for each entry.
current_idx = 1;

for flip=flips
    %% 1) Flip
    flip_joints = all_joints;
    flip_stack = stacked;
    if flip
        % Reverse joints
        flip_joints(:, 1) = size(im1, 2) - flip_joints(:, 1) + 1;
        % Swap indices 2-4 with indices 5-7 (left side <-> right side)
        % FIXME: This code will break if I use a data set other than
        % FLIC
        num_joints = size(d1.joint_locs, 1);
        assert(num_joints == size(d2.joint_locs, 1));
        flip_joints(1:num_joints, :) = flip_lr(flip_joints(1:num_joints, :));
        flip_joints(num_joints+1:end, :) = flip_lr(flip_joints(num_joints+1:end, :));
        % Reverse images
        flip_stack = flip_stack(:, end:-1:1, :);
        % Flip flow
        flip_stack(:, :, 7) = -flip_stack(:, :, 7);
    end

    for rotate=rotations
        %% 2) Rotate
        rot_stack = flip_stack;
        rot_joints = flip_joints;
        if rotate ~= 0
            rot_joints = map_rotate_points(rot_joints, rot_stack, rotate, 'ori2new');
            rot_stack = improtate(rot_stack, rotate, 'bilinear');
            rot_mat = [cosd(rotate), -sind(rotate); sind(rotate), cosd(rotate)];
            flow = rot_stack(:, :, 7:8);
            flat_flow = reshape(flow, [size(flow, 1) * size(flow, 2), 2]);
            flat_flow = (rot_mat * flat_flow.').';
            rot_stack(:, :, 7:8) = reshape(flat_flow, size(flow));
        end

        %% 3) Get bounding box for joint
        maxes = max(rot_joints(poselet_indices), [], 1);
        mins = min(rot_joints(poselet_indices), [], 1);
        % Always crop a square patch
        pose_side = max(maxes - mins);
        % box_center is (x, y)
        box_center = mins + (maxes - mins) ./ 2;

        for scale=scales
            %% 4) Scale box
            side = round(pose_side / scale);

            % We repeat this translation process for each random
            % translation
            total_trans = max(randtrans, 1);
            wiggle_room = side - pose_side;
            if wiggle_room <= 1
                % Don't translate at scale 1 or below (effectively)
                total_trans = 1;
            end;
            
            for unused=1:total_trans
                %% 5) Translate box
                trans_box_center = box_center;
                if randtrans > 0 && wiggle_room > 1
                    trans_amount = wiggle_room * rand(1, 2) - wiggle_room / 2;
                    trans_box_center = trans_box_center + trans_amount;
                end
                box = round(cat(2, trans_box_center - side / 2, [side side]));

                %% 6) Get the crop!
                cropped = impcrop(rot_stack, box);

                %% 8) Rescale crop to CNN and rescale joints/flow to be in image coordinates
                % permute(x, [2 1 3]) puts the width dimension first in x, which is what
                % caffe wants (IIRC Caffe uses width * height * channels * num).
                final_stack = permute(imresize(cropped, cnn_window), [2 1 3]);

                % Scale factors for flow and joints
                scale_factors = size(final_stack) ./ size(cropped);
                scale_factors = scale_factors(2:-1:1);

                % Scaling flow
                final_stack(:, :, 7:8) = bsxfun(@times, final_stack(:, :, 7:8), reshape(scale_factors, [1 1 2]));

                % Scaling joints
                scale_joints = rot_joints;
                for i=1:size(scale_joints, 1)
                    scale_joints(i, :) = (scale_joints(i, :) - box(1:2)) .* scale_factors;
                end

                % Return column vector [x1 y1 x2 y2 ... xn yn]'
                poselet_joints = scale_joints(poselet_indices);
                rvs(current_idx).joint_labels = reshape(poselet_joints', [numel(poselet_joints), 1]);
                % Return full w * h * c matrix
                rvs(current_idx).stack = final_stack;
                
                current_idx = current_idx + 1;
            end
        end
    end
end
end

function normed = norm_im(im)
normed = single(im) / 255.0;
end

% TODO: Consider ways of normalising flow
function normed = norm_flow(flow)
normed = single(flow);
end

function flipped = flip_lr(joints)
% XXX: This will break for datasets other than FLIC
flipped = joints([4:6 1:3 10 8:9 7 11:12 14 13 15:29], :);
assert(all(size(flipped) == size(joints)));
end