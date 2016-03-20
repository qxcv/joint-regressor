function rvs = get_stacks(d1, d2, pair_scale, subposes, left_parts, right_parts, cache_dir, cnn_window, aug)
%GET_STACKS Get the image/flow stacks for a given data pair and set of
%transformations.
% d1: First datum
% d2: Second datum
% pair_scale: side length of CNN receptive field in image pixels (from
%             mark_scales.m)
% subposes: subpose data from config file
% left_parts: indices of parts which are on the left (shoulders, elbows,
%             hands, wrists, etc.)
% right_parts: indices of parts which are on the right
% cache_dir: path to cache directory
% cnn_window: width and height of CNN window
% aug: data structure defining which augmentations to perform
% rvs: struct array with .labels (joint locations) and .stack (full input
%      data) attributes.
[im1, im2, flow] = get_pair_data(d1, d2, cache_dir);
% Concatenate along channels
stacked = cat(3, norm_im(im1), ...
                 norm_im(im2), ...
                 norm_flow(flow));
% Now join joints
all_joints = cat(1, d1.joint_locs, d2.joint_locs);

% We'll store rvs in a struct array with a "stack" and "joints" attribute
% for each entry.
current_idx = 1;

% Flips
assert(any(strcmp(aug.flip_mode, {'random', 'both', 'none'})));
if strcmp(aug.flip_mode, 'random')
    flips = randi([0 1]);
elseif strcmp(aug.flip_mode, 'both')
    flips = [0 1];
else
    flips = 0;
end

% Rotations
assert(isscalar(aug.rand_rots));
assert(numel(aug.rot_range) == 2);
rotwidth = aug.rot_range(2) - aug.rot_range(1);
rotations = rotwidth * rand([1 aug.rand_rots]) + aug.rot_range(1);

% Random translations
assert(isscalar(aug.rand_trans));
rand_trans = aug.rand_trans;

for flip=flips
    %% 1) Flip
    flip_joints = all_joints;
    flip_stack = stacked;
    if flip
        % Reverse joints
        flip_joints(:, 1) = size(im1, 2) - flip_joints(:, 1) + 1;
        % Swap left side for right side
        num_joints = size(d1.joint_locs, 1);
        assert(num_joints == size(d2.joint_locs, 1));
        flip_joints(1:num_joints, :) = flip_lr(...
            flip_joints(1:num_joints, :), left_parts, right_parts);
        flip_joints(num_joints+1:end, :) = flip_lr(...
            flip_joints(num_joints+1:end, :), left_parts, right_parts);
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
        
        %% 3) Choose a subpose
        for subpose_num=1:length(subposes)
            this_subpose = subposes(subpose_num).subpose;
            subpose_indices = [this_subpose, this_subpose + length(d1.joint_locs)];
            
            %% 4) Get bounding box for joint
            maxes = max(rot_joints(subpose_indices, :), [], 1);
            mins = min(rot_joints(subpose_indices, :), [], 1);
            assert(numel(maxes) == 2 && numel(mins) == 2);
            % Always crop a square patch
            pose_side = max(maxes - mins);
            % box_center is (x, y)
            box_center = mins + (maxes - mins) ./ 2;
            extra = sprintf('sp_idx=%i; pose_side=%f; pair_scale=%f', ....
                subpose_num, pose_side, pair_scale);
            % sqrt(2) thing is to account for fact that joints are rotated
            assert(sqrt(2) * pair_scale > pose_side, ...
                ['mark_scales should have defined a scale larger than '...
                 'the longest side of the largest subpose. ' extra]);
            
            %% 5) Random translations
            % We repeat this translation process for each random
            % translation
            total_trans = max(rand_trans, 1);
            wiggle_room = pair_scale - pose_side;
            if wiggle_room <= 1
                % Don't translate at scale 1 or below (effectively)
                total_trans = 1;
            end;
            
            for unused=1:total_trans
                %% 6) Translate box
                trans_box_center = box_center;
                % TODO: Instead of finding wiggle room automatically, I
                % should specify a maximum distance by which the pose can
                % be translated, in pixels. That could then be scaled
                % apprporiately according to the datum scale, then the 
                if rand_trans > 0 && wiggle_room > 1
                    trans_amount = wiggle_room * rand(1, 2) - wiggle_room / 2;
                    trans_box_center = trans_box_center + trans_amount;
                end
                box = cat(2, round(trans_box_center - pair_scale / 2), ...
                    [pair_scale - 1, pair_scale - 1]);
                assert(all(round(box) == box), ...
                    'Box should have integer coords and size');
                
                %% 7) Get the crop!
                cropped = impcrop(rot_stack, box);
                cropped_size = size(cropped);
                assert(all(cropped_size(1:2) == pair_scale), ...
                    'Cropped region should be pair_scale * pair_scale');
                
                %% 8) Rescale crop to CNN and rescale joints/flow to be in image coordinates
                % permute(x, [2 1 3]) puts the width dimension first in x, which is what
                % caffe wants (IIRC Caffe uses width * height * channels * num).
                final_stack = permute(imresize(cropped, cnn_window), [2 1 3]);
                
                % Scaling flow
                final_stack(:, :, 7:8) = rescale_flow_mags(...
                    final_stack(:, :, 7:8), size(final_stack), size(cropped));
                
                % Scale factors for joints
                scale_factors = size(final_stack) ./ size(cropped);
                scale_factors = scale_factors(2:-1:1);
                scale_joints = rot_joints;
                for i=1:size(scale_joints, 1)
                    scale_joints(i, :) = (scale_joints(i, :) - box(1:2)) .* scale_factors;
                end
                
                % Return column vector [x1 y1 x2 y2 ... xn yn]'
                subpose_joints = scale_joints(subpose_indices, :);
                rvs(current_idx).joint_labels = reshape(subpose_joints', [numel(subpose_joints), 1]); %#ok<AGROW>
                % Return full w * h * c matrix
                rvs(current_idx).stack = final_stack; %#ok<AGROW>
                rvs(current_idx).subpose_num = subpose_num; %#ok<AGROW>
                
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

function flipped = flip_lr(joints, left_side, right_side)
flipped = joints;
flipped(left_side, :) = joints(right_side, :);
flipped(right_side, :) = joints(left_side, :);
assert(all(size(flipped) == size(joints)));
end