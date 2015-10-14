function [rv_stack, rv_joints] = get_stack(conf, d1, d2, flip, rotate, scale, randtrans)
%GET_STACK Get the image/flow stack for a given data pair.
% d1: First datum
% d2: Second datum
% flip: Should we flip the image left/right?
% rotate: How many degrees should we rotate by?
% randtrans: If this is specified, and there's some space between the
% bounding box of the pose and the edge of the image, then the pose
% box will be translated randomly around in that space. Otherwise, the pose
% will be centered perfectly in the crop.
im1 = readim(d1);
im2 = readim(d2);
flow = cached_imflow(d1, d2, conf.cache_dir);
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

%% 1) Flip
if flip
    stacked = stacked(:, end:-1:1, :);
    all_joints(:, 1) = size(im1, 2) - all_joints(:, 1) + 1;
    stacked(:, :, 7) = -stacked(:, :, 7);
end

%% 2) Rotate
if rotate ~= 0
    all_joints = map_rotate_points(all_joints, stacked, rotate, 'ori2new');
    stacked = improtate(stacked, rotate, 'bilinear');
    rot_mat = [cosd(rotate), -sind(rotate); sind(rotate), cosd(rotate)];
    flow = stacked(:, :, 7:8);
    flat_flow = reshape(flow, [size(flow, 1) * size(flow, 2), 2]);
    flat_flow = (rot_mat * flat_flow.').';
    stacked(:, :, 7:8) = reshape(flat_flow, size(flow));
end

%% 3) Get bounding box for joints
maxes = max(all_joints, [], 1);
mins = min(all_joints, [], 1);
% Always crop a square patch
pose_side = max(maxes - mins);
% box_center is (x, y)
box_center = mins + (maxes - mins) ./ 2;

%% 4) Scale box
side = round(pose_side / scale);

%% 5) Translate box
wiggle_room = side - pose_side;
if randtrans && wiggle_room > 1
    trans_amount = wiggle_room * rand(1, 2) - wiggle_room / 2;
    box_center = box_center + trans_amount;
end
box = round(cat(2, box_center - side / 2, [side side]));

%% 6) Get the crop!
cropped = impcrop(stacked, box);

%% 8) Rescale crop to CNN and rescale joints/flow to be in image coordinates
% permute(x, [2 1 3]) puts the width dimension first in x, which is what
% caffe wants (IIRC Caffe uses width * height * channels * num).
rv_stack = permute(imresize(cropped, conf.cnn.window), [2 1 3]);

% Scale factors for flow and joints
scale_factors = size(rv_stack) ./ size(cropped);
scale_factors = scale_factors(2:-1:1);

% Scaling flow
rv_stack(:, :, 7:8) = bsxfun(@times, rv_stack(:, :, 7:8), reshape(scale_factors, [1 1 2]));

% Scaling joints
for i=1:size(all_joints, 1)
    all_joints(i, :) = (all_joints(i, :) - box(1:2)) .* scale_factors;
end

% Return column vector [x1 y1 x2 y2 ... xn yn]'
rv_joints = reshape(all_joints', [numel(all_joints), 1]);

function normed = norm_im(im)
normed = single(im) / 255.0;

% TODO: Consider ways of normalising flow
function normed = norm_flow(flow)
normed = single(flow);