function [rv_stack, rv_joints] = get_stack(conf, d1, d2, flip, rotate, scale, translate)
%GET_STACK Get the image/flow stack for a given data pair.
% d1: First datum
% d2: Second datum
% flip: Should we flip the image left/right?
% rotate: How many degrees should we rotate by?
% translate: Matrix giving [x y] amounts to translate by, as fractions of
% computed boundix box size (after cropping to the skeleton and scaling).

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
          
% Translations!
%% 1) Flip
if flip
    stacked = stacked(:, end:-1:1, :);
    all_joints(:, 1) = size(im1, 2) - all_joints(:, 1) + 1;
end

%% 2) Rotate
if rotate ~= 0
    all_joints = map_rotate_points(all_joints, stacked, rotate, 'ori2new');
    stacked = improtate(stacked, rotate, 'bilinear');
end

%% 3) Get bounding box for joints
maxes = max(all_joints, [], 1);
mins = min(all_joints, [], 1);
% Always crop a square patch
side = max(maxes - mins);
% box_center is (x, y)
box_center = mins + (maxes - mins) ./ 2;

%% 4) Scale box
side = round(side / scale);

%% 5) Translate box
box_center = round(box_center + side * translate);
box = round(cat(2, box_center - side / 2, [side side]));

%% 6) Get the crop!
cropped = impcrop(stacked, box);

%% 8) Rescale crop to CNN
% imresize size is (rows, cols), which corresponds to (height, width).
rv_stack = imresize(cropped, conf.cnn.window);

%% 9) Rescale joints to be in image coordinates
scale_factors = size(rv_stack) ./ size(cropped);
scale_factors = scale_factors(2:-1:1);
for i=1:size(all_joints, 1)
    all_joints(i, :) = (all_joints(i, :) - box(1:2)) .* scale_factors;
end
% Return column vector [x1 y1 x2 y2 ... xn yn]'
rv_joints = reshape(all_joints', [numel(all_joints), 1]);

function normed = norm_im(im)
normed = single(im) / 255.0;

% Try to keep most flow in [-1, 1]
function normed = norm_flow(flow)
normed = single(flow);