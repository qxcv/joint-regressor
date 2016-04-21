function [im_stack, flow, box, xtrim, ytrim, scale] = cropscale_pos(im_stack, flow, box, cnn_window, true_scale)
% Crop positive example to speed up latent search.
%
% Remember that true_scale should be taken from pair.scale for training
% positives!
%
% ytrim, xtrim and scale are for transforming search results back into
% image scale at the end of the function. ytrim and xtrim indicate the
% number of pixels which are missing on the top and left of the image,
% respectively, whilst scale is the factor by which the image was scaled
% after trimming. To recover joint coordinates in image-scale, you'll have
% to divide the results of detection on the cropscale'd image by `scale`,
% then add [xoff yoff] to each joint (assuming [x y] coordinates).

assert(all(size(im_stack) == size(flow) | [0 0 1]));
assert(isscalar(cnn_window));
assert(isscalar(true_scale));

x1 = box.xy(:,1);
y1 = box.xy(:,2);
x2 = box.xy(:,3);
y2 = box.xy(:,4);
siz = x2(1)-x1(1)+1;

x1 = min(x1); y1 = min(y1); x2 = max(x2); y2 = max(y2);

% crop image around bounding box
pad = siz * 0.5;
x1 = max(1, round(x1-pad));
y1 = max(1, round(y1-pad));
x2 = min(size(im_stack,2), round(x2+pad));
y2 = min(size(im_stack,1), round(y2+pad));

% Store offsets so that we can recover original coordinates
xtrim = x1 - 1;
ytrim = y1 - 1;

im_stack = im_stack(y1:y2, x1:x2, :);
flow = flow(y1:y2, x1:x2, :);
box.xy(:,[1 3]) = box.xy(:,[1 3]) - x1 + 1;
box.xy(:,[2 4]) = box.xy(:,[2 4]) - y1 + 1;

% further scale it to the "true scale". This might involve a bit of
% upscaling which will be undone later, but I'm happy to take that hit.
scale = cnn_window / true_scale;
% Just a heuristic check. If it trigers, make sure everything is working
% okay and then change the bounds so that it stops triggering.
assert(1/4 <= scale && scale <= 4, 'Scale %f seems too large', scale);
im_stack = imresize(im_stack, scale);
flow = smart_resize_flow(flow, size(im_stack));
assert(all(size(im_stack) == size(flow) | [0 0 1]));

box.xy = (box.xy - 1)*scale + 1;
