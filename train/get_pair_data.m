function [im1, im2, flow] = get_pair_data(d1, d2)
%GET_PAIR_DATA Get images and flow for data.
im1 = readim(d1);
im2 = readim(d2);
flow = imflow(im1, im2);
assert(all(size(im1) == size(im2)));
assert(size(im1, 1) == size(flow, 1) && size(im1, 2) == size(flow, 2));
assert(size(flow, 3) == 2);
assert(size(im1, 3) == 3);
end

