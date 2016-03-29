function [ prettified ] = pretty_flow(flow)
%PRETTY_FLOW Make some pretty flow in the HSV space, convert back to RGB.
flow_u = flow(:, :, 1);
flow_v = flow(:, :, 2);
hue = atan2(flow_u, flow_v) / (2*pi) + 0.5;
mags = sqrt(flow_u.^2 + flow_v.^2);
% Normalised to size of image. Capped at the size of the image, and then
% rescaled to [0.25, 0.75].
saturation = min(10 * mags / size(mags, 1), 1);
value = 0.5 * ones(size(hue));
hsv = cat(3, hue, saturation, value);
prettified = rgb2hsv(hsv);
end

