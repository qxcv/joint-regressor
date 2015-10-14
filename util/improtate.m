function rotated = improtate(im, angle, sampling)
%IMPROTATE Rotate and pad, all in one!
if cos(angle) == 0
    alphas = [0 0];
else
    % TODO: This doesn't really work :(
    alphas = abs(tand(angle) ./ [sind(angle), cosd(angle)]);
end
pads = size(im) .* [alphas(1) alphas(2) 0];
padded = padarray(im, round(pads), 'replicate', 'both');
if nargin < 3
    sampling = 'bicubic';
end
full_rot = imrotate(padded, angle, sampling, 'crop');
rot_w = round(sum(abs([size(im, 2) * cosd(angle), size(im, 1) * sind(angle)])));
rot_h = round(sum(abs([size(im, 1) * cosd(angle), size(im, 2) * sind(angle)])));
bounds = round([(size(full_rot, 2) - rot_w) / 2, ...
    (size(full_rot, 1) - rot_h) / 2, rot_w, rot_h]);
rotated = imcrop2(full_rot, bounds);
end

