function out = impcrop(image, box)
%IMPCROP Just like imcrop, except we replicate-pad outside of the box
% Builds on imcrop2 (so it handles multiple channels correctly)

% Padding function freaks out if it's given a single. It's a shame that the
% Mex API doesn't handle type conversions safely and conveniently; it's
% probably to blame for all of the "OMG, I WANTED AN INT16 BUT YOU GAVE ME
% AN INT8!!!!!" problems.

assert(isa(box, 'double') || isa(box, 'single'));
box = double(box);

imw = size(image, 2);
imh = size(image, 1);
box = round(box);
bw = box(3);
bh = box(4);
bl = box(1);
bt = box(2);

out = imcrop2(image, box);
if size(out, 1) == 0 && size(out, 2) == 0
    % There's no intersection, return zeros
    out = zeros([bh + 1, bw + 1, size(image, 3)]);
else
    if bl < 1
        % Left-pad
        out = padarray(out, [0, min(1 - bl, bw), 0], 'replicate', 'pre');
    end
    if bt < 1
        % Top-pad
        out = padarray(out, [min(1 - bt, bh), 0, 0], 'replicate', 'pre');
    end
    if box(1) + bw > imw
        % Right-pad
        out = padarray(out, [0, min(box(1) + bw - imw, bw), 0], 'replicate', 'post');
    end
    if box(2) + bh > imh
        % Bottom-pad
        out = padarray(out, [min(box(2) + bh - imh, bh), 0, 0], 'replicate', 'post');
    end
end
end
