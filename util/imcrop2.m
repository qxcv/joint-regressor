function cropped = imcrop2(image, rect)
% Like Matlab's imcrop(I, rect), but doesn't intentionally throw errors
% when there are more than three channels (imcrop probably only does that
% to save its bacon in the case that it needs to call imshow();
% nevertheless, I think it's inelegant to put that check in imcrop, which
% is channel-invariant, rather than in imshow).

% First, check for invalid rect (a bit like what imcrop does)
rect = floor(rect);
if rect(1) > size(image, 2) || rect(2) > size(image, 1) ...
        || rect(1) + rect(3) < 1 || rect(2) + rect(4) < 1 ...
        || size(image, 1) == 0 || size(image, 2) == 0
    cropped = zeros(0, 0);
    return;
end

left = max(rect(1), 1);
right = max(min(rect(1) + rect(3), size(image, 2)), left);
top = max(rect(2), 1);
bottom = max(min(rect(2) + rect(4), size(image, 1)), top);
cropped = image(top:bottom, left:right, :);
end