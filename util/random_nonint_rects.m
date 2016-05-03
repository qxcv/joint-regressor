function boxes = random_nonint_rects(frame, avoid_box, ...
    min_size, max_size, count)
%RANDOM_NONINT_RECTS Produces crop rectangles for negatives.
% Does this by producing a desired number of rectangles by:
%  1) Randomly sampling the space of allowable rectangle sizes
%  2) Randomly sampling the space of rectangles which fit inside the given
%  bounding box.
%  3) Discarding any rectangles which intersect avoid_box.
%  4) Stopping if it take longer than 100 iterations to find an appropriate
%     box.
%
% *frame:* rectangle of [xmin ymin width height] format which generated
% rectangles must lie within.
% *avoid_box:* rectangle of [xmin ymin width height] format which
% rectangles will not intersect with.
% *min_size:* a minimum width and height.
% *max_size:* a maximum width and height.
% *count:* number of boxes to generate

% Boxes are in format [xmin ymin width height], as expected by imcrop
boxes = zeros(count, 4);
iterations = 0;
num_boxes = 0;
% This avoids creating boxes that are too big without sacrificing
% uniformity
scalar_err = 'min_size and max_size should have same isscalar()';
if isscalar(max_size)
    max_size = min([max_size frame(3:4)]);
    assert(isscalar(min_size), scalar_err);
else
    max_size = min(max_size, frame(3:4));
    assert(~isscalar(min_size), scalar_err);
end

assert(all(max_size >= min_size), 'Boxes need volume');

for i=1:count
    have_rect = false;
    while ~have_rect
        % We could speed this up using heuristics to avoid obvious
        % collisions, but I don't want to hurt the uniformity of the
        % sampling process.
        iterations = iterations + 1;
        if iterations > 1000 * count
            warning(...
                ['Exceeded maximum number of iterations; returning %i ' ...
                 'boxes after %i iterations'], num_boxes, iterations);
             boxes = boxes(1:num_boxes, :);
             return;
        end
        if isscalar(max_size) == 1
            % If we have a scalar size, then we need to make the box square
            side = rand(1) * (max_size - min_size) + min_size;
            rect_size = [side side];
            if side < 100
                disp(rect_size);
                disp([min_size max_size]);
            end
        else
            rect_size = rand(1, 2) .* (max_size - min_size) + min_size;
        end
        rect_loc = rand(1, 2) .* frame(3:4) + frame(1:2);
        rect = [rect_loc rect_size];
        if rectint(frame, rect) < rectint(rect, rect) - eps ...
                || rectint(rect, avoid_box) > eps
            continue
        end
        boxes(i, :) = rect;
        num_boxes = num_boxes + 1;
        break
    end
end
end
