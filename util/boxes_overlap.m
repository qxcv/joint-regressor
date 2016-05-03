function ious = boxes_overlap(box, sp_bboxes)
% Get ious for one box (box) vs. many (sp_bboxes)
% Box format is [x y w h]
assert(all(size(box) == [1 4]));
assert(ismatrix(sp_bboxes) && size(sp_bboxes, 2) == 4);

box_area = prod(box(3:4));
areas = prod(sp_bboxes(:, 3:4), 2);
assert(isscalar(box_area));
assert(isvector(areas) && length(areas) == size(sp_bboxes, 1));

x_inter = intersect_1d(box(1), box(3), sp_bboxes(:, 1), sp_bboxes(:, 3));
y_inter = intersect_1d(box(2), box(4), sp_bboxes(:, 2), sp_bboxes(:, 4));
intersections = x_inter .* y_inter;

unions = areas + box_area - intersections;
assert(isvector(unions) && all(size(unions) == size(intersections)));

ious = intersections ./ unions;
assert(length(ious) == size(sp_bboxes, 1));
end

function area = intersect_1d(x1, w1, x2, w2)
% Intersection of [x1, x1+w1], [x2, x2+w2].
assert(isscalar(x1) && isscalar(w1) && isvector(x2) && isvector(w2));
assert(length(x2) == length(w2));
area = min(w1, min(w2, min(max(0, x1+w1-x2), max(0, x2+w2-x1))));
assert(all(area >= 0));
end
