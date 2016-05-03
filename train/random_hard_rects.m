function boxes = random_hard_rects(f1_joints, f2_joints, subposes, ...
    box_size, negs_per_sp)
%RANDOM_HARD_RECTS Get cropping rectangles for hard negatives
% Note that there is no requirement that the boxes are contained entirely
% within the source image, so you'll have to use impcrop or something to
% handle corner cases.
boxes = [];

% Stop generating boxes for a subpose after you've done this many
max_iter = 1000 * negs_per_sp;
% IOU between relevant subpose and generated box must be >= this
in_view_iou = 0.2;
% IOU must be <= this for all subposes
centred_iou = 0.5;
% [x y w h] bounding box for each subpose
sp_bboxes = zeros([length(subposes) 4]);
for sp_idx=1:length(subposes)
    parts = subposes(sp_idx).subpose;
    sp_coords = cat(1, f1_joints(parts, :), f2_joints(parts, :));
    assert(ismatrix(sp_coords) && size(sp_coords, 2) == 2);
    mins = min(sp_coords, [], 1);
    maxes = max(sp_coords, [], 1);
    midpoint = mean(cat(1, mins, maxes), 1);
    this_loc = midpoint - box_size / 2;
    sp_bboxes(sp_idx, :) = [this_loc box_size box_size];
    
    % Sanity check: real bbox should be inside our square bbox
    % Might need to fix box_size if that is not the case.
    tight_bbox = get_bbox(sp_coords);
    this_right = this_loc + sp_bboxes(sp_idx, 3:4);
    tight_right = tight_bbox(1:2) + tight_bbox(3:4);
    assert(all(floor(this_loc) <= tight_bbox(1:2)) ...
        && all(ceil(this_right) >= tight_right));
end

boxes = [];

for sp_idx=1:length(subposes)
    % Need to find a box which (a) has the subpose in view (b) does not
    % have the subpose centred and (c) does not have any other subpose
    % centred either.
    iters = 0;
    while size(boxes, 1) < negs_per_sp
        this_box = random_box(sp_bboxes(sp_idx, :), box_size, in_view_iou);
        inters = boxes_overlap(this_box, sp_bboxes);
        if all(inters < centred_iou)
            % If the box isn't focused on any particular part then we can
            % use it.
            boxes = [boxes; this_box];
            continue
        end
        
        % Termination check
        iters = iters + 1;
        if iters > max_iter
            warning(...
                ['Exceeded maximum number of iterations on %s; got %i ' ...
                 'boxes after %i iterations'], subposes(sp_idx).name, ...
                 size(boxes, 1), iters);
        end
    end
end

assert(size(boxes, 2) == 4);
end

function box = random_box(sp_bbox, box_size, min_iou)
assert(isscalar(box_size) && isvector(sp_bbox) && all(sp_bbox(3:4) == box_size));

% prohib_space is space along each side of the box that we can't occupy,
% perm_space is space we can
prohib_space = sqrt(min_iou * 2 * box_size^2 / (1+min_iou));
perm_space = box_size - prohib_space;
assert(0 <= prohib_space && 0 <= perm_space);

% Put the upper left corner somewhere allowable
box_loc = 2 * perm_space * rand(1, 2) + sp_bbox([1 2]) - perm_space;
box = [box_loc box_size box_size];

assert(boxes_overlap(sp_bbox, box) > min_iou, 'IOU constraint violated');
end
