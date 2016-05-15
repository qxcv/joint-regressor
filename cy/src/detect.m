function [boxes, model, ex] = detect(im1_info, im2_info, model, varargin)
% Detect objects in image using a model (SSVM + CNN) and either a score
% threshold or fixed number of boxes to return.
%
% Returns a struct array with bounding boxes for all parts in all
% detections and a score for each detection.
%
% This function updates the model (by running the QP solver) if upper and
% lower bound differs

% MathWorks: "Hey, you know what would be a great idea? Not adding default
% arguments to our language! Think of how elegant Matlab would be without
% that cruft!"
parser = inputParser;
parser.CaseSensitive = true;
parser.StructExpand = false;
parser.addRequired('im1_info', @isstruct);
parser.addRequired('im2_info', @isstruct);
parser.addRequired('model', @isstruct);
parser.addOptional('PairInfo', [], @isstruct);
parser.addOptional('CNNSavePath', [], @isstr);
parser.addOptional('Thresh', [], @isscalar);
parser.addOptional('NumResults', [], @isscalar);
parser.addOptional('BBox', [], @isstruct);
parser.addOptional('TrueScale', [], @isscalar);
parser.addOptional('Overlap', [], @isscalar);
parser.addOptional('ID', [], @isscalar);
parser.addOptional('Label', [], @isscalar);
parser.addOptional('CacheDir', [], @isstr);
parser.parse(im1_info, im2_info, model, varargin{:});
r = parser.Results;
pair = r.PairInfo;
cnn_save_path = r.CNNSavePath;
thresh = r.Thresh;
num_results = r.NumResults;
bbox = r.BBox;
true_scale = r.TrueScale;
overlap = r.Overlap;
id = r.ID;
label = r.Label;
cache_dir = r.CacheDir;
% Now some sanity checks on arguments
assert(xor(isempty(thresh), isempty(num_results)), ...
    'Can take detection return threshold xor number of results to return');
train_args_supplied = ~cellfun(@isempty, {overlap, id, label});
% is_train_call implies we're being called from train()
is_train_call = all(train_args_supplied);
assert(is_train_call || ~any(train_args_supplied), ...
    'If you''re calling from train(), make sure you include everything needed');
assert(~is_train_call || label < 0 || (~isempty(bbox) && ~isempty(pair)), ...
    'If you have a positive, you need a bbox and pair info for supervision');

INF = 1e10;

if is_train_call && ~isempty(bbox)
    latent = true;
    if label > 0
        assert(~isempty(thresh));
        % XXX: Is this broken? Why would it overwrite thresh?
        thresh = -INF;
    end
else
    latent = false;
end

% Compute the feature pyramid and prepare filter
im1 = readim(im1_info);
im2 = readim(im2_info);
persistent warned_about_cache;
if ~isempty(cache_dir)
    flow = cached_imflow(im1_info, im2_info, cache_dir);
else
    if isempty(warned_about_cache)
        warning('JointRegressor:detect:nocache', ...
            'Recomputing image flow! (will only warn once)\n');
        warned_about_cache = true;
    end
    flow = imflow(im1, im2);
end
im_stack = cat(3, im1, im2);
% if has box information, crop it
if ~isempty(bbox)
    % crop positives and evaluation images to speed up search
    % TODO: Replace model.cnn.window(1) with model.cnn.window once window
    % is a scalar.
    if ~isempty(pair)
        assert(pair.scale == true_scale, ...
            'true_scale is just pair.scale when pair info is supplied');
    end
    [im_stack, flow, bbox, cs_xtrim, cs_ytrim, cs_scale] = ...
        cropscale_pos(im_stack, flow, bbox, model.cnn.window(1), true_scale);
end

cnn_args = {im_stack, flow, model};
if ~isempty(cnn_save_path)
    % Cache output in save path
    cnn_args{end+1} = cnn_save_path;
end
[pyra, unary_map] = imCNNdet(cnn_args{:});

levels = 1:length(pyra);

% Define global QP if we are writing features
% Randomize order to increase effectiveness of model updating
write = false;
if is_train_call
    global qp; %#ok<TLEV>
    write  = true;
    levels = levels(randperm(length(levels)));
end

% Cache various statistics derived from model
[components, apps] = modelcomponents(model);

num_subposes = length(components);
assert(num_subposes > 1, 'Only %d parts?\n', num_subposes);
boxes = struct('boxes', {}, 'types', {}, 'rscore', {});
cnt = 0;

ex.blocks = [];
ex.id = [label id 0 0 0];
ex.debug = [];

% det_side is roughly the width and height of a detection, in
% heatmap coordinates.
det_side = model.cnn.window(1) / model.cnn.step;

if latent && label > 0
    % record best when doing latent on positive example
    best_ex = ex;
    best_box = [];
end

% Iterate over random permutation of scales and components,
for level = levels
    % Iterate through mixture components
    sizs = pyra(level).sizs;
    
    % Skip if there is no overlap of root filter with bbox
    if latent
        skipflags = false([1 num_subposes]);
        % because all mixtures for one part is the same size, we only need to do this once
        for subpose_idx = 1:num_subposes
            ovmask = testoverlap(det_side, det_side,...
                sizs(1), sizs(2), ...
                pyra(level), bbox.xy(subpose_idx,:), overlap);
            skipflags(subpose_idx) = ~any(ovmask(:));
        end
        % If many subposes are too small, we skip this level
        % This check used to skip a level if ANY subposes were too small,
        % but that caused problems.
        if mean(skipflags) > 1/4
            fprintf('detect() skipping level %i/%i\n', level, length(levels));
            continue;
        end
    end
    % Local scores
    
    for subpose_idx = 1:num_subposes
        components(subpose_idx).appMap = unary_map{level}{subpose_idx};
        assert(ndims(components(subpose_idx).appMap) == 3, ...
            'Need h*w*K unary map');
        
        % appid will be 1x1, and gives the unary weight associated with
        % this subpose
        f = components(subpose_idx).appid;
        assert(isscalar(f), 'Should have only one weight ID');
        assert(isscalar(apps{f}), 'Should have only one weight for that ID');
        % .score will now be h*w*K for each part
        weighted_apps = components(subpose_idx).appMap * apps{f};
        assert(ndims(weighted_apps) == 3);
        assert(size(weighted_apps, 3) == model.K);
        components(subpose_idx).score = weighted_apps;
        components(subpose_idx).level = level;
        
        if latent
            assert(label > 0, 'This doesn''t make sense on negatives');
            
            ovmask = testoverlap(det_side, det_side, sizs(1), sizs(2), ...
                pyra(level), bbox.xy(subpose_idx,:), overlap);
            assert(ismatrix(ovmask));
            tmpscore = components(subpose_idx).score;
            tmpscore_K = size(tmpscore, 3);
            assert(tmpscore_K == model.K);
            ovmask = repmat(ovmask, 1, 1, tmpscore_K);
            assert(all(size(ovmask) == size(tmpscore)));
            
            % If a location doesn't overlap enough with the ground
            % truth, then we set it to -INF
            %tmpscore(~ovmask) = -INF;
            
            % If a poselet is a long way from the GT poselet, then we
            % also set it to -INF
            near_pslts = pair.near{subpose_idx};
            assert(~isempty(near_pslts));
            assert(all(1 <= near_pslts & near_pslts <= model.K));
            far_pslts = true([1 model.K]);
            far_pslts(near_pslts) = false;
            assert(sum(far_pslts) == model.K - length(near_pslts));
            assert(length(far_pslts) == model.K);
            % XXX: masking the appMap doesn't seem to help training
            % much (although it does make debugging easier by showing
            % me where things are broken, so I might keep it).
            %components(subpose_idx).appMap(:, :, far_pslts) = -INF;
            %tmpscore(:, :, far_pslts) = -INF;
            assert(all(size(components(subpose_idx).score) == size(tmpscore)), ...
                'tmpscore changed size');
            assert(any(tmpscore(:) > -INF), ...
                'All scores have been masked out (!!)');
            components(subpose_idx).score = tmpscore;
        end
    end
    
    % Walk from leaves to root of tree, passing message to parent
    for subpose_idx = num_subposes:-1:2
        child = components(subpose_idx);
        par_idx = components(subpose_idx).parent;
        assert(0 < par_idx && par_idx < subpose_idx);
        parent = components(par_idx);
        
        % msg is for score; Ix, Iy and Im are for x location, y location
        % and part type (*m*ixture?) backtracking, respectively. Each
        % matrix is of size H*W*K (so each entry corresponds to a single
        % parent configuration).
        [msg, components(subpose_idx).Ix, ...
              components(subpose_idx).Iy, ...
              components(subpose_idx).Im] ...
            = passmsg(child, parent, model.sbin);
        components(par_idx).score = components(par_idx).score + msg;
    end
    
    % Add bias to root score (model.root == 1)
    components(model.root).score = ...
        components(model.root).score + components(model.root).b;
    rscore = components(model.root).score;
    assert(ndims(rscore) == 3);
    
    % keep the positive example with the highest score in latent mode
    if latent && label > 0
        thresh = max(thresh, max(rscore(:)));
    end

    if ~isempty(thresh)
        [Y, X, T] = ndfind(rscore >= thresh);
    else
        % TODO: If some of our num_results boxes have rscores below those
        % of the boxes we've seen already (at other levels), then they will
        % be ignored, so we should not both backtracking to fetch them.
        [Y, X, T] = ndbestn(rscore, num_results);
    end

    % Walk back down tree following pointers
    % (DEBUG) Assert extracted feature re-produces score
    % If we never iterate through X, we'll never write an example, so start
    % with ~wrote_ex
    wrote_ex = false;
    for i = 1:length(X)
        cnt = cnt + 1;
        x = X(i);
        y = Y(i);
        t = T(i);
        
        [box, types, ex] = ...
            backtrack(x, y, t, det_side, components, pyra(level), ex, write, model.sbin);
        
        this_rscore = rscore(y, x, t);
        b.boxes = num2cell(box, 2);
        b.types = num2cell(types);
        b.rscore = this_rscore;
        boxes(end+1) = b; %#ok<AGROW>
        assert(length(boxes) == cnt);
        % wrote_ex should tell us whether the *last* ex we looked at was
        % written, so we need to set it to false whenever we look at a new
        % ex (and only set it to true if that ex is written).
        wrote_ex = false;
        if write && (~latent || label < 0)
            wrote_ex = qp_write(ex);
            qp.ub = qp.ub + qp.Cneg*max(1+this_rscore, 0);
        elseif latent && label > 0
            if isempty(best_box)
                best_box = boxes(end);
                best_ex = ex;
            elseif best_box(end).rscore < this_rscore
                % update best
                best_box = boxes(end);
                best_ex = ex;
            end
        end
    end
    
    % Crucial DEBUG assertion:
    % If we're computing features, assert extracted feature re-produces
    % score (see qp_writ.m for computing original score)
    if wrote_ex
        w = -(qp.w + qp.w0.*qp.wreg) / qp.Cneg;
        sv_score = score(w,qp.x,qp.n);
        sv_rscore = rscore(y,x,t);
        delta = abs(sv_score - sv_rscore);
        if delta >= 1e-5
            fprintf(...
                'Delta %f = |%f (SV score) - %f (rscore)| too big\n', ...
                delta, sv_score, sv_rscore);
        end
    end
    
    % Optimize qp with coordinate descent, and update model
    if write && (~latent || label < 0) && ...
            (qp.lb < 0 || 1 - qp.lb/qp.ub > .05 || qp.n == length(qp.sv))
        model = optimize(model);
        [components, apps] = modelcomponents(model);
    end
end

if latent && ~isempty(boxes) && label > 0
    boxes = best_box;
    if write
        qp_write(best_ex);
    end
end

% Sort by root score, best-first
if length(boxes) > 1
    [~, best_idxs] = sort([boxes.rscore], 'Descend');
    boxes = boxes(best_idxs);
end

% Make sure we only return num_results results (if NumResults supplied)
if ~isempty(num_results) && length(boxes) > num_results
    boxes = boxes(1:num_results);
end

% Undo cropscale_pos transformation on coordinates
if ~isempty(bbox) && ~isempty(boxes)
    boxes = unscale_boxes(boxes, cs_xtrim, cs_ytrim, cs_scale);
end
end

% Backtrack through dynamic programming messages to estimate part locations
% and the associated feature vector
function [box,types,ex] = backtrack(root_x,root_y,root_t,det_side,parts,pyra,ex,write,sbin)
numparts = length(parts);
ptr = zeros(numparts,3);
box = zeros(numparts,4);
types = zeros(1, numparts);
root = 1;
root_p = parts(root);
ptr(root, :) = [root_x, root_y, root_t];
scale = pyra.scale;

box(root,:) = get_subpose_box(root_x, root_y, det_side, pyra.pad, scale);
types(root) = root_t;

if write
    ex.id(3:6) = [root_p.level round(root_x+det_side/2) round(root_y+det_side/2) root_t];
    ex.blocks = [];
    ex.blocks(end+1).i = root_p.biasI;
    ex.blocks(end).x = 1;
    ex.blocks(end).debug = dbginfo('Root bias', root_y, root_x, root_t);
    root_app = parts(root).appMap(root_y, root_x, root_t);
    ex.blocks(end+1).i = root_p.appI;
    ex.blocks(end).x = root_app;
    ex.blocks(end).debug = dbginfo('Root appearance', root_y, root_x, root_t);
end

for child_k = 2:numparts
    child = parts(child_k);
    par_k = child.parent;
    
    par_x = ptr(par_k,1);
    par_y = ptr(par_k,2);
    par_t = ptr(par_k,3);
    assert(min([par_x par_y par_t]) > 0);
    
    ptr(child_k,1) = child.Ix(par_y,par_x,par_t);
    ptr(child_k,2) = child.Iy(par_y,par_x,par_t);
    ptr(child_k,3) = child.Im(par_y,par_x,par_t);
    
    box(child_k,:) = get_subpose_box(ptr(child_k, 1), ptr(child_k, 2), ...
        det_side, pyra.pad, scale);
    types(child_k) = ptr(child_k,3);
    
    if write
        child_x = ptr(child_k,1);
        child_y = ptr(child_k,2);
        child_t = ptr(child_k,3);
        
        % deformation
        assert(isscalar(child.gauI));
        ex.blocks(end+1).i = child.gauI;
        ex.blocks(end).x = defvector(child, child_x, child_y, par_x, par_y, child_t, par_t, sbin);
        ex.blocks(end).debug = dbginfo(sprintf('Subpose %i DG', child_k), ...
            child_y, child_x, child_t);
        
        % unary
        child_app = parts(child_k).appMap(child_y, child_x, child_t);
        ex.blocks(end+1).i = child.appI;
        ex.blocks(end).x = child_app;
        ex.blocks(end).debug = dbginfo(sprintf('Subpose %i appearance', child_k), ...
            child_y, child_x, child_t);
    end
end
end

function box = get_subpose_box(heatmap_x, heatmap_y, det_side, pad, scale)
% Grab the bounding box associated with a subpose detection at the given
% location in the output heatmap. det_side is cnn_edge_length / cnn_step,
% pad is the padding added to the original image before producing the
% heatmap, divided by the step, scale is the factor by which the heatmap
% needs to be multiplied to get back to the original image resolution.

% RV is in original image coordinates, so if the detection is perfect then
% orig_image(y1:y2, x1:x2, :) will select exactly the region containing the
% subpose.

x1 = (heatmap_x - 1 - pad)*scale+1;
y1 = (heatmap_y - 1 - pad)*scale+1;
x2 = x1 + det_side*scale - 1;
y2 = y1 + det_side*scale - 1;
box = [x1 y1 x2 y2];
end

function rv = dbginfo(msg, root_y, root_x, root_t)
rv = [msg sprintf(' (y=%i, x=%i, t=%i)', root_y, root_x, root_t)];
end

% Undo cropscale_pos transform to get box detections in original image
% coordinates
function dets = unscale_boxes(dets, xtrim, ytrim, scale)
% Each box is [x1, y1, x2, y2]
assert(isscalar(xtrim) && isscalar(ytrim) && isscalar(scale));
add_vec = [xtrim, ytrim, xtrim, ytrim];
% {dets.boxes} is a cell array of s*1 cell arrays, where s is the number of
% subposes.
scaled_boxes = cellfun(...
    @(bs) cellfun(@(b) (b / scale) + add_vec, bs, 'UniformOutput', false), ...
    {dets.boxes}, 'UniformOutput', false);
[dets.boxes] = deal(scaled_boxes{:});
end

% Update QP with coordinate descent
% and return the asociated model
function model = optimize(model)
global qp;
fprintf('.');
if qp.lb < 0 || qp.n == length(qp.a)
    qp_opt();
    qp_prune();
else
    qp_one();
end
model = vec2model(qp_w(), model);
end
