function [boxes,model,ex] = detect(im1_info, im2_info, pair, cnn_save_path, ...
    model, thresh, bbox, overlap, id, label)
% Detect objects in image using a model and a score threshold.
% Higher threshold leads to fewer detections.
%
% The function returns a matrix with one row per detected object.  The
% last column of each row gives the score of the detection.  The
% column before last specifies the component used for the detection.
% Each set of the first 4 columns specify the bounding box for a part
%
% If bbox is not empty, we pick best detection with significant overlap.
% If label is included, we write feature vectors to a global QP structure
%
% This function updates the model (by running the QP solver) if upper and
% lower bound differs
%
% detect() is called from two places in train.m. The signatures of those
% calls are:
%                         1       2      3   4   5  6  7
% 1) [box,model] = detect(neg(i), model, -1, [], 0, i, -1);
%                 1        2      3  4     5        6   7
% 2) box = detect(pos(ii), model, 0, bbox, overlap, ii, 1);

% XXX: Need to fix bounding boxes (sixz/sixy don't make sense any more, so
% I'll have to rethink how this is done; in heatmap, bbox edge for a part
% will probably just be center+-cnn_scale/(2*32)) and also fix all of the
% problems in passmsg.

INF = 1e10;

if nargin > 3 && ~isempty(bbox)
    latent = true;
    if label > 0
        thresh = -INF;
    end
else
    latent = false;
end

% Compute the feature pyramid and prepare filter
im1 = readim(im1_info);
im2 = readim(im2_info);
flow = imflow(im1_info.image_path, im2_info.image_path);
im_stack = cat(3, im1, im2);
% if has box information, crop it
if latent && label > 0
    % crop positive images to speed up latent search
    [im_stack, flow, bbox] = cropscale_pos(im_stack, flow, bbox, model.cnn.psize);
end

[pyra, unary_map] = imCNNdet(im_stack, flow, model);

if ~isempty(cnn_save_path)
    fprintf('Saving image pyramid and unary map to "%s"\n', cnn_save_path);
    save(cnn_save_path, 'pyra', 'unary_map');
end

levels = 1:length(pyra);

% Define global QP if we are writing features
% Randomize order to increase effectiveness of model updating
write = false;
if nargin > 5
    % Matlab was giving a warning that having global inside a block could
    % be "very inefficient" (might be about loops or something?)
    global qp; %#ok<TLEV>
    write  = true;
    levels = levels(randperm(length(levels)));
end
if nargin < 6
    id = 0;
end
if nargin < 7
    label = 0;
end

% Cache various statistics derived from model
[components, apps] = modelcomponents(model);

num_subposes = length(components);
assert(num_subposes > 1, 'Only %d parts?\n', num_subposes);
boxes = zeros(100000, 4 * length(components) + num_subposes + 1);
cnt = 0;

ex.blocks = [];
ex.id = [label id 0 0 0];

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
        skipflag = 0;
        for subpose_idx = 1:num_subposes
            % because all mixtures for one part is the same size, we only need to do this once
            % XXX: This doesn't make sense now that sixz/sixy don't make
            % sense.
            ovmask = testoverlap(components(subpose_idx).sizx(1), ...
                components(subpose_idx).sizy(1), sizs(1), sizs(2), ...
                pyra(level), bbox.xy(subpose_idx,:), overlap);
            if ~any(ovmask)
                skipflag = 1;
                break;
            end
        end
        if skipflag == 1
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
            ovmask = testoverlap(components(subpose_idx).sizx, components(subpose_idx).sizy, ...
                sizs(1), sizs(2), pyra(level), bbox.xy(subpose_idx,:), overlap);
            assert(ismatrix(ovmask));
            tmpscore = components(subpose_idx).score;
            tmpscore_K = size(tmpscore, 3);
            assert(tmpscore_K == model.K);
            ovmask = repmat(ovmask, 1, 1, tmpscore_K);
            assert(all(size(ovmask) == size(tmpscore)));
            % label supervision
            if label > 0
                % If a location doesn't overlap enough with the ground
                % truth, then we set it to -INF
                tmpscore(~ovmask) = -INF;
                
                % If a poselet is a long way from the GT poselet, then we
                % also set it to 0.
                near_pslts = pair.near{subpose_idx};
                assert(~isempty(near_pslts));
                assert(all(1 <= near_pslts & near_pslts <= model.K));
                far_pslts = true([1 model.K]);
                far_pslts(near_pslts) = false;
                assert(sum(far_pslts) == model.K - length(near_pslts));
                far_idxs = model.global_IDs{subpose_idx}(far_pslts);
                components(subpose_idx).appMap(:, :, far_idxs) = -INF;
            elseif label < 0
                tmpscore(ovmask) = -INF;
            end
            assert(all(size(components(subpose_idx).score) == size(tmpscore)));
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
    
    % Add bias to root score
    components(model.root).score = ...
        components(model.root).score + components(model.root).b;
    rscore = components(model.root).score;
    assert(ndims(rscore) == 3);
    
    % keep the positive example with the highest score in latent mode
    if latent && label > 0
        thresh = max(thresh, max(rscore(:)));
    end
    
    [Y, X, T] = ndfind(rscore >= thresh);
    % Walk back down tree following pointers
    % (DEBUG) Assert extracted feature re-produces score
    for i = 1:length(X)
        cnt = cnt + 1;
        x = X(i);
        y = Y(i);
        t = T(i);
        
        [box, types, ex] = ...
            backtrack(x, y, t, components, pyra(level), ex, write, model.sbin);
        
        boxes(cnt,:) = [box types rscore(y, x, t)];
        if write && (~latent || label < 0)
            qp_write(ex);
            qp.ub = qp.ub + qp.Cneg*max(1+rscore(y, x, t),0);
        elseif latent && label > 0
            if isempty(best_box)
                best_box = boxes(cnt,:);
                best_ex = ex;
            elseif best_box(end) < rscore(y, x, t)
                % update best
                best_box = boxes(cnt,:);
                best_ex = ex;
            end
        end
    end
    
    % Crucial DEBUG assertion:
    % If we're computing features, assert extracted feature re-produces
    % score (see qp_writ.m for computing original score)
    if write && (~latent || label < 0) && ~isempty(X) && qp.n < length(qp.a)
        w = -(qp.w + qp.w0.*qp.wreg) / qp.Cneg;
        assert(abs(score(w,qp.x,qp.n) - rscore(y,x,t)) < 1e-5);
    end
    
    % Optimize qp with coordinate descent, and update model
    if write && (~latent || label < 0) && ...
            (qp.lb < 0 || 1 - qp.lb/qp.ub > .05 || qp.n == length(qp.sv))
        model = optimize(model);
        [components, apps] = modelcomponents(model);
    end
end

boxes = boxes(1:cnt,:);

if latent && ~isempty(boxes) && label > 0
    boxes = best_box;
    if write
        qp_write(best_ex);
    end
end
end

% Backtrack through dynamic programming messages to estimate part locations
% and the associated feature vector
function [box,types,ex] = backtrack(x,y,t,parts,pyra,ex,write,sbin)
numparts = length(parts);
ptr = zeros(numparts,3);
box = zeros(numparts,4);
types = zeros(1, numparts);
root = 1;
p = parts(root);
ptr(root, :) = [x, y, t];
scale = pyra.scale;
% XXX: siz{x,y} thing is wrong
bb_start = [(x - 1 - pyra.pad)*scale+1, (y - 1 - pyra.pad)*scale+1];
bb_end = bb_start + [p.sizx*scale - 1, p.sizy*scale - 1];

box(root,:) = [bb_start bb_end];
types(root) = t;

% XXX: Need to verify usage of .blocks
% Pretty sure this is okay, since .blocks is just used in qp_write and the
% blocks are associated with a specific weight index (which I think is set
% correctly) so order shouldn't matter.
if write
    ex.id(3:6) = [p.level round(x+p.sizx/2) round(y+p.sizy/2) t];
    ex.blocks = [];
    ex.blocks(end+1).i = p.biasI;
    ex.blocks(end).x   = 1;
    f = parts(root).appMap(y, x, t);
    ex.blocks(end+1).i = p.appI;
    ex.blocks(end).x   = f;
end

for k = 2:numparts
    p_k   = parts(k);
    par = p_k.parent;
    
    x   = ptr(par,1);
    y   = ptr(par,2);
    t   = ptr(par,3);
    assert(min([x y t]) > 0);
    
    ptr(k,1) = p_k.Ix(y,x,t);
    ptr(k,2) = p_k.Iy(y,x,t);
    ptr(k,3) = p_k.Im(y,x,t);
    
    x1  = (ptr(k,1) - 1 - pyra.pad)*scale+1;
    y1  = (ptr(k,2) - 1 - pyra.pad)*scale+1;
    % XXX: sizx thing is wrong
    x2  = x1 + p_k.sizx*scale - 1;
    y2  = y1 + p_k.sizy*scale - 1;
    box(k,:) = [x1 y1 x2 y2];
    types(k) = ptr(k,3);
    
    if write        
        % deformation
        assert(isscalar(p_k.gauI));
        ex.blocks(end+1).i = p_k.gauI;
        ex.blocks(end).x   = defvector(p_k, ptr(k,1), ptr(k,2), x, y, ptr(k, 3), t, sbin);
        
        c_x = ptr(k,1);
        c_y = ptr(k,2);
        c_t = ptr(k,3);
        
        % unary
        f = parts(k).appMap(c_y, c_x, c_t);
        ex.blocks(end+1).i = p_k.appI;
        ex.blocks(end).x = f;
    end
end
box = reshape(box', 1, 4*numparts);
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